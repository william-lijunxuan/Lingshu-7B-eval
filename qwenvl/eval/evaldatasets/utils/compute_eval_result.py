import json
import re
import statistics
from pathlib import Path
from typing import List, Optional, Tuple

from nltk.translate.meteor_score import meteor_score
from bert_score.scorer import BERTScorer

JSON_PATH = Path(
    "/mnt/d/skinalor/model/Lingshu-7B-eval/qwenvl/eval/eval_results/Lingshu-7B/20251123_115421/Derm1m0Baseline/results_Lingshu-7B_baseline.json"
    # "/mnt/d/skinalor/model/Lingshu-7B-eval/qwenvl/eval/eval_results/Lingshu-7B/20251123_115421/Derm1m0Baseline/results_Lingshu-7B_finetuning.json"
)


def extract_model_answer(response_text: Optional[str]) -> Optional[str]:
    """
    Extract the `answer` field from the model response string.

    The response is expected to contain a JSON-like object text, optionally
    wrapped in markdown fences like ```json ... ```.
    Tries JSON parsing first, then falls back to regex.
    """
    if not response_text:
        return None

    text = str(response_text).strip()

    # Strip markdown fences such as ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    # Try to parse as JSON first
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            obj = json.loads(json_str)
            if isinstance(obj, dict) and "answer" in obj:
                ans = obj["answer"]
                if isinstance(ans, str):
                    return ans.strip().lower()
    except Exception:
        # Fall back to regex if JSON parsing fails
        pass

    # Regex fallback: match "answer": "..." before "top3"
    m = re.search(
        r'"answer"\s*:\s*"(.*?)"\s*,\s*\n\s*"top3"',
        text,
        flags=re.DOTALL,
    )
    if not m:
        return None

    answer = m.group(1).strip()
    return answer.lower()


def load_pairs(
    path: Path,
) -> Tuple[List[str], List[str], List[bool], int]:
    """
    Load reference–candidate answer pairs from the JSON file.

    Returns:
        refs: list of gold answers (length == total_records)
        cands: list of model answers (length == total_records)
        valid_mask: list of bools, True if this record has both answers
        total_records: total number of records in the json file
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data)

    refs: List[str] = []
    cands: List[str] = []
    valid_mask: List[bool] = []

    for item in data:
        gold_raw = item.get("answer", None)
        gold_str = str(gold_raw).strip().lower() if gold_raw is not None else ""

        pred_raw = item.get("response", "")
        pred = extract_model_answer(pred_raw)

        if not gold_str or not pred:
            refs.append("")
            cands.append("")
            valid_mask.append(False)
        else:
            refs.append(gold_str)
            cands.append(pred)
            valid_mask.append(True)

    return refs, cands, valid_mask, total_records


def main() -> None:
    refs, cands, valid_mask, total_records = load_pairs(JSON_PATH)

    print(f"Total records in json: {total_records}")
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    print(f"Valid pairs for evaluation: {len(valid_indices)}")

    if not valid_indices:
        print("No valid pairs found. Abort.")
        return

    # Compact lists for scoring (only valid pairs)
    refs_valid = [refs[i] for i in valid_indices]
    cands_valid = [cands[i] for i in valid_indices]

    # ===== BERTScore =====
    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=False,
        idf=False,
        batch_size=64,
        nthreads=4,
    )

    # cands_valid: model outputs, refs_valid: gold
    P_valid, R_valid, F_valid = scorer.score(cands_valid, refs_valid)

    # ===== METEOR (per valid pair) =====
    meteor_sum = 0.0
    meteor_scores_valid: List[float] = []
    for ref, cand in zip(refs_valid, cands_valid):
        score = meteor_score([ref.split()], cand.split())
        meteor_scores_valid.append(score)
        meteor_sum += score

    # ===== Per-example printing =====
    print("\nPer-example scores (valid pairs):")
    for k, idx in enumerate(valid_indices):
        p = P_valid[k].item()
        r = R_valid[k].item()
        f = F_valid[k].item()
        m = meteor_scores_valid[k]

        gold = refs[idx]
        pred = cands[idx]

        gold_short = gold.replace("\n", " ")[:120]
        pred_short = pred.replace("\n", " ")[:120]

        print(
            f"Index {idx:4d} | "
            f"P={p:.4f}, R={r:.4f}, F={f:.4f}, METEOR={m:.4f}"
        )
        print(f"  GOLD: {gold_short}")
        print(f"  PRED: {pred_short}")
        print()

    # ===== Averages =====
    # Average over valid pairs only
    avg_p_valid = P_valid.mean().item()
    avg_r_valid = R_valid.mean().item()
    avg_f_valid = F_valid.mean().item()
    avg_meteor_valid = statistics.mean(meteor_scores_valid)

    # Average over all records (invalid ones treated as 0)
    sum_p = P_valid.sum().item()
    sum_r = R_valid.sum().item()
    sum_f = F_valid.sum().item()
    avg_p_total = sum_p / total_records
    avg_r_total = sum_r / total_records
    avg_f_total = sum_f / total_records
    avg_meteor_total = meteor_sum / total_records

    print("\nBERTScore (average over valid pairs):")
    print(f"  P_valid: {avg_p_valid:.4f}")
    print(f"  R_valid: {avg_r_valid:.4f}")
    print(f"  F_valid: {avg_f_valid:.4f}")

    print("BERTScore (average over ALL records, using total_records):")
    print(f"  P_total: {avg_p_total:.4f}")
    print(f"  R_total: {avg_r_total:.4f}")
    print(f"  F_total: {avg_f_total:.4f}")

    print("\nMETEOR:")
    print(f"  Average METEOR over valid pairs: {avg_meteor_valid:.4f}")
    print(f"  Average METEOR over ALL records: {avg_meteor_total:.4f}")


if __name__ == "__main__":
    main()
