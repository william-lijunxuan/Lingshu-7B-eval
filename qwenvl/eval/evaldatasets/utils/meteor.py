from utils import calculate_meteor
path =".log"

import json
from pathlib import Path

def read_pairs_from_results_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    preds, gts = [], []
    for item in data:
        pred = item.get("response", "")
        gt = ""
        for turn in item.get("conversations", []):
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                gt = turn.get("value", "")
        preds.append(pred)
        gts.append(gt)
    return preds, gts


# def main():
#     json_path="/home/william/model/Lingshu-7B-eval/qwenvl/eval/eval_results/Lingshu-7B/20251103_165053/Derm1m0Baseline/results.json"
#     predictions, ground_truths = read_pairs_from_results_json(Path(json_path))
#
#     n = min(len(predictions), len(ground_truths))
#     predictions = predictions[:n]
#     ground_truths = ground_truths[:n]
#
#     meteor = calculate_meteor(predictions, ground_truths)
#     print("========== METEOR ==========")
#     print(f"File: {json_path}")
#     print(f"Pairs: {n}")
#     print(f"METEOR (avg): {meteor:.6f}")

def main():
    # json_path = "/home/william/model/Lingshu-7B-eval/qwenvl/eval/eval_results/Lingshu-7B/20251031_170556/Derm1m0Baseline/results.json" #finetuning 0.128322
    json_path = "/home/william/model/Lingshu-7B-eval/qwenvl/eval/eval_results/Lingshu-7B/20251103_150525/Derm1m0Baseline/results.json" # baseline  0.099707
    predictions, ground_truths = read_pairs_from_results_json(Path(json_path))

    meteor_avg = calculate_meteor(predictions, ground_truths)

    print("\n========== METEOR SUMMARY ==========")
    print(f"File: {json_path}")
    print(f"METEOR (avg via calculate_meteor): {meteor_avg:.6f}")



if __name__ == "__main__":
    main()