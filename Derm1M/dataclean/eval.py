import unicodedata
import re
def _norm(s: str) -> str:
    """Lightweight medical-term normalization."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    # British ↔ American variants
    s = s.replace("naevus", "nevus").replace("naevi", "nevi").replace("haemangioma", "hemangioma")
    # Common abbreviations
    s = s.replace(" pih", " post-inflammatory hyperpigmentation").replace("pih", "post-inflammatory hyperpigmentation")
    s = s.replace("bcc", "basal cell carcinoma").replace("scc", "squamous cell carcinoma").replace(" sk", " seborrheic keratosis")
    s = s.replace(" ak", " actinic keratosis")
    # Hyphens → spaces, collapse non-alnum
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _alias_map():
    """Alias → canonical string map."""
    groups = [
        # exact pairs requested
        {"scar", "scarring", "cicatrix"},
        {"melanocytic nevus", "melanocytic nevi", "nevus", "nevi", "mole", "moles", "naevus", "naevi"},
        {"epidermal cyst", "epidermoid cyst", "sebaceous cyst", "infundibular cyst"},
        {"angioma", "hemangioma", "cherry angioma", "senile hemangioma"},
        {"acrochordon", "skin tag", "skin tags"},
        # useful dermatology aliases
        {"basal cell carcinoma", "bcc"},
        {"squamous cell carcinoma", "scc"},
        {"seborrheic keratosis", "sk"},
        {"actinic keratosis", "ak"},
        {"post inflammatory hyperpigmentation", "post-inflammatory hyperpigmentation", "pih"},
        {"melasma", "chloasma"},
        {"tinea", "dermatophytosis", "ringworm"},
        {"molluscum contagiosum", "molluscum"},
        {"urticaria", "hives"},
        {"acne vulgaris", "acne"},
        {"psoriasis", "psoriasis vulgaris"},
        {"vitiligo"},
        {"lichen planus"},
        {"impetigo"},
        {"folliculitis"},
        {"paronychia"},
        {"onychomycosis"},
        {"cellulitis"},
        {"abscess"},
        {"rosacea"},
        {"viral wart", "wart", "verruca", "hpv wart"},
        {"Behçet's syndrome", "behcets disease", "behcets","Behçet syndrome"},
        {"hemangioma", "haemangioma"},  # safety duplicate
        {"Tinea capitis", "tinea"},
        {"hypertrichosis", "localized hypertrichosis"},
        {"hypertrichosis", "localized hypertrichosis", "faun tail nevus", "faun tail"},
        {"tinea", "tinea capitis", "dermatophytosis", "ringworm"},
        {"amyloidosis", "cutaneous amyloidosis"},
        {"mucinosis", "follicular mucinosis"},
        {"pigmented progressive purpuric dermatosis", "pigmentary purpuric dermatosis", "purpura pigmentosa chronica"},
        {"scar", "scar condition", "scarring", "cicatrix"},
        {"angiofibroma", "facial angiofibromas"},
    ]
    alias2canon = {}
    for g in groups:
        # choose a stable canonical: first sorted normalized term
        norm_group = sorted(_norm(x) for x in g if _norm(x))
        if not norm_group:
            continue
        canon = norm_group[0]
        for x in norm_group:
            alias2canon[x] = canon
    return alias2canon

_ALIAS2CANON = _alias_map()

def _canonical(term: str) -> str:
    n = _norm(term)
    return _ALIAS2CANON.get(n, n)

def judge_close_end_vqa_json(answer: str, response: str) -> bool:
    """
    Return True if `answer` matches response['answer'] exactly after normalization,
    or if both map to the same canonical term via alias groups; otherwise False.
    """
    try:
        pred = json.loads(response).get("answer", None)
    except Exception:
        # try to recover if response has leading/trailing noise
        try:
            start = response.find("{")
            end = response.rfind("}")
            pred = json.loads(response[start:end+1]).get("answer", None) if start != -1 and end != -1 else None
        except Exception:
            pred = None

    if pred is None:
        return False

    a_norm = _norm(answer)
    p_norm = _norm(pred)
    if a_norm == p_norm:
        return True

    a_can = _canonical(a_norm)
    p_can = _canonical(p_norm)
    return a_can == p_can

import json
import os

# Assume _norm, _alias_map, _ALIAS2CANON, _canonical, judge_close_end_vqa_json
# are already defined exactly as you provided (do not modify them).

RESULTS_PATH = r"\home\william\dataset\skin\Derm1M\1k\hulu_results.json"


def _to_posix_path(p: str) -> str:
    p = p.strip()
    if p.startswith("\\home\\") or p.startswith("\home\\"):
        p = "/" + p.lstrip("\\/")
    return p.replace("\\", "/")


def main():
    path = _to_posix_path(RESULTS_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    skipped = 0

    for item in data:
        answer = item.get("answer", None)
        response = item.get("response", None)
        if answer is None or response is None:
            skipped += 1
            continue

        total += 1
        if judge_close_end_vqa_json(answer, response):
            correct += 1

    acc = correct / total if total else 0.0
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"accuracy={acc:.6f}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
