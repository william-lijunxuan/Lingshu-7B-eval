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
        {"scar", "scarring", "cicatrix", "scar condition"},
        {"melanocytic nevus", "melanocytic nevi", "nevus", "nevi", "mole", "moles", "naevus", "naevi"},
        {"epidermal cyst", "epidermoid cyst", "sebaceous cyst", "infundibular cyst"},
        {"angioma", "hemangioma", "cherry angioma", "senile hemangioma", "haemangioma"},
        {"acrochordon", "skin tag", "skin tags"},

        # useful dermatology aliases
        {"basal cell carcinoma", "bcc"},
        {"squamous cell carcinoma", "scc"},
        {"seborrheic keratosis", "sk"},
        {"actinic keratosis", "ak"},
        {"post inflammatory hyperpigmentation", "post-inflammatory hyperpigmentation", "pih"},
        {"melasma", "chloasma"},
        {"tinea", "tinea capitis", "dermatophytosis", "ringworm"},
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
        {"Behçet's syndrome", "behcets disease", "behcets", "Behçet syndrome"},

        # hypertrichosis family
        {"hypertrichosis", "localized hypertrichosis", "faun tail nevus", "faun tail"},

        # amyloid / mucin
        {"amyloidosis", "cutaneous amyloidosis"},
        {"mucinosis", "follicular mucinosis"},

        # purpura variants
        {"pigmented progressive purpuric dermatosis", "pigmentary purpuric dermatosis", "purpura pigmentosa chronica"},

        # angiofibroma variants
        {"angiofibroma", "facial angiofibromas"},

        # drug eruption variants (CSV specific)
        {"drug eruption", "drug eruptions & reactions"},

        # grover disease
        {"grover's disease", "transient acantholytic dermatosis"},

        # puppp
        {"puppp", "pruritic urticarial papules and plaques of pregnancy"},

        # lichen simplex
        {"lichen simplex chronicus", "neurodermatitis"},

        # keratoderma
        {"keratoderma", "hyperkeratosis palmaris et plantaris"},

        # wound infection descriptive leaf
        {"local infection of wound", "abrasion, local infection of wound"},

        # cafe spelling
        {"café au lait macule", "cafe au lait macule"},

        # Behçet split-token variant seen in preds
        {"beh et s syndrome", "Behçet's syndrome", "behcets disease", "behcets", "Behçet syndrome"},

        # plural/singular
        {"freckle", "freckles"},

        # Kaposi punctuation/apostrophe variants
        {"kaposi sarcoma", "kaposi's sarcoma", "kaposi s sarcoma"},

        # striae synonyms
        {"striae", "stretch mark", "stretch marks"},

        # mucocele synonyms
        {"mucocele", "mucous cyst", "mucous gland cyst"},

        # HSV wording
        {"herpes simplex", "herpes simplex virus", "hsv"},

        # AD/eczema wording
        {"atopic dermatitis", "eczema", "atopic eczema"},

        # slapped cheek / parvovirus B19
        {"parvovirus b19 infection", "slapped cheek syndrome", "fifth disease"},

        # angioma common name variant
        {"angioma", "strawberry nevus", "strawberry haemangioma", "strawberry hemangioma"},

        # seborrheic keratosis plural
        {"seborrheic keratosis", "seborrheic keratoses"},

        # cutaneous larva migrans naming
        {"cutaneous larva migrans", "creeping eruption", "sand worm eruption"},
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

# Put this near your alias definitions
PARENT_MAP = {
    # dermatitis hierarchy
    "dermatitis": {
        "seborrheic dermatitis",
        "atopic dermatitis",
        "allergic contact dermatitis",
        "irritant contact dermatitis",
        "follicular mite dermatitis",
    },
    "contact dermatitis": {
        "allergic contact dermatitis",
        "irritant contact dermatitis",
    },
    # vascular tumor hierarchy
    "angioma": {
        "infantile hemangioma",
        "strawberry hemangioma",
        "strawberry haemangioma",
        "hemangioma",
    },
    "hemangioma": {
        "infantile hemangioma",
        "strawberry hemangioma",
        "strawberry haemangioma",
    },
    # psoriasis hierarchy
    "psoriasis": {
        "oral psoriasis",
        "psoriasis vulgaris",
    },
    # alopecia hierarchy
    "alopecia": {
        "alopecia areata",
        "androgenetic alopecia",
    },
    # ichthyosis hierarchy
    "ichthyosis": {
        "lamellar ichthyosis",
        "autosomal recessive congenital ichthyosis",
        "lamellar ichthyosis",
        "x linked ichthyosis",
    },
    # wart hierarchy
    "warts": {
        "flat wart",
        "verruca vulgaris",
        "viral wart",
    },
    # scar hierarchy
    "scar": {
        "atrophic scar",
        "hypertrophic scar",
        "cicatrix",
        "scarring",
    },
    "cicatrix": {
        "atrophic scar",
        "hypertrophic scar",
        "scar",
        "scarring",
    },
    # nail disease hierarchy
    "nail disease": {
        "onychomycosis",
    },
    # tinea hierarchy (if you allow child-of-tinea)
    "tinea": {
        "tinea capitis",
        "tinea corporis",
        "tinea cruris",
        "tinea pedis",
        "tinea unguium",
    },
    # mucocele naming is already alias, but keep here harmless
    "mucocele": {
        "mucous cyst",
        "mucous gland cyst",
    },
    "erythema dyschromicum perstans": {
        "follicular erythema dyschromicum perstans",
    },
    "abscess": {
        "deep seated abscess",
    },

    # basal cell carcinoma hierarchy
    "basal cell carcinoma": {
        "superficial basal cell carcinoma",
        "nodular basal cell carcinoma",
        "pigmented basal cell carcinoma",
    },
}

# Normalize keys once for safety
PARENT_MAP = {_norm(k): {_norm(x) for x in v} for k, v in PARENT_MAP.items()}



def _extract_pred_answer(response):
    if response is None:
        return None

    if isinstance(response, dict):
        return response.get("answer", None)

    if isinstance(response, list):
        return None

    if isinstance(response, (bytes, bytearray)):
        response = response.decode("utf-8", errors="ignore")

    if not isinstance(response, str):
        response = str(response)

    s = response.strip()

    # remove markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj.get("answer", None)
        return None
    except Exception:
        pass

    # try to decode the first JSON object/array from the first "{" or "["
    m = re.search(r"[\{\[]", s)
    if not m:
        return None
    s2 = s[m.start():]

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s2)
        if isinstance(obj, dict):
            return obj.get("answer", None)
        return None
    except Exception:
        # last resort: try substring between first { and last }
        start = s2.find("{")
        end = s2.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = s2[start:end+1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj.get("answer", None)
            except Exception:
                return None
        return None


def judge_close_end_vqa_json(answer: str, response: str) -> bool:
    pred = _extract_pred_answer(response)
    if pred is None:
        print("answer:", answer, "pred:", pred,"response",response)
        return False

    a_norm = _norm(answer)
    p_norm = _norm(pred)

    # exact match after normalization
    if a_norm == p_norm:
        return True

    # alias canonical match
    a_can = _canonical(a_norm)
    p_can = _canonical(p_norm)
    if a_can == p_can:
        return True

    # hierarchical match: answer is parent, pred is child (after canonicalization)
    # we apply canonical here to reduce variant issues
    a_h = _canonical(a_can)
    p_h = _canonical(p_can)

    children = PARENT_MAP.get(a_h)
    if children and p_h in children:
        return True

    # print("answer:", a_h, "pre:", p_h, False)
    # print("answer:",answer,"pred:",pred)
    return False


import json
import os

# Assume _norm, _alias_map, _ALIAS2CANON, _canonical, judge_close_end_vqa_json
# are already defined exactly as you provided (do not modify them).

RESULTS_PATH = r"\home\william\dataset\skin\Derm1M\1k\medresults.json"


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
