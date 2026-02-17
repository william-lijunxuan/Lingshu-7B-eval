import csv
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


CSV_PATH = Path("/home/william/dataset/skin/Derm1M/Derm1M_v2_pretrain_HD.csv")


def norm_term(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return s


def split_hd_path(hd: str):
    if hd is None:
        return []
    hd = str(hd).strip()
    if not hd:
        return []
    parts = [p.strip() for p in hd.split(">") if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in hd.split("/") if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in hd.split("|") if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in hd.split("->") if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in re.split(r"[;,]", hd) if p.strip()]
    return parts


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(str(CSV_PATH))

    disease_labels = set()
    hierarchical_leaf_labels = set()

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames or []}

        dl_col = None
        for key in ["disease_label", "label", "answer", "disease"]:
            if key in cols:
                dl_col = cols[key]
                break

        hd_col = None
        for key in ["hierarchical_disease_label", "hierarchical_label", "hd_label", "disease_label_hierarchical"]:
            if key in cols:
                hd_col = cols[key]
                break

        if dl_col is None and hd_col is None:
            raise ValueError(f"Cannot find disease_label or hierarchical_disease_label columns in: {reader.fieldnames}")

        for row in reader:
            if dl_col is not None:
                v = norm_term(row.get(dl_col, ""))
                if v:
                    disease_labels.add(v)

            if hd_col is not None:
                v = norm_term(row.get(hd_col, ""))
                if v:
                    parts = split_hd_path(v)
                    if parts:
                        hierarchical_leaf_labels.add(parts[-1])

    # Baseline groups (you can extend here; output will include only terms that exist in the CSV)
    base_groups = [
        {"scar", "scarring", "cicatrix"},
        {"melanocytic nevus", "melanocytic nevi", "nevus", "nevi", "mole", "moles", "naevus", "naevi"},
        {"epidermal cyst", "epidermoid cyst", "sebaceous cyst", "infundibular cyst"},
        {"angioma", "hemangioma", "cherry angioma", "senile hemangioma"},
        {"acrochordon", "skin tag", "skin tags"},
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
        {"Behçet's syndrome", "behcets disease", "behcets", "Behçet syndrome"},
        {"hemangioma", "haemangioma"},
        {"Tinea capitis", "tinea"},
        {"hypertrichosis", "localized hypertrichosis"},
    ]

    # Keep only terms present in disease_label OR hierarchical leaf labels
    present = {t.lower() for t in disease_labels} | {t.lower() for t in hierarchical_leaf_labels}

    filtered_groups = []
    for g in base_groups:
        kept = []
        for term in g:
            term_norm = norm_term(term)
            if term_norm and term_norm.lower() in present:
                kept.append(term_norm)
        if len(kept) >= 2:
            filtered_groups.append(set(sorted(kept, key=lambda x: x.lower())))
        elif len(kept) == 1:
            filtered_groups.append(set(kept))

    # Merge duplicates (same set)
    uniq = []
    seen = set()
    for g in filtered_groups:
        key = tuple(sorted([x.lower() for x in g]))
        if key not in seen:
            seen.add(key)
            uniq.append(g)

    def fmt_set(s):
        items = sorted(list(s), key=lambda x: x.lower())
        inner = ", ".join([repr(x) for x in items])
        return "{" + inner + "}"

    print("groups = [")
    print("    # exact pairs requested")
    for g in uniq[:5]:
        print(f"    {fmt_set(g)},")
    print("    # useful dermatology aliases")
    for g in uniq[5:]:
        print(f"    {fmt_set(g)},")
    print("]")

    print()
    print(f"# disease_label unique: {len(disease_labels)}")
    print(f"# hierarchical leaf unique: {len(hierarchical_leaf_labels)}")


if __name__ == "__main__":
    main()
