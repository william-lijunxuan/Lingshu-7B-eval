import json
from pathlib import Path


SRC_JSONL = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k.jsonl")
DST_JSONL = Path("/home/william/dataset/skin/Derm1M/eval_Derm1M_train_json_1k_clean.jsonl")

EXCLUDE_NAMES = {
    "edu__Leslie Baumann - Cosmetic Dermatology- Principles and Practice, Second Edition-McGraw-Hill Professional (2009)_00362_01634.png",
    "IIYI__310_5.png",
    "IIYI__398_2.png",
    "IIYI__463_3.png",
    "IIYI__888_5.png",
    "IIYI__1481_4.png",
    "IIYI__19046_3.png",
    "IIYI__23680_7.png",
    "IIYI__25793_8.png",
    "IIYI__25793_10.png",
    "IIYI__29551_4.png",
    "note__(James, Andrew-s Disease of the Skin) William D. James MD, Timothy Berger MD, Dirk Elston MD - Andrews- Diseases of the Skin- Clinical Dermatology -Saunders (2005)-OCR2_00048_00818_0.png",
    "pubmed__60_fc_PMC7083040_13071_2020_3989_Fig3_HTML.png",
    "youtube___AdCI0gdjSw_frame_18333_3.jpg",
    "youtube__0h6HijUUBDI_frame_5808_0.jpg",
    "youtube__3q-oDgu2bRM_frame_18542_0.jpg",
    "youtube__8nuikEynZRw_frame_4799_0.jpg",
    "youtube__8nuikEynZRw_frame_4799_1.jpg",
    "youtube__A501NloO1zo_frame_817_0_0.jpg",
    "youtube__A501NloO1zo_frame_818_0_0.jpg",
    "youtube__bR-1sfCV8J8_frame_2721_0.jpg",
    "youtube__bR-1sfCV8J8_frame_2721_1.jpg",
    "youtube__bPReTKZ-ubM_frame_16652_0.jpg",
    "youtube__Cmh27PQnZ2Y_frame_68_0.jpg",
    "youtube__Cmh27PQnZ2Y_frame_128_0_0.jpg",
    "youtube__Cv0R5xWXQLk_frame_1209_0_1.jpg",
    "youtube__Cv0R5xWXQLk_frame_1210_0_1.jpg",
    "youtube__ePJi5BJCt2g_frame_6653_4.jpg",
    "youtube__ePsccovofxw_frame_630_0_0.jpg",
    "youtube__E-Yl8SmpDTE_frame_54264_0.jpg",
    "youtube__FjzTv46aMVU_frame_28992_0_0.jpg",
    "youtube__fPVKwQz3xjQ_frame_522_2.jpg",
    "youtube__fPVKwQz3xjQ_frame_761_0_0.jpg",
    "youtube__FTmm6CHCg1Y_frame_6131_0.jpg",
    "youtube__H_4m5Auy1A4_frame_27668_0_0.jpg",
    "youtube__HSlDDGPU0Ko_frame_3647_3.jpg",
    "youtube__HSlDDGPU0Ko_frame_3652_1.jpg",
    "youtube__Iae-Oq4h3n0_frame_41521_0.jpg",
    "youtube__iZrp1daJWk0_frame_8128_3.jpg",
    "youtube__j2DJ84S0-bc_frame_82147_3.jpg",
    "youtube__LHleqLW9SeE_frame_4982_6.jpg",
    "youtube__lPrwmsnMAhA_frame_19910_2.jpg",
    "youtube__lPrwmsnMAhA_frame_19910_3.jpg",
    "youtube__LwIaNBbKVIE_frame_21663_0.jpg",
    "youtube__LYCLy4NOjAU_frame_35839_1.jpg",
    "youtube__m9dvsKwOnYk_frame_2034_0_0.jpg",
    "youtube__mP3RMAfe-WE_frame_1459_0_0.jpg",
    "youtube__NayiHIO2g6M_frame_0_0.jpg",
    "youtube__NayiHIO2g6M_frame_0_5.jpg",
    "youtube__O5Exoh6RmAE_frame_2604_0_0.jpg",
    "youtube__O5Exoh6RmAE_frame_3523_0_0.jpg",
    "youtube__O5Exoh6RmAE_frame_23805_0_0.jpg",
    "youtube__oHWaQZ7pdQU_frame_22319_0_0.jpg",
    "youtube__OPuoC_ke90o_frame_24211_0.jpg",
    "youtube__PUnFmMUxR5E_frame_23611_0.jpg",
    "youtube__R8kBKIU7zsk_frame_38228_0.jpg",
    "youtube__sIXqpeUxI80_frame_660_0_0.jpg",
    "youtube__SwgvZPEW1SQ_frame_122_2.jpg",
    "youtube__tDbEXNAtvSY_frame_8856_0.jpg",
    "youtube__TJs_ohAW6zw_frame_1400_3.jpg",
    "youtube__tRG3RZRYd6M_frame_16073_2.jpg",
    "youtube__TwGvoI0UTo0_frame_289_0_0.jpg",
    "youtube__tWwvBx_yUHU_frame_48782_0.jpg",
    "youtube__U2Irvmm1xjI_frame_5196_7.jpg",
    "youtube__UC1tgoWJmXY_frame_11644_0_0.jpg",
    "youtube__UXVvuhIuFn8_frame_4613_0_0.jpg",
    "youtube__VXkj2n2b77k_frame_1357_0_0.jpg",
    "youtube__VXkj2n2b77k_frame_1358_0_0.jpg",
    "youtube__W26mLqM0sm8_frame_15761_0_0.jpg",
    "youtube__wBGS7b4iRKs_frame_41525_1.jpg",
    "youtube__WBWpJQw1Tzw_frame_21758_0_0.jpg",
    "youtube__WBWpJQw1Tzw_frame_23715_0_0.jpg",
    "youtube__wFwNjR0Rogc_frame_128634_0.jpg",
    "youtube__wNgRo63rWvE_frame_86243_0.jpg",
    "youtube__XcT9xkkGcB4_frame_21039_0.jpg",
    "youtube__XcT9xkkGcB4_frame_21099_1.jpg",
    "youtube__XG2kyy31AVc_frame_35054_0_0.jpg",
    "youtube__XQikPSSVfvs_frame_1921_1.jpg",
    "youtube__XQikPSSVfvs_frame_1928_0.jpg",
    "youtube__XQikPSSVfvs_frame_4362_0.jpg",
    "youtube__xxmKbUPvK7Q_frame_13195_1.jpg",
    "youtube__zfYyheySci0_frame_177640_1.jpg",
    "youtube__zfYyheySci0_frame_177675_1.jpg",
    "youtube__ZSEU-POCCjA_frame_28500_1.jpg",
    "youtube__zvQV7xhlTXw_frame_14678_0_0.jpg",
    "6_7NLjYuWUw_frame_64004_1.jpg",
    "pubmed__35_e4_PMC5674723_abd_92_05_0748_g03.png",
    "pubmed__5e_c4_PMC10182843_403_2023_2627_Fig1_HTML_0.png",
    "IIYI__22639_2.png",
    "IIYI__29920_1.png",
    "IIYI__21153_5.png",
    "IIYI__21153_3.png",
    "IIYI__17687_5.png",
    "IIYI__17687_2.png",
    "IIYI__17481_1.png",
    "IIYI__9655_2.png",
    "IIYI__10033_4.png",
    "IIYI__9399_5.png",
    "IIYI__1912_5.png",
    "IIYI__1288_3.png",
    "edu__Neil S. Prose, Leonard Kristal - Weinberg’s Color Atlas of Pediatric Dermatology-McGraw-Hill Professional (2016)_00226_05475.png",
    "youtube__z8Z3z6d1v1M_frame_16651_0.jpg",
    "youtube__PfrzELmr6Qk_frame_2560_0_0.jpg",
    "youtube__BNmHFWTuvpI_frame_2817_0_0.jpg",
    "youtube__BNmHFWTuvpI_frame_2304_0_0.jpg",
    "youtube__AzCE5RPdYiE_frame_5013_5.jpg",
    "youtube__A9Mlgkpx9k8_frame_60935_0_0.jpg",
    "youtube__4jqnpkS_bvk_frame_63249_0_0.jpg",
    "youtube__4jqnpkS_bvk_frame_28731_1.jpg",
    "youtube__1-RON5mApb4_frame_15789_0_0.jpg",
    "pubmed__e9_8a_PMC5715227_gr1_0.jpg",
    "pubmed__1b_8d_PMC8251990_JOCD-20-5-g001_2.jpg",
    "pubmed__1b_8d_PMC8251990_JOCD-20-5-g001_1.jpg",
    "IIYI__13755_4.png",
    "IIYI__13755_8.png",
    "IIYI__13755_3.png",
    "IIYI__16987_2.png",
    "IIYI__28441_2.png",
    "twitter__9812_1.png",
    "youtube__1y8vX5CJzew_frame_1440_0.jpg",
    "youtube__3XYzeRz-GF8_frame_19780_0_0.jpg",
    "youtube__6_7NLjYuWUw_frame_64004_1.jpg",
    "youtube__6rCyewKh6tQ_frame_1247_1.jpg",
    "youtube__A9Mlgkpx9k8_frame_47838_0_0.jpg",
    "youtube__Aed7rRm4EcE_frame_176_0_0.jpg",
    "youtube__bdXQXJLDpMk_frame_2749_5.jpg",
    "youtube__c4abYpRoaBI_frame_391_0_0.jpg",
    "youtube__fPVKwQz3xjQ_frame_749_0.jpg",
    "youtube__KFDw2YDSXJw_frame_3039_1.jpg",
    "youtube__xxmKbUPvK7Q_frame_13195_3.jpg",
    "IIYI__25540_2.png",
    "IIYI__25933_5.png",
    "IIYI__22639_6.png",
    "IIYI__23905_2.png",
    "IIYI__23926_5.png",
    "IIYI__27823_1.png",
    "IIYI__29980_3.png",
    "IIYI__30776_6.png",
    "note__(James, Andrew-s Disease of the Skin) William D. James MD, Timothy Berger MD, Dirk Elston MD - Andrews- Diseases of the Skin- Clinical Dermatology -Saunders (2005)-OCR2_00294_04363_0.png",
    "note__Errichetti, Enzo-Ioannides, Dimitrios-Lallas, Aimilios - Dermoscopy in general dermatology-CRC Press (2019)_00068_25156_1.png",
    "youtube__EHC3ejfvcrY_frame_600_1.jpg",
    "youtube__EHC3ejfvcrY_frame_605_0_0.jpg",
    "youtube__YG61viZx2Pk_frame_22328_4.jpg",
}


def _normalize_path(p: str) -> str:
    return str(p).strip().replace("\\", "/")


def _flatten_image_name(rel_path: str) -> str:
    rel_path = _normalize_path(rel_path).lstrip("/")
    return rel_path.replace("/", "__")


def main() -> None:
    kept = 0
    removed = 0
    total = 0

    with SRC_JSONL.open("r", encoding="utf-8") as fin, DST_JSONL.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            img = obj.get("image", "")
            if not img:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
                continue

            raw = _normalize_path(img)
            flat = _flatten_image_name(raw)

            if raw in EXCLUDE_NAMES or flat in EXCLUDE_NAMES:
                removed += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"total={total} kept={kept} removed={removed} out={DST_JSONL}")


if __name__ == "__main__":
    main()

