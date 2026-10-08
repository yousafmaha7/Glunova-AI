"""
Tests for report_extraction.py

The report texts below are SYNTHETIC test inputs written to exercise the
parsing rules (layouts, units, OCR noise). They are not patient data and not
experimental results. Add anonymised lines from real reports here whenever a
new lab layout is found, so the rules are checked against them.

Run:
    python test_report_extraction.py
"""

from report_extraction import FileText, parse_texts


def parse(*texts):
    return parse_texts([FileText(f"file{i}.txt", t, "test")
                        for i, t in enumerate(texts)])


def value(ex, key):
    f = ex.findings.get(key)
    return None if f is None else f.value


def close(a, b, tol=1e-3):
    return a is not None and abs(a - b) <= tol


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def cbc_typical_table():
    ex = parse("""
    COMPLETE BLOOD COUNT
    Test               Result   Unit          Reference Range
    Haemoglobin        11.5     g/dL          12.0 - 16.0
    RBC Count          4.21     x10^6/uL      3.8 - 5.2
    TLC                9.1      x10^3/uL      4.0 - 11.0
    MCH                27.3     pg            27 - 32
    MCHC               33.0     g/dL          32 - 36
    Neutrophils        62       %             40 - 75
    Lymphocytes        30       %             20 - 45
    Eosinophils        3        %             1 - 6
    """)
    assert close(value(ex, "hb"), 11.5)
    assert close(value(ex, "rbc"), 4.21)
    assert close(value(ex, "wbc"), 9.1)
    assert close(value(ex, "mchc"), 33.0)
    assert close(value(ex, "lymphocytes"), 30 * 9.1 / 100)
    assert close(value(ex, "eosinophils"), 3 * 9.1 / 100)
    assert ex.findings["lymphocytes"].status == "calculated"


@case
def cbc_units_inside_label_and_absolute_counts():
    ex = parse("""
    Hb (g/dl) 10.8
    WBC (10^3/uL) 7.6
    RBC (millions/cumm) 3.95
    Lymphocytes (Absolute count 10^3/uL) 2.15
    Eosinophils (Absolute count 10^3/uL) 0.22
    """)
    assert close(value(ex, "hb"), 10.8)
    assert close(value(ex, "wbc"), 7.6)
    assert close(value(ex, "rbc"), 3.95)
    assert close(value(ex, "lymphocytes"), 2.15)
    assert close(value(ex, "eosinophils"), 0.22)


@case
def analyser_printout_percent_and_absolute():
    ex = parse("""
    WBC   8.40  10^3/uL
    LYM%  28.1  %
    LYM#  2.36  10^3/uL
    EOS%  2.5   %
    EOS#  0.21  10^3/uL
    HGB   12.1  g/dL
    """)
    assert close(value(ex, "lymphocytes"), 2.36)
    assert close(value(ex, "eosinophils"), 0.21)
    assert close(value(ex, "hb"), 12.1)


@case
def counts_per_cubic_mm_are_converted():
    ex = parse("""
    Total Leucocyte Count   9,100 /cumm   4000-11000
    Red Blood Cells   4,250,000 /cumm
    Haemoglobin  118 g/L
    Absolute Lymphocyte Count  2700 cells/uL
    """)
    assert close(value(ex, "wbc"), 9.1)
    assert close(value(ex, "rbc"), 4.25)
    assert close(value(ex, "hb"), 11.8)
    assert close(value(ex, "lymphocytes"), 2.7)
    assert ex.findings["wbc"].status == "converted"


@case
def mch_line_is_not_read_as_haemoglobin():
    ex = parse("""
    MCH (Mean Corpuscular Haemoglobin)   28.0 pg
    MCHC (Mean Corpuscular Haemoglobin Concentration)  33.4 g/dl
    HbA1c   5.4 %
    """)
    assert value(ex, "hb") is None
    assert close(value(ex, "mchc"), 33.4)


@case
def fasting_glucose_mgdl_and_mmol():
    ex = parse("Blood Sugar Fasting   92 mg/dl   70-110")
    assert close(value(ex, "fasting_glucose"), 92)
    ex = parse("Fasting Plasma Glucose  5.1 mmol/L")
    assert close(value(ex, "fasting_glucose"), 5.1 * 18.016)
    assert ex.findings["fasting_glucose"].status == "converted"


@case
def random_and_postprandial_glucose_are_ignored():
    ex = parse("""
    Random Blood Sugar    140 mg/dl
    Glucose 2 hr post 75g   153 mg/dl
    Fasting status: 12 hours
    """)
    assert value(ex, "fasting_glucose") is None


@case
def ogtt_fasting_line():
    ex = parse("""
    OGTT (75 g)
    Glucose (Fasting)   95 mg/dl
    Glucose 1 hour      182 mg/dl
    Glucose 2 hour      150 mg/dl
    """)
    assert close(value(ex, "fasting_glucose"), 95)


@case
def antenatal_card():
    ex = parse("""
    Name: XXXX        Age: 28 Years     G2 P1 A0
    Gestational age: 26 weeks
    Blood Group: B +ve
    Family history: DM in mother
    Weight 68 kg     Height 160 cm
    Visit 1  BP 120/80 mmHg
    Visit 2  BP 110/70 mmHg
    """)
    assert close(value(ex, "age"), 28)
    assert close(value(ex, "gravidity"), 2)
    assert value(ex, "blood_type") == "B"
    assert value(ex, "family_history") == "Yes"
    assert close(value(ex, "bmi"), 68 / 1.6 ** 2)
    assert close(value(ex, "systolic"), 115)
    assert close(value(ex, "diastolic"), 75)
    assert ex.findings["bmi"].status == "calculated"


@case
def antenatal_card_other_forms():
    ex = parse("""
    Age/Sex: 31Y/F
    Primigravida
    ABO & Rh: O Positive
    No family history of diabetes
    BMI: 27.4 kg/m2
    Height: 5'3"   Weight: 65 kg
    """)
    assert close(value(ex, "age"), 31)
    assert close(value(ex, "gravidity"), 1)
    assert value(ex, "blood_type") == "O"
    assert value(ex, "family_history") == "No"
    assert close(value(ex, "bmi"), 27.4)       # printed BMI wins


@case
def blood_group_variants():
    for text, expected in [
        ("Blood Group: AB+", "AB"),
        ("Blood group : A Negative", "A"),
        ("Blood Group  0 +ve", "O"),          # OCR reads O as zero
        ("Blood Group: Not done", None),
    ]:
        assert value(parse(text), "blood_type") == expected, text


@case
def ocr_noise_in_numbers():
    ex = parse("Haemoglobin 1O.5 g/dl\nWBC9.l")
    assert close(value(ex, "hb"), 10.5)


@case
def value_on_next_line():
    ex = parse("Haemoglobin\n11.2 g/dl")
    assert close(value(ex, "hb"), 11.2)


@case
def conflicting_values_are_reported():
    ex = parse("Haemoglobin 11.5 g/dl", "Hb 10.2 g/dl")
    assert close(value(ex, "hb"), 11.5)
    assert len(ex.conflicts["hb"]) == 2


@case
def percentage_without_wbc_is_not_used():
    ex = parse("Lymphocytes 30 %")
    assert value(ex, "lymphocytes") is None
    assert "lymphocytes" in ex.unusable


@case
def unclear_unit_is_rejected():
    ex = parse("WBC 350")
    assert value(ex, "wbc") is None
    assert "wbc" in ex.unusable


@case
def gestational_age_and_g6pd_not_confused():
    ex = parse("Gestational age 32 weeks\nG6PD: Normal")
    assert value(ex, "age") is None
    assert value(ex, "gravidity") is None


@case
def broken_table_rows_from_pdf_text_layer():
    ex = parse("WBC 7.8\nLymphocytes 31\n% 20-45\nEosinophils\n2\n% 1-6")
    assert close(value(ex, "lymphocytes"), 31 * 7.8 / 100)
    assert close(value(ex, "eosinophils"), 2 * 7.8 / 100)
    assert "%" in ex.findings["eosinophils"].line


@case
def column_wise_layout_is_not_guessed():
    # Labels and values in separate blocks: pairing them would be guesswork.
    ex = parse("Haemoglobin\nRBC\nWBC\n11.4\n4.12\n9.6")
    assert value(ex, "wbc") is None
    assert value(ex, "hb") is None


if __name__ == "__main__":
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}  {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    raise SystemExit(1 if failed else 0)
