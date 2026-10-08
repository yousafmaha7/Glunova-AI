"""
============================================================
          GLUNOVA AI - REPORT READING (LOCAL OCR)
============================================================

Reads antenatal and laboratory reports (PDF or photo) on the server and
extracts candidate values for the 14 Glunova AI inputs.

Design rules
------------
* Everything runs locally. PDFs with a text layer are read directly
  (pypdfium2); photos and scanned pages are read with Tesseract OCR.
  No data is sent to any external service.
* The result only PRE-FILLS the assessment form. The user must check every
  value before the assessment is run.
* Every value keeps the file and line it was read from, and every unit
  conversion or calculation is written into a note, so it can be checked.
* When a value is ambiguous it is flagged or left empty. An empty field is
  safer than a wrong value.

Public API
----------
    ocr_available() -> bool
    read_reports([(file_name, file_bytes), ...]) -> Extraction
    summarise_for_form(extraction, input_ranges) -> dict
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

from PIL import Image, ImageOps, ImageSequence


# ============================================================
# SETTINGS
# ============================================================

MAX_PAGES_PER_FILE = 10          # bounds OCR time per upload
MIN_TEXT_LAYER_CHARS = 40        # below this a PDF page is treated as scanned
OCR_RENDER_DPI = 300
OCR_MIN_WIDTH = 1800             # small photos are upscaled before OCR
TESSERACT_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

# 1 mmol/L glucose = 18.016 mg/dL (molar mass of glucose 180.16 g/mol).
GLUCOSE_MMOL_TO_MGDL = 18.016
LB_TO_KG = 0.45359237
INCH_TO_M = 0.0254

IMAGE_TYPES = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp")
FILE_TYPES = ("pdf",) + IMAGE_TYPES


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass
class Finding:
    key: str            # form field key, e.g. "hb"
    value: object       # number, or "A"/"B"/"AB"/"O", or "Yes"/"No"
    source: str         # file name
    line: str           # line of text the value was read from
    note: str = ""      # conversion / calculation / caution
    status: str = "read"  # read | converted | calculated


@dataclass
class FileText:
    name: str
    text: str = ""
    method: str = ""
    error: str = ""


@dataclass
class Extraction:
    findings: dict = field(default_factory=dict)    # key -> Finding
    conflicts: dict = field(default_factory=dict)   # key -> [Finding, ...]
    unusable: dict = field(default_factory=dict)    # key -> Finding (reason)
    files: list = field(default_factory=list)       # [FileText, ...]


class OCRUnavailableError(RuntimeError):
    pass


# ============================================================
# 1. GETTING TEXT OUT OF FILES
# ============================================================

def _tesseract():
    import pytesseract

    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def ocr_available():
    """True if the Tesseract program can be called on this server."""
    try:
        _tesseract().get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_image(image):
    if not ocr_available():
        raise OCRUnavailableError(
            "Tesseract OCR is not installed on this server, so photos and "
            "scanned pages cannot be read."
        )

    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image)

    gray = image.convert("L")

    if gray.width < OCR_MIN_WIDTH:
        factor = OCR_MIN_WIDTH / gray.width
        gray = gray.resize(
            (OCR_MIN_WIDTH, int(gray.height * factor)), Image.LANCZOS
        )

    gray = ImageOps.autocontrast(gray)

    return _tesseract().image_to_string(gray, config=TESSERACT_CONFIG)


def _pdf_text(data):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    texts, methods = [], []

    try:
        for i in range(min(len(pdf), MAX_PAGES_PER_FILE)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            textpage.close()

            if len(re.sub(r"\s", "", text)) >= MIN_TEXT_LAYER_CHARS:
                texts.append(text)
                methods.append("PDF text")
            else:
                bitmap = page.render(scale=OCR_RENDER_DPI / 72)
                texts.append(_ocr_image(bitmap.to_pil()))
                methods.append("OCR")
            page.close()
    finally:
        pdf.close()

    method = " + ".join(sorted(set(methods))) or "empty"
    return "\n".join(texts), method


def _image_text(data):
    image = Image.open(io.BytesIO(data))
    texts = []
    for i, frame in enumerate(ImageSequence.Iterator(image)):
        if i >= MAX_PAGES_PER_FILE:
            break
        texts.append(_ocr_image(frame.copy()))
    return "\n".join(texts), "OCR"


def read_file_text(name, data):
    """Return FileText with the text of one PDF or image file."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    result = FileText(name=name)

    try:
        if ext == "pdf":
            result.text, result.method = _pdf_text(data)
        elif ext in IMAGE_TYPES:
            result.text, result.method = _image_text(data)
        else:
            result.error = "Unsupported file type."
    except OCRUnavailableError as e:
        result.error = str(e)
    except Exception as e:  # corrupt file, unreadable image, etc.
        result.error = f"Could not read this file ({type(e).__name__}: {e})."

    return result


# ============================================================
# 2. TEXT NORMALISATION
# ============================================================

_CHAR_MAP = [
    ("¹²", "^12"), ("³", "^3"), ("⁶", "^6"), ("⁹", "^9"), ("²", "2"),
    ("×", "x"), ("µ", "u"), ("μ", "u"), ("–", "-"), ("—", "-"),
    ("：", ":"), ("′", "'"), ("″", '"'), ("\u00a0", " "),
]


def normalise_line(line):
    for old, new in _CHAR_MAP:
        line = line.replace(old, new)
    # Common OCR confusions inside numbers: 1O.5 -> 10.5, 9.O -> 9.0, 1l -> 11
    line = re.sub(r"(?<=\d)[Oo](?=[\d.])", "0", line)
    line = re.sub(r"(?<=\d\.)[Oo]", "0", line)
    line = re.sub(r"(?<=\d)[lI](?=\d)", "1", line)
    line = re.sub(r"(?<=\d\.)[lI]", "1", line)
    return re.sub(r"[ \t]+", " ", line).strip()


_UNIT_WORDS = re.compile(
    r"\b(?:g\s*/\s*dl|g\s*/\s*l|mg\s*/\s*dl|mmol\s*/\s*l|pg|fl|cumm|mm3|ul|"
    r"cells|millions?|mmhg|kg|cm|x|l|dl)\b"
)


def _is_fragment(line):
    """A line with only numbers, units and symbols (no label words)."""
    rest = re.sub(r"[x*]?\s*10\s*\^?\s*\d+", " ", line.lower())
    rest = _UNIT_WORDS.sub(" ", rest)
    return not re.search(r"[a-z]", rest)


def _join_fragments(lines):
    """
    PDF text layers and OCR sometimes break a table row, e.g.
        "Eosinophils" / "2" / "% 1-6".
    A unit fragment ("% 1-6") is attached to the line above. A lone value
    is attached to a label line above it only when the next line is not
    another lone value; several values in a row usually mean a column-wise
    layout, where matching values to labels would be guesswork.
    """
    out = []
    for i, line in enumerate(lines):
        if out and _is_fragment(line):
            prev = out[-1]
            if not line[0].isdigit():
                out[-1] = f"{prev} {line}"
                continue
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            prev_is_label = not _is_fragment(prev) and not values_in(prev)
            nxt_is_value = bool(nxt) and _is_fragment(nxt) and nxt[0].isdigit()
            if prev_is_label and not nxt_is_value:
                out[-1] = f"{prev} {line}"
                continue
        out.append(line)
    return out


def split_lines(text):
    lines = [
        ln for ln in (normalise_line(x) for x in re.split(r"[\r\n]+", text))
        if ln
    ]
    return _join_fragments(lines)


# ============================================================
# 3. NUMBER HANDLING
# ============================================================

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# A number not glued to a preceding letter/digit/point/caret
# (so "mm3", "10^3" and "A1c" are not read as values).
_NUM_RE = re.compile(rf"(?<![A-Za-z0-9.^])({_NUM})(?![0-9])")
_RANGE_RE = re.compile(rf"({_NUM})\s*(?:-|to)\s*({_NUM})", re.I)
_TIME_AFTER = re.compile(
    r"^\s*(?:h|hr|hrs|hour|hours|min|mins|minutes|wk|wks|weeks?|days?|"
    r"months?)\b", re.I
)


def _to_float(raw):
    return float(raw.replace(",", ""))


def _clean_segment(seg):
    """Remove unit noise that contains digits, e.g. (10^3/uL), x10^9/L."""
    seg = re.sub(r"\([^()]*[A-Za-z][^()]*\)", " ", seg)
    seg = re.sub(r"\[[^\[\]]*[A-Za-z][^\[\]]*\]", " ", seg)
    seg = re.sub(r"[x*]\s*10\s*\^?\s*\d{1,2}", " ", seg, flags=re.I)
    seg = re.sub(r"\b10\s*\^\s*\d{1,2}", " ", seg)
    seg = re.sub(r"\b10\s*e\s*\d{1,2}\b", " ", seg, flags=re.I)
    return seg


def values_in(segment):
    """
    Candidate result values in a text segment, in order, skipping
    reference ranges ("4.0 - 11.0"), limits ("< 100") and durations
    ("12 hours"). Returns [(value, text_after_value), ...].
    """
    seg = _clean_segment(segment)

    in_range = set()
    for m in _RANGE_RE.finditer(seg):
        in_range.update({m.start(1), m.start(2)})

    out = []
    for m in _NUM_RE.finditer(seg):
        if m.start(1) in in_range:
            continue
        before = seg[:m.start(1)].rstrip()
        if before.endswith(("<", ">", "<=", ">=", "≤", "≥")):
            continue
        after = seg[m.end(1):]
        if _TIME_AFTER.match(after):
            continue
        out.append((_to_float(m.group(1)), after))
    return out


# ============================================================
# 4. FIELD RULES
# ============================================================
# Labels are matched on the lower-case line. "(?![a-z])" is used instead of
# a trailing \b so that OCR output such as "WBC9.1" still matches.

LAB_LABELS = {
    "hb": (
        r"\b(?:ha?emoglobin|hgb|hb)(?![a-z0-9])",
        r"a1c|glyc|electrophor|corpuscular|mean\s+cell|\bmch",
    ),
    "rbc": (
        r"\b(?:total\s+)?(?:rbc|r\.b\.c\.?|red\s+(?:blood\s+)?cells?"
        r"(?:\s+count)?|erythrocytes?(?:\s+count)?)(?![a-z])",
        r"rdw|distribution|morpholog|nucleated|urine|indices",
    ),
    "wbc": (
        r"\b(?:total\s+)?(?:wbc|w\.b\.c\.?|tlc|t\.l\.c\.?|"
        r"total\s+leu[ck]ocyte(?:s)?(?:\s+count)?|"
        r"white\s+(?:blood\s+)?cells?(?:\s+count)?|"
        r"leu[ck]ocytes?(?:\s+count)?)(?![a-z])",
        r"urine|pus|esterase|differential|\bdlc",
    ),
    "mchc": (
        r"\bmchc(?![a-z])|mean\s+(?:corpuscular|cell)\s+ha?emoglobin\s+conc",
        r"$^",
    ),
    "fasting_glucose": (
        r"\b(?:fasting(?:\s+(?:blood|plasma|serum))?(?:\s+(?:glucose|sugar))?"
        r"|f\.?b\.?s|fbg|fpg|(?:blood|plasma|serum)?\s*(?:glucose|sugar)"
        r"\s*[,(\-:]*\s*fasting)(?![a-z])",
        r"random|\brbs|post|\bpp\b|prandial|a1c|insulin|lipid|cholesterol|"
        r"triglycer|status|duration|overnight|\b[12]\s*-?\s*h",
    ),
}

DIFF_LABELS = {
    "lymphocytes": (
        r"\b(?:lymphocytes?|lymphs?|lym|alc)(?![a-z])",
        r"atypical|reactive|variant",
    ),
    "eosinophils": (
        r"\b(?:eosinophils?|eos|eosin|aec)(?![a-z])",
        r"$^",
    ),
}

# Above these values a "unit not printed" differential count is treated as a
# percentage, below as an absolute count (x10^3/uL). Always flagged.
DIFF_PCT_THRESHOLD = {"lymphocytes": 8.0, "eosinophils": 1.0}

ABS_MARKER = re.compile(
    r"abs|absolute|#|\balc\b|\baec\b|10\^3|10\^9|/ul|/cumm|/mm3|/l\b|cells",
    re.I,
)

FIELD_LABELS = {
    "age": "Maternal age",
    "gravidity": "Gravidity",
    "bmi": "BMI",
    "family_history": "Family history of diabetes",
    "blood_type": "Blood group",
    "systolic": "Mean systolic BP",
    "diastolic": "Mean diastolic BP",
    "fasting_glucose": "Fasting glucose",
    "hb": "Haemoglobin",
    "rbc": "RBC",
    "wbc": "WBC",
    "mchc": "MCHC",
    "lymphocytes": "Lymphocytes (absolute)",
    "eosinophils": "Eosinophils (absolute)",
}


def _segment_after(line, label_re, exclude_re):
    low = line.lower()
    if re.search(exclude_re, low):
        return None
    m = re.search(label_re, low)
    if not m:
        return None
    return line[m.end():]



# ---------------- unit conversions for laboratory values ----------------

def _convert_lab(key, value, line):
    """Return (value, note, status) or (None, reason, '') if rejected."""
    low = line.lower()

    if key == "hb":
        if re.search(r"g\s*/\s*l\b", low) and "g/dl" not in low.replace(" ", ""):
            return value / 10, f"Converted from {value:g} g/L.", "converted"
        if value > 30:
            return value / 10, (
                f"Read {value:g}; above 30, so assumed to be g/L and "
                "divided by 10."), "converted"
        return value, "", "read"

    if key == "mchc":
        if value > 100:
            return value / 10, f"Converted from {value:g} g/L.", "converted"
        return value, "", "read"

    if key == "rbc":
        if value > 1e5:
            return value / 1e6, (
                f"Converted from {value:,.0f} cells/mm³."), "converted"
        if value > 15:
            return None, f"RBC value {value:g} has an unclear unit.", ""
        return value, "", "read"

    if key == "wbc":
        if value > 500:
            return value / 1000, (
                f"Converted from {value:,.0f} cells/mm³."), "converted"
        if value > 100:
            return None, f"WBC value {value:g} has an unclear unit.", ""
        return value, "", "read"

    if key == "fasting_glucose":
        if "mmol" in low:
            return value * GLUCOSE_MMOL_TO_MGDL, (
                f"Converted from {value:g} mmol/L (x 18.016)."), "converted"
        if value < 30:
            return value * GLUCOSE_MMOL_TO_MGDL, (
                f"Read {value:g} with no unit; below 30, so assumed to be "
                "mmol/L and multiplied by 18.016."), "converted"
        return value, "", "read"

    return value, "", "read"


# ---------------- individual parsers ----------------

def _parse_labs(lines, source, add):
    for key, (label, exclude) in LAB_LABELS.items():
        for line in lines:
            seg = _segment_after(line, label, exclude)
            if seg is None:
                continue
            vals = values_in(seg)
            if not vals:
                continue
            value, note, status = _convert_lab(key, vals[0][0], line)
            if value is None:
                add(Finding(key, None, source, line, note, "rejected"))
                continue
            add(Finding(key, value, source, line, note, status))


def _parse_differential(lines, source, add_raw):
    """Collect absolute and percentage lymphocyte/eosinophil counts."""
    for key, (label, exclude) in DIFF_LABELS.items():
        for line in lines:
            seg = _segment_after(line, label, exclude)
            if seg is None:
                continue
            vals = values_in(seg)
            if not vals:
                continue

            has_abs = bool(ABS_MARKER.search(line))
            first, after = vals[0]
            first_is_pct = after.lstrip().startswith("%")

            if first_is_pct:
                add_raw(key, "pct", first, source, line, "")
                if has_abs and len(vals) > 1:
                    add_raw(key, "abs", vals[1][0], source, line, "")
            elif has_abs:
                add_raw(key, "abs", first, source, line, "")
            elif "%" in line:
                add_raw(key, "pct", first, source, line, "")
            else:
                kind = "pct" if first >= DIFF_PCT_THRESHOLD[key] else "abs"
                add_raw(key, kind, first, source, line, (
                    "Unit not printed; read as "
                    + ("a percentage." if kind == "pct"
                       else "an absolute count.")))


def _parse_age(lines, source, add):
    for line in lines:
        seg = _segment_after(
            line, r"\bage(?![a-z])",
            r"gestation|gest\.|\bga\b|weeks|wks|\bpog\b|\bpoa\b|\blmp\b|\bedd\b",
        )
        if seg is None:
            continue
        m = re.search(
            r"(?<![\d.])(\d{1,2})(?:\.\d+)?\s*(y|yr|yrs|year|years|m|mo|"
            r"months?|d|days?)?(?![a-z])",
            seg.lower(),
        )
        if not m:
            continue
        unit = m.group(2) or ""
        if unit and not unit.startswith("y"):
            continue
        add(Finding("age", float(m.group(1)), source, line))


def _parse_gravidity(lines, source, add):
    for line in lines:
        low = line.lower()
        m = (re.search(r"\bg\s*(\d{1,2})\s*[,/ ]*\s*p\s*\d", low)
             or re.search(
                 r"gravid(?:a|ity)\s*(?:no\.?|number)?\s*[:=\-]?\s*"
                 r"(\d{1,2})(?![\d.])", low))
        if m:
            add(Finding("gravidity", float(m.group(1)), source, line))
        elif re.search(r"\bprimi\s*-?\s*gravid|\bprimigravid|\bprimi\b", low):
            add(Finding("gravidity", 1.0, source, line,
                        "Recorded as primigravida.", "converted"))


def _parse_bp(lines, source):
    readings = []
    for line in lines:
        low = line.lower()
        if not re.search(r"\bb\.?\s?p\.?(?![a-z])|blood\s+pressure|mm\s*hg", low):
            continue
        for m in re.finditer(r"(?<![\d.])(\d{2,3})\s*/\s*(\d{2,3})(?![\d.])",
                             line):
            sys_, dia = float(m.group(1)), float(m.group(2))
            if 60 <= sys_ <= 250 and 30 <= dia <= 150 and sys_ > dia:
                readings.append((sys_, dia, line))
    return readings


def _parse_body(lines, source, add):
    """BMI if printed, otherwise weight and height for a calculation."""
    weights, heights = [], []

    for line in lines:
        seg = _segment_after(line, r"\bbmi(?![a-z])|body\s+mass\s+index", r"$^")
        if seg is not None:
            vals = values_in(seg)
            if vals:
                add(Finding("bmi", vals[0][0], source, line))

        seg = _segment_after(
            line, r"\b(?:weight|wt)(?![a-z])",
            r"gain|loss|birth|baby|fetal|foetal|\befw\b|molecular",
        )
        if seg is not None:
            vals = values_in(seg)
            if vals:
                w, after = vals[0]
                if re.match(r"\s*(?:lb|lbs|pound)", after.lower()):
                    w *= LB_TO_KG
                if 30 <= w <= 200:
                    weights.append((w, line))

        seg = _segment_after(line, r"\b(?:height|ht)(?![a-z])",
                             r"fundal|\bsfh\b|uter")
        if seg is not None:
            ft = re.search(
                r"(\d)\s*(?:'|ft|feet|foot)\s*(\d{1,2}(?:\.\d+)?)?\s*"
                r"(?:\"|''|in|inch|inches)?", seg.lower())
            h = None
            if ft:
                h = (int(ft.group(1)) * 12 + float(ft.group(2) or 0)) * INCH_TO_M
            else:
                vals = values_in(seg)
                if vals:
                    v = vals[0][0]
                    if 100 <= v <= 220:
                        h = v / 100
                    elif 1.2 <= v <= 2.2:
                        h = v
            if h and 1.2 <= h <= 2.2:
                heights.append((h, line))

    return weights, heights


def _parse_blood_group(lines, source, add):
    for line in lines:
        seg = _segment_after(
            line,
            r"blood\s*group|blood\s*type|\babo(?![a-z])|\bb\.\s*group|grouping",
            r"$^",
        )
        if seg is None:
            continue
        m = re.search(
            r"(?<![A-Z0-9])(AB|A|B|O|0)(?=\s*(?:\(|RH|\+|-|POS|NEG|$|[,;.\s]))",
            seg.upper(),
        )
        if not m:
            continue
        group = m.group(1)
        if group == "0":
            tail = seg.upper()[m.end():]
            if not re.match(r"\s*(?:\(|RH|\+|-|POS|NEG)", tail):
                continue
            group = "O"
        add(Finding("blood_type", group, source, line,
                    "Rh factor is not used by the model."))


def _parse_family_history(lines, source, add):
    neg = re.compile(r"\b(?:no|nil|none|negative|absent|not\s+known|nad)\b|-ve")
    for line in lines:
        low = line.lower()
        if not re.search(r"family\s*(?:history|hx|h/o)|\bf/h\b|\bfh\b", low):
            continue
        dis = re.search(r"diab|\bdm\b|sugar|t2dm|niddm|iddm", low)
        if dis:
            window = low[max(0, dis.start() - 25):dis.end() + 8]
            value = "No" if neg.search(window) else "Yes"
        elif neg.search(low):
            value = "No"
        else:
            continue
        add(Finding("family_history", value, source, line,
                    "Read from free text; please confirm."))


# ============================================================
# 5. COMBINING ALL FILES
# ============================================================

def _same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def read_reports(files):
    """
    files: list of (file_name, file_bytes).
    Returns an Extraction with at most one chosen Finding per field.
    """
    return parse_texts([read_file_text(name, data) for name, data in files])


def parse_texts(file_texts):
    """Parse already-extracted text (list of FileText). Used by tests too."""
    result = Extraction()
    candidates = {}         # key -> [Finding]
    diff_raw = {}           # key -> {"abs": [...], "pct": [...]}
    bp_readings = []
    weights, heights = [], []
    rejected = {}

    def add(f):
        if f.status == "rejected":
            rejected.setdefault(f.key, f)
            return
        candidates.setdefault(f.key, []).append(f)

    def add_raw(key, kind, value, source, line, note):
        diff_raw.setdefault(key, {"abs": [], "pct": []})[kind].append(
            (value, source, line, note))

    for ft in file_texts:
        name = ft.name
        result.files.append(ft)
        if ft.error or not ft.text.strip():
            continue

        lines = split_lines(ft.text)
        _parse_labs(lines, name, add)
        _parse_differential(lines, name, add_raw)
        _parse_age(lines, name, add)
        _parse_gravidity(lines, name, add)
        _parse_blood_group(lines, name, add)
        _parse_family_history(lines, name, add)
        bp_readings += [(s, d, ln, name) for s, d, ln in _parse_bp(lines, name)]
        w, h = _parse_body(lines, name, add)
        weights += [(v, ln, name) for v, ln in w]
        heights += [(v, ln, name) for v, ln in h]

    # Pick one value per field; keep disagreeing values as conflicts.
    for key, items in candidates.items():
        chosen = items[0]
        others = [f for f in items[1:] if not _same(f.value, chosen.value)]
        result.findings[key] = chosen
        if others:
            result.conflicts[key] = [chosen] + others

    # Blood pressure: mean of all readings found.
    if bp_readings:
        n = len(bp_readings)
        sys_mean = sum(r[0] for r in bp_readings) / n
        dia_mean = sum(r[1] for r in bp_readings) / n
        listed = ", ".join(f"{s:g}/{d:g}" for s, d, _, _ in bp_readings)
        note = (f"Mean of {n} reading{'s' if n > 1 else ''} ({listed}). "
                "Check that these are the readings the study averaged.")
        sources = ", ".join(sorted({r[3] for r in bp_readings}))
        line = " | ".join(r[2] for r in bp_readings[:3])
        status = "calculated" if n > 1 else "read"
        result.findings["systolic"] = Finding(
            "systolic", sys_mean, sources, line, note, status)
        result.findings["diastolic"] = Finding(
            "diastolic", dia_mean, sources, line, note, status)

    # BMI: calculate only if not printed.
    if "bmi" not in result.findings and weights and heights:
        (w, wl, wn), (h, hl, hn) = weights[0], heights[0]
        result.findings["bmi"] = Finding(
            "bmi", w / h ** 2, ", ".join(sorted({wn, hn})),
            wl if wl == hl else f"{wl} | {hl}",
            (f"Calculated from weight {w:.1f} kg and height {h:.2f} m. "
             "Check that this matches the BMI used in the study "
             "(e.g. booking or pre-pregnancy weight)."),
            "calculated")

    # Differential counts: absolute preferred; else % x WBC / 100.
    for key, kinds in diff_raw.items():
        if kinds["abs"]:
            v, src, ln, note = kinds["abs"][0]
            status = "read"
            if v > 50:
                v, status = v / 1000, "converted"
                note = (note + " " if note else "") + \
                    "Converted from cells/uL to x10^3/uL."
            result.findings[key] = Finding(key, v, src, ln, note.strip(),
                                           status)
        elif kinds["pct"] and "wbc" in result.findings:
            pct, src, ln, note = kinds["pct"][0]
            wbc = result.findings["wbc"].value
            result.findings[key] = Finding(
                key, pct * wbc / 100, src, ln,
                (note + " " if note else "") +
                f"Calculated as {pct:g}% x WBC {wbc:g} / 100.",
                "calculated")
        elif kinds["pct"]:
            pct, src, ln, _ = kinds["pct"][0]
            rejected.setdefault(key, Finding(
                key, None, src, ln,
                f"Only a percentage ({pct:g}%) was found and WBC is missing, "
                "so the absolute count could not be calculated.", "rejected"))

    # Keep the reason for fields that were seen but could not be used.
    for key, f in rejected.items():
        if key not in result.findings:
            result.unusable[key] = f

    return result


# ============================================================
# 6. PREPARING VALUES FOR THE FORM
# ============================================================

STATUS_TEXT = {
    "read": "Read",
    "converted": "Unit converted",
    "calculated": "Calculated",
}


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def summarise_for_form(ex, input_ranges):
    """
    Decide which values may pre-fill the form.

    input_ranges: {key: (minimum, maximum, decimals)} - the same limits the
    form uses. Values are rounded to the form's precision; values outside
    the limits are NOT filled and are listed under "attention".

    Returns {"prefill", "rows", "attention", "not_found", "files"}.
    """
    prefill, rows, attention = {}, [], []

    for key, label in FIELD_LABELS.items():
        f = ex.findings.get(key)
        if f is None:
            continue
        v = f.value
        if key in input_ranges:
            lo, hi, dec = input_ranges[key]
            v = round(float(v), dec)
            if not lo <= v <= hi:
                attention.append(
                    f"{label}: read as {v:g}, outside the accepted range "
                    f"{lo:g}-{hi:g}, so it was not filled."
                )
                continue
        prefill[key] = v
        rows.append({
            "Parameter": label,
            "Filled value": _fmt(v),
            "How": STATUS_TEXT.get(f.status, f.status),
            "Note": f.note,
            "Read from": f"{f.source}: {f.line[:90]}",
        })

    for key, items in ex.conflicts.items():
        if key in prefill:
            shown = ", ".join(_fmt(x.value) for x in items)
            attention.append(
                f"{FIELD_LABELS[key]}: different values were found "
                f"({shown}); the first one was filled."
            )

    for key, f in ex.unusable.items():
        attention.append(f"{FIELD_LABELS[key]}: {f.note}")

    for ft in ex.files:
        if ft.error:
            attention.append(f"{ft.name}: {ft.error}")
        elif not ft.text.strip():
            attention.append(f"{ft.name}: no text could be read.")

    return {
        "prefill": prefill,
        "rows": rows,
        "attention": attention,
        "not_found": [
            label for key, label in FIELD_LABELS.items() if key not in prefill
        ],
        "files": [(ft.name, ft.method, ft.text) for ft in ex.files],
    }
