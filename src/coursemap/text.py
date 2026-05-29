import re


def normalize_school_name(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    text = re.sub(r"\s+", "", text)
    text = text.replace("고등학교", "고")
    return text


def clean_subject_name(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^]]*\]", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("Ⅰ", "1").replace("Ⅱ", "2")
    text = text.replace("Ⅲ", "3").replace("Ⅳ", "4").replace("Ⅴ", "5")
    text = re.sub(r"(토의)?학습실?\d*$", "", text)
    text = re.sub(r"[A-Za-z가-힣]?\d+-\d+$", "", text)
    text = re.sub(r"[A-Za-z]\d*$", "", text)
    text = re.sub(r"[A-Ga-g]반$", "", text)
    text = re.sub(r"[-_]?[\d]+반$", "", text)
    text = re.sub(r"^\d+교시", "", text)
    if len(text) % 2 == 0:
        half = len(text) // 2
        if text[:half] == text[half:]:
            text = text[:half]
    return text
