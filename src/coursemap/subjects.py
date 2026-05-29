from coursemap.text import clean_subject_name


SUBJECT_DICT = {
    "AI기초": ("인공지능기초", "정보·AI"),
    "인공지능기초": ("인공지능기초", "정보·AI"),
    "정보": ("정보", "정보·AI"),
    "데이터과학": ("데이터과학", "정보·AI"),
    "미적분": ("미적분", "자연·공학"),
    "기하": ("기하", "자연·공학"),
    "물리학1": ("물리학1", "자연·공학"),
    "화학1": ("화학1", "자연·공학"),
    "생명과학1": ("생명과학1", "자연·공학"),
    "사회문화": ("사회·문화", "인문·사회"),
    "생활과윤리": ("생활과윤리", "인문·사회"),
    "정치와법": ("정치와법", "인문·사회"),
    "세계사": ("세계사", "인문·사회"),
    "체육": ("체육", "예체능"),
    "음악": ("음악", "예체능"),
    "미술": ("미술", "예체능"),
}

EXCLUDE_KEYWORDS = (
    "창체", "자율", "동아리", "봉사", "진로활동", "조회", "종례",
    "학급", "행사", "스포츠클럽", "자습", "공강", "중간고사", "기말고사",
    "고사", "재량휴업", "개교기념일", "현장체험학습", "예방교육",
    "선행교육근절교육", "자기주도학습", "자기주도적학습", "자기주도", "현장체험",
)


def is_excluded_subject(value) -> bool:
    text = clean_subject_name(value)
    return not text or any(key in text for key in EXCLUDE_KEYWORDS)


DOMAINS = [
    "인문·사회",
    "자연·공학",
    "정보·AI",
    "예체능",
    "제2외국어·국제",
    "진로·융합",
    "기초·공통",
    "기타",
]


def classify_subject(value) -> tuple[str, str]:
    cleaned = clean_subject_name(value)
    if cleaned in SUBJECT_DICT:
        return SUBJECT_DICT[cleaned]
    rules = [
        ("정보·AI", ("정보", "인공지능", "AI", "데이터", "자료구조", "코딩", "프로그래밍", "소프트웨어")),
        ("자연·공학", ("수학", "미적분", "기하", "확률과통계", "물리", "화학", "생명", "지구과학", "과학", "과제연구", "공학", "환경")),
        ("인문·사회", ("국어", "문학", "독서", "고전읽기", "언어와매체", "화법과작문", "논술", "사회", "윤리", "역사", "세계사", "동아시아사", "지리", "정치", "경제", "철학", "논리학", "심리학", "교육학", "한국사", "후마니타스")),
        ("예체능", ("체육", "운동", "스포츠", "음악", "미술", "무용", "연극", "디자인", "드로잉", "서양화", "발레", "성악", "보컬", "피아노", "바이올린", "관악", "현악", "타악기", "국악", "대금", "해금", "거문고", "작곡", "안무", "연기", "공연", "무대", "시창", "청음", "합창", "합주", "만화", "애니메이션", "입체조형", "영화")),
        ("제2외국어·국제", ("영어", "영어권문화", "영어독해", "영어회화", "실용영어", "심화영어", "비즈니스영어", "중국어", "일본어", "일본문화", "중국문화", "프랑스어", "독일어", "스페인어", "한문", "국제", "비교문화", "지역이해")),
        ("진로·융합", ("진로", "융합", "프로젝트", "탐구", "교양", "창업", "창의경영", "지식재산", "기술·가정", "보건", "간호", "가정과학", "생활과과학", "식품", "자동차", "신재생에너지")),
    ]
    for domain, keywords in rules:
        if any(keyword in cleaned for keyword in keywords):
            return cleaned, domain
    return cleaned, "기타"


def classify_subject_with_method(value) -> tuple[str, str, str]:
    cleaned = clean_subject_name(value)
    if cleaned in SUBJECT_DICT:
        standard, domain = SUBJECT_DICT[cleaned]
        return standard, domain, "dict"

    standard, domain = classify_subject(cleaned)
    if domain != "기타":
        return standard, domain, "rule"
    return standard, domain, "unassigned"
