# utils/severity.py
HIGH_SEVERITY = {"carcinoma","malignancy","metastasis","hemorrhage","mass","lesion","pneumothorax","consolidation","infiltrate","acute"}
MEDIUM_SEVERITY = {"opacity","nodule","effusion","atelectasis","fibrosis"}

def severity_from_text(text: str):
    text_l = text.lower()
    score = 0
    for w in HIGH_SEVERITY:
        if w in text_l:
            return "high"
    for w in MEDIUM_SEVERITY:
        if w in text_l:
            score = max(score, 1)
    return "low" if score==0 else "medium"

def supportive_template(summary: str):
    return (
        "I know this may sound worrying. Here's a plain-language summary:\n\n"
        f"{summary}\n\n"
        "This is not a diagnosis. Consider asking your clinician:\n"
        "1) Could you explain what this finding likely means for me?\n"
        "2) Do I need follow-up tests or treatment?\n"
        "3) Is immediate action needed?\n"
    )
