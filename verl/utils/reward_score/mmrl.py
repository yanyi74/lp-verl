import re

from mathruler.grader import extract_boxed_content, grade_answer

# Answer extraction order mirrors fliter_mmrl/revisual_accuracy.py (default reward_mode=revisual):
#   1. \boxed{} content
#   2. <answer>…</answer> tag
#   3. natural-language conclusion patterns
#
# Note: the offline labeling pipeline (fliter_mmrl) also supports a "math" mode that calls
# accuracy_reward_func(text, gt, gt_answer_real_body=..., format_strict=...) from
# LLaVA-OneVision-1.5-RL, but that external package is not a verl dependency. The revisual
# path (used here) is functionally equivalent for all ViRL39K / MMRL visual-math data.
_CONCLUSION_PATTERNS = [
    r"(?:Therefore|Thus|So|Hence),?\s+(?:the\s+)?(?:answer|solution|result)\s+(?:is|=)\s+(.*?)(?:\.|$)",
    r"(?:answer|solution|result)\s*(?::|=)\s*(.*?)(?:\.|$)",
    r"(?:Finally|In conclusion),\s+(?:we\s+(?:get|have|find))?\s*(.*?)(?:\.|$)",
]


def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
    answer = extract_boxed_content(solution_str)

    if not answer:
        m = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        if m:
            answer = m.group(1).strip()

    if not answer:
        for pattern in _CONCLUSION_PATTERNS:
            m = re.search(pattern, solution_str, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                break

    return 1.0 if grade_answer(answer, ground_truth) else 0.0
