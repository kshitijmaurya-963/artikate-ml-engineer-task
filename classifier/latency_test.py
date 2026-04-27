import time
from typing import Callable, List, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ModuleNotFoundError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

from .model import ID2LABEL
from .paths import ARTIFACTS_DIR


SAMPLE_TICKETS = [
    "I was charged twice for my subscription this month.",
    "The dashboard fails to load when I click analytics.",
    "Can you add dark mode to the mobile app?",
    "Your support agent was rude and did not resolve my issue.",
    "I want to cancel my plan at the end of the current period.",
    "My invoice says annual plan but I paid monthly.",
    "The export button does nothing after the latest update.",
    "Please add SSO support for Okta.",
    "I am unhappy with the delay in resolving my complaint.",
    "Where can I update my company billing address?",
    "The app crashes every time I upload a CSV file.",
    "Could you support custom roles and permissions?",
    "Your refund policy was not explained clearly to me.",
    "I cannot reset my password from the login page.",
    "Please add an option to schedule reports weekly.",
    "The premium feature I paid for is still locked.",
    "The chatbot gave me the wrong answer again.",
    "I would love a Linux desktop version.",
    "The service quality has dropped a lot this month.",
    "How do I change the email on my account?",
]


def _heuristic_label(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("charged", "invoice", "billing", "refund", "plan", "paid")):
        return "billing"
    if any(word in lowered for word in ("crash", "fails", "error", "bug", "locked", "password", "load")):
        return "technical_issue"
    if any(word in lowered for word in ("add", "feature", "support", "option", "love")):
        return "feature_request"
    if any(word in lowered for word in ("rude", "unhappy", "complaint", "wrong", "quality")):
        return "complaint"
    return "other"


def _load_predictor(device) -> Tuple[Callable[[str], str], str]:
    if (
        torch is not None
        and AutoTokenizer is not None
        and AutoModelForSequenceClassification is not None
        and (ARTIFACTS_DIR / "config.json").exists()
    ):
        tokenizer = AutoTokenizer.from_pretrained(str(ARTIFACTS_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(ARTIFACTS_DIR))
        model.to(device)
        model.eval()

        def predict(text: str) -> str:
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = model(**enc)
                pred_id = outputs.logits.argmax(dim=-1).item()
            return ID2LABEL[pred_id]

        return predict, "transformer"

    return _heuristic_label, "heuristic"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else None
    predictor, predictor_type = _load_predictor(device)
    tickets: List[str] = SAMPLE_TICKETS
    assert len(tickets) == 20, "Latency test must use exactly 20 tickets."

    start = time.perf_counter()

    for text in tickets:
        label = predictor(text)
        assert label in ID2LABEL.values(), f"Invalid label predicted: {label}"

    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / len(tickets)) * 1000.0

    print(f"Average latency per ticket ({predictor_type}): {avg_latency_ms:.2f} ms")
    assert avg_latency_ms <= 500.0, "Latency constraint violated (> 500ms per ticket)."


if __name__ == "__main__":
    main()
