try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 5
LABEL2ID = {
    "billing": 0,
    "technical_issue": 1,
    "feature_request": 2,
    "complaint": 3,
    "other": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_tokenizer_and_model():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ModuleNotFoundError(
            "transformers is required to load the classifier model. Install dependencies from requirements.txt."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tokenizer, model
