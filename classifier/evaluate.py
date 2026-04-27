import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .dataset import load_jsonl
from .model import ID2LABEL
from .paths import ARTIFACTS_DIR, DATA_DIR


class EvalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = label
        return item


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_path = DATA_DIR / "tickets_eval.jsonl"
    eval_examples = load_jsonl(str(eval_path))
    texts = [ex.text for ex in eval_examples]
    labels = [ex.label for ex in eval_examples]

    if not texts:
        raise ValueError(
            f"No evaluation examples found in {eval_path}. "
            "Populate data/tickets_eval.jsonl before running evaluation."
        )

    # Ensure at least 100 manually verified examples per assignment requirement
    assert len(texts) >= 100, "tickets_eval.jsonl must contain at least 100 examples."

    tokenizer = AutoTokenizer.from_pretrained(str(ARTIFACTS_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(ARTIFACTS_DIR))
    model.to(device)
    model.eval()

    label2id = {label: idx for idx, label in ID2LABEL.items()}
    y_true_ids = [label2id[l] for l in labels]

    ds = EvalDataset(texts, y_true_ids, tokenizer)
    loader = DataLoader(ds, batch_size=32)

    y_pred_ids = []

    with torch.no_grad():
        for batch in loader:
            labels_batch = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            y_pred_ids.extend(preds)

    acc = accuracy_score(y_true_ids, y_pred_ids)
    f1_per_class = f1_score(y_true_ids, y_pred_ids, average=None)
    cm = confusion_matrix(y_true_ids, y_pred_ids)

    print("Accuracy:", acc)
    for idx, f1_val in enumerate(f1_per_class):
        print(f"F1({ID2LABEL[idx]}): {f1_val:.3f}")

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true_ids, y_pred_ids, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))]))


if __name__ == "__main__":
    main()
