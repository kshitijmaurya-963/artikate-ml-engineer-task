import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .dataset import load_jsonl
from .model import load_tokenizer_and_model, LABEL2ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "tickets_train.jsonl"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "classifier"


class TicketDataset(Dataset):
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
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = load_jsonl(str(TRAIN_DATA_PATH))
    if not examples:
        raise ValueError(
            f"No training examples found in {TRAIN_DATA_PATH}. "
            "Populate data/tickets_train.jsonl with labeled tickets before training."
        )
    texts = [ex.text for ex in examples]
    labels = [LABEL2ID[ex.label] for ex in examples]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer, model = load_tokenizer_and_model()
    model.to(device)

    train_ds = TicketDataset(X_train, y_train, tokenizer)
    val_ds = TicketDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    optim = AdamW(model.parameters(), lr=2e-5)
    num_epochs = int(os.getenv("EPOCHS", "3"))
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()

        # Simple validation loss printout (detailed metrics in evaluate.py)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_losses.append(outputs.loss.item())
        print(f"Epoch {epoch+1}/{num_epochs} - val_loss={np.mean(val_losses):.4f}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save_pretrained(str(ARTIFACTS_DIR))
    tokenizer.save_pretrained(str(ARTIFACTS_DIR))


if __name__ == "__main__":
    main()
