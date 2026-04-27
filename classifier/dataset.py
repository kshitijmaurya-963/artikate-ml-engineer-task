import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TicketExample:
    text: str
    label: str


def load_jsonl(path: str) -> List[TicketExample]:
    p = Path(path)
    examples: List[TicketExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(TicketExample(text=obj["text"], label=obj["label"]))
    return examples
