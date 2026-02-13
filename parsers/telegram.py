import json
from pathlib import Path

SKIP_AUTHORS = {"Parsalov Rhymes News"}


def _extract_text(msg: dict) -> str | None:
    raw = msg.get("text")
    if not raw:
        return None
    if isinstance(raw, str):
        line = raw
    elif isinstance(raw, list):
        line = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in raw
        )
    else:
        return None
    line = line.strip()
    return line if len(line) > 1 else None


def _load_file(filepath: Path) -> tuple[list[str], list[tuple[str, str]]]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    msg_by_id: dict[int, str] = {}
    reply_pairs = []
    sequential_pairs = []
    prev_text = None

    for msg in data["messages"]:
        if msg.get("type") != "message":
            continue
        if msg.get("from") in SKIP_AUTHORS:
            prev_text = None
            continue

        text = _extract_text(msg)
        if not text:
            prev_text = None
            continue

        texts.append(text)
        msg_id = msg.get("id")
        if msg_id is not None:
            msg_by_id[msg_id] = text

        reply_to = msg.get("reply_to_message_id")
        if reply_to and reply_to in msg_by_id:
            reply_pairs.append((msg_by_id[reply_to], text))
        elif prev_text:
            sequential_pairs.append((prev_text, text))

        prev_text = text

    return texts, reply_pairs + sequential_pairs


def load(dirpath: str) -> tuple[list[str], list[tuple[str, str]]]:
    all_texts = []
    all_pairs = []

    for filepath in sorted(Path(dirpath).glob("*.json")):
        texts, pairs = _load_file(filepath)
        all_texts.extend(texts)
        all_pairs.extend(pairs)
        print(f"    {filepath.name}: {len(texts)} сообщений, {len(pairs)} пар")

    return all_texts, all_pairs
