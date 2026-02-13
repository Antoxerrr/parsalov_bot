from pathlib import Path

from bs4 import BeautifulSoup


def _extract_messages(dirpath: Path) -> list[tuple[str, str]]:
    """Возвращает список (автор, текст) в хронологическом порядке."""
    messages = []

    for filepath in sorted(dirpath.glob("messages*.html")):
        with open(filepath, encoding="windows-1251", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for item in soup.select("div.message"):
            divs = item.find_all("div", recursive=False)
            if len(divs) < 2:
                continue

            header = divs[0]
            is_other = header.find("a") is not None
            author = "other" if is_other else "me"

            text_div = divs[1]
            for kludge in text_div.select("div.kludges"):
                kludge.decompose()

            text = text_div.get_text(strip=True)
            if len(text) > 1:
                messages.append((author, text))

    return messages


def _group_and_pair(messages: list[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
    """Группирует последовательные сообщения от одного автора,
    строит пары: вопрос (склеенный) → каждое отдельное сообщение ответа."""
    all_texts = []
    pairs = []

    # группируем последовательные сообщения от одного автора
    groups: list[tuple[str, list[str]]] = []
    for author, text in messages:
        all_texts.append(text)
        if groups and groups[-1][0] == author:
            groups[-1][1].append(text)
        else:
            groups.append((author, [text]))

    # строим пары: группа A → каждое сообщение из группы B
    for i in range(1, len(groups)):
        question_texts = groups[i - 1][1]
        answer_texts = groups[i][1]
        question = " ".join(question_texts)

        for answer in answer_texts:
            pairs.append((question, answer))

    return all_texts, pairs


def _load_chat(dirpath: Path) -> tuple[list[str], list[tuple[str, str]]]:
    messages = _extract_messages(dirpath)
    return _group_and_pair(messages)


def load(dirpath: str) -> tuple[list[str], list[tuple[str, str]]]:
    all_texts = []
    all_pairs = []
    root = Path(dirpath)

    if list(root.glob("messages*.html")):
        texts, pairs = _load_chat(root)
        all_texts.extend(texts)
        all_pairs.extend(pairs)
        print(f"    {root.name}: {len(texts)} сообщений, {len(pairs)} пар")
    else:
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            if not list(subdir.glob("messages*.html")):
                continue
            texts, pairs = _load_chat(subdir)
            all_texts.extend(texts)
            all_pairs.extend(pairs)
            print(f"    {subdir.name}: {len(texts)} сообщений, {len(pairs)} пар")

    return all_texts, all_pairs
