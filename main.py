import os
import random
import traceback
from collections import Counter, defaultdict

import markovify
from dotenv import load_dotenv
import openai
from openai import OpenAI
from telebot import TeleBot

from parsers import telegram, vk

load_dotenv()

TG_CORPUS_DIR = "corpus/telegram"
VK_CORPUS_DIR = "corpus/vk"
CANDIDATES_COUNT = 100
CONTEXT_SIZE = 10
LLM_MODEL = "openai/gpt-5"
# LLM_MODEL = "openai/gpt-4.1-mini"

bot = TeleBot(os.environ["TELEGRAM_BOT_TOKEN"])
llm = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url="https://api.vsellm.ru/v1")

chat_history: dict[int, list[str]] = defaultdict(list)

SYSTEM_PROMPT = (
    "Ты — бот в чате друзей. Тебе дают контекст переписки и список вариантов ответа. "
    "Выбери один вариант по следующему приоритету:\n"
    "1. Если есть вариант, который логично и смешно отвечает на сообщение — выбери его.\n"
    "2. Если полностью логичного ответа нет — возьми самый близкий по смыслу вариант "
    "и доработай его до логичного ответа, МАКСИМАЛЬНО сохранив стиль и лексику из примеров."
    "Сарказм и шутки приветствуются.\n"
    "Не выбирай слишком короткие скучные ответы вроде 'да', 'нет', 'ок' — предпочитай что-то поинтереснее. "
    "Верни ТОЛЬКО текст выбранного варианта, без номера, без пояснений и без метаданных вроде '[На ...]'."
    "Отвечай всегда ОТ ПЕРВОГО ЛИЦА, словно обращались к тебе, и ты отвечаешь на сообщение."
)

TRIGGER = "парсалио"

STOP_WORDS = {
    "и", "в", "на", "не", "что", "я", "с", "он", "а", "это", "она", "они",
    "мы", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "от",
    "то", "как", "его", "ну", "ее", "нет", "ещё", "еще", "уже", "так",
    "тут", "там", "вот", "все", "всё", "если", "из", "для", "до", "ой",
    "ага", "ок", "ладно", "ну", "блин", "о", "эт",
}


def tokenize(text: str) -> set[str]:
    return set(text.lower().split()) - STOP_WORDS


def build_retrieval_index(pairs: list[tuple[str, str]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, (question, _) in enumerate(pairs):
        for word in tokenize(question):
            index[word].append(i)
    return index


def retrieve_replies(
    query: str,
    pairs: list[tuple[str, str]],
    index: dict[str, list[int]],
    n: int = CANDIDATES_COUNT,
) -> list[tuple[str, str]]:
    query_words = tokenize(query)
    if not query_words:
        return []

    scores: dict[int, int] = Counter()
    for word in query_words:
        for pair_idx in index.get(word, []):
            scores[pair_idx] += 1

    top = sorted(scores, key=scores.get, reverse=True)[:n]
    seen = set()
    results = []
    for idx in top:
        reply = pairs[idx][1]
        if reply not in seen:
            seen.add(reply)
            results.append(pairs[idx])
    return results


def build_model(texts: list[str], state_size: int) -> markovify.NewlineText:
    return markovify.NewlineText("\n".join(texts), state_size=state_size)


def generate_candidates(
    models: list[markovify.NewlineText],
    user_words: list[str],
    n: int = CANDIDATES_COUNT,
) -> list[str]:
    candidates = set()
    per_model = n // len(models)

    for model in models:
        for word in user_words:
            if len(candidates) >= n:
                break
            try:
                s = model.make_sentence_with_start(word, strict=False, tries=20)
                if s:
                    candidates.add(s)
            except (KeyError, markovify.text.ParamError):
                continue

        for _ in range(per_model * 2):
            if len(candidates) >= n:
                break
            s = model.make_sentence(tries=50)
            if s:
                candidates.add(s)

    return list(candidates)


def format_candidates(
    retrieved: list[tuple[str, str]],
    markov: list[str],
) -> str:
    lines = []
    i = 1
    for question, answer in retrieved:
        lines.append(f'{i}. [На "{question}"] → {answer}')
        i += 1
    for s in markov:
        lines.append(f"{i}. {s}")
        i += 1
    return "\n".join(lines)


def pick_best(
    context: list[str],
    retrieved: list[tuple[str, str]],
    markov: list[str],
) -> str:
    context_text = "\n".join(context[-CONTEXT_SIZE:])
    candidates_text = format_candidates(retrieved, markov)
    user_prompt = f"Переписка:\n{context_text}\n\nВарианты:\n{candidates_text}"

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# --- Загрузка ---

print("Загрузка корпуса...")
all_texts = []
all_pairs = []

if os.path.isdir(TG_CORPUS_DIR):
    print("  Telegram:")
    t, p = telegram.load(TG_CORPUS_DIR)
    all_texts.extend(t)
    all_pairs.extend(p)

if os.path.isdir(VK_CORPUS_DIR):
    print("  VK:")
    t, p = vk.load(VK_CORPUS_DIR)
    all_texts.extend(t)
    all_pairs.extend(p)

print(f"Итого: {len(all_texts)} сообщений, {len(all_pairs)} пар")

model_s2 = build_model(all_texts, state_size=2)
model_s3 = build_model(all_texts, state_size=3)
markov_models = [model_s2, model_s3]
reply_pairs = all_pairs
retrieval_index = build_retrieval_index(reply_pairs)
print("Бот запущен.")


# --- Хендлеры ---

def is_triggered(text: str) -> bool:
    return text.lower().startswith(TRIGGER)


def strip_trigger(text: str) -> str:
    return text[len(TRIGGER):].strip()


@bot.message_handler(func=lambda m: m.text and not is_triggered(m.text))
def track_messages(message):
    history = chat_history[message.chat.id]
    history.append(f"{message.from_user.first_name}: {message.text}")
    if len(history) > CONTEXT_SIZE:
        del history[:-CONTEXT_SIZE]


def get_user_text(message) -> str:
    reply_to = message.reply_to_message
    if reply_to and reply_to.text:
        return reply_to.text
    return strip_trigger(message.text)


def safe_handler(func):
    def wrapper(message):
        try:
            func(message)
        except openai.BadRequestError:
            bot.reply_to(message, "Мама запретила отвечать мне на это сообщение (из могилы)")
        except Exception:
            tb = traceback.format_exc()
            bot.reply_to(message, f"Ошибка:\n{tb[-3900:]}")
    return wrapper


@bot.message_handler(func=lambda m: m.text and is_triggered(m.text))
@safe_handler
def handle_parsalio(message):
    bot.send_chat_action(message.chat.id, "typing")
    user_text = get_user_text(message)
    context = chat_history.get(message.chat.id, [])

    if not user_text and not context:
        bot.reply_to(message, model_s2.make_sentence(tries=100) or "...")
        return

    if not user_text and context:
        last = context[-1]
        user_text = last.split(": ", 1)[1] if ": " in last else last

    words = user_text.split() if user_text else []
    markov = generate_candidates(markov_models, words, n=50)

    query = user_text or (context[-1] if context else "")
    retrieved = retrieve_replies(query, reply_pairs, retrieval_index, n=50)
    random.shuffle(retrieved)

    if not retrieved and not markov:
        bot.reply_to(message, "...")
        return

    full_context = list(context)
    if user_text:
        full_context.append(user_text)

    reply = pick_best(full_context, retrieved, markov)
    bot.reply_to(message, reply)


bot.infinity_polling(restart_on_change=False, timeout=60)
