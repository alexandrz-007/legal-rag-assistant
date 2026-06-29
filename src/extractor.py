"""
extractor.py — Извлечение сущностей из юридических документов.

Использует LLM для извлечения структурированных данных из текста:
- Стороны (наименования, ФИО, роли)
- Даты (заключения, действия, окончания)
- Суммы (стоимость, штрафы, пени)
- Сроки (хранения, действия, рассмотрения)
- Номера статей и пунктов
- Типы клаузул (NDA, ответственность, форс-мажор)

Возвращает JSON-структуру.
"""
import json
from typing import Dict, List
from dataclasses import dataclass, asdict

from langchain_openai import ChatOpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_TEMPERATURE,
)


# ── Структура результата ───────────────────────────────────────────

@dataclass
class ExtractedEntities:
    """Извлечённые из документа сущности."""
    document_type: str            # Тип документа (договор, NDA, политика...)
    parties: List[Dict]           # Стороны [{name, role, representative}]
    dates: List[Dict]             # Даты [{date, description}]
    amounts: List[Dict]           # Суммы [{amount, currency, description}]
    deadlines: List[Dict]         # Сроки [{period, description}]
    clauses: List[Dict]           # Клаузулы [{type, description}]
    key_terms: List[str]          # Ключевые термины


# ── Промпт для извлечения ──────────────────────────────────────────

EXTRACTION_PROMPT = """Извлеки из юридического текста следующие сущности и верни в формате JSON.

ТЕКСТ ДОКУМЕНТА:
{text}

ЗАДАЧА: Верни СТРОГО валидный JSON (без markdown, без пояснений) следующей структуры:

{{
  "document_type": "тип документа (например: договор, NDA, политика, оферта, регламент, претензия, доверенность)",
  "parties": [
    {{"name": "наименование или ФИО", "role": "роль (Исполнитель/Заказчик/Арендатор и т.д.)", "representative": "ФИО представителя"}}
  ],
  "dates": [
    {{"date": "дата", "description": "что это за дата"}}
  ],
  "amounts": [
    {{"amount": "сумма цифрами", "currency": "валюта", "description": "за что"}}
  ],
  "deadlines": [
    {{"period": "срок", "description": "на что"}}
  ],
  "clauses": [
    {{"type": "тип (NDA/ответственность/форс-мажор/конфиденциальность/расторжение)", "description": "краткое описание"}}
  ],
  "key_terms": ["ключевой термин 1", "ключевой термин 2"]
}}

ПРАВИЛА:
- Если категория пуста — верни пустой массив []
- Суммы указывай цифрами (например: 450000)
- Даты в формате ДД.ММ.ГГГГ если возможно
- Верни ТОЛЬКО JSON, без markdown-обёртки
"""


# ── LLM клиент ─────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """Создать LLM-клиент через OpenRouter."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,  # Максимальная точность для извлечения
    )


# ── Главная функция ────────────────────────────────────────────────

def extract_entities(text: str) -> Dict:
    """
    Извлечь сущности из текста документа.

    Args:
        text: полный текст документа

    Returns:
        Dict с ключами: document_type, parties, dates, amounts,
        deadlines, clauses, key_terms
    """
    llm = get_llm()

    # Обрезаем слишком длинный текст (LLM имеет лимит контекста)
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...текст обрезан...]"

    prompt = EXTRACTION_PROMPT.format(text=text)

    try:
        response = llm.invoke([
            {"role": "system", "content": "Ты — юридический аналитик. Извлекай сущности точно и полно. Возвращай только JSON."},
            {"role": "user", "content": prompt},
        ])
        raw = response.content
    except Exception as e:
        print(f"  [extractor] Ошибка: {e}")
        return _empty_result()

    # Парсим JSON (LLM иногда оборачивает в markdown)
    raw = raw.strip()
    if raw.startswith("```"):
        # Убираем markdown-обёртку
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        print(f"  [extractor] Не удалось распарсить JSON")
        return _empty_result()


def _empty_result() -> Dict:
    """Пустой результат при ошибке."""
    return {
        "document_type": "неизвестно",
        "parties": [],
        "dates": [],
        "amounts": [],
        "deadlines": [],
        "clauses": [],
        "key_terms": [],
    }


def extract_from_file(file_path: str) -> Dict:
    """
    Извлечь сущности из файла (PDF или DOCX).

    Args:
        file_path: путь к файлу

    Returns:
        Dict с извлечёнными сущностями
    """
    from src.loader import load_document

    docs = load_document(file_path)
    if not docs:
        return _empty_result()

    # Объединяем все страницы в один текст
    full_text = "\n".join(d.page_content for d in docs)

    print(f"  [extractor] Извлечение из {len(full_text)} символов...")
    return extract_entities(full_text)
