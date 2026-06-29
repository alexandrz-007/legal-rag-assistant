"""
rag.py — RAG-пайплайн (Retrieval-Augmented Generation).

Главный модуль системы. Соединяет векторный поиск и LLM.

Поток:
  1. Вопрос пользователя → semantic search → топ-K чанков
  2. Формирование контекста: чанки + их источники
  3. Системный промпт для юридического Q&A
  4. LLM генерирует ответ на основе контекста
  5. Возврат: ответ + список источников (документ, страница, фрагмент)

Ключевые принципы:
- Если в контексте нет ответа → LLM должен сказать "недостаточно данных"
- Обязательно ссылки на источник: [документ, страница N]
- Цитирование конкретных пунктов/статей
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SEARCH_TOP_K,
)
from src.vectorstore import search_with_scores, init_vectorstore


# ── Структура ответа ───────────────────────────────────────────────

@dataclass
class Source:
    """Источник, на который ссылается ответ."""
    filename: str          # Имя файла документа
    page: int              # Номер страницы
    fragment: str          # Фрагмент текста (первые 200 символов)
    relevance: float       # Оценка релевантности (0-1)


@dataclass
class Answer:
    """Полный ответ RAG-системы."""
    question: str          # Исходный вопрос
    answer: str            # Ответ LLM
    sources: List[Source]  # Источники, на основе которых дан ответ
    context_chunks: int    # Сколько чанков использовано


# ── Системный промпт ───────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — юридический AI-ассистент. Отвечай на вопросы на основе предоставленных документов.

ПРАВИЛА:
1. Внимательно прочитай ВСЕ предоставленные фрагменты. Информация может быть в любом из них.
2. Если в контексте ЕСТЬ ответ или хотя бы часть ответа — сформулируй чёткий ответ. Не отказывайся отвечать, если данные присутствуют, даже частично.
3. Отказывайся отвечать ТОЛЬКО если в контексте действительно нет никакой релевантной информации по вопросу.
4. Указывай источники в формате: [документ, страница N]
5. Цитируй конкретные пункты, статьи, суммы, даты из текста.
6. Если информация есть в нескольких фрагментах — объединяй её в один ответ.
7. Отвечай на русском языке, чётко и по делу.

ВАЖНО: Не путай "информации нет" с "информация не полностью соответствует вопросу". Если есть хотя бы релевантные данные — отвечай на их основе.

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}
"""


# ── LLM клиент ─────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    """
    Создать LLM-клиент через OpenRouter.

    OpenRouter совместим с OpenAI API, поэтому используем ChatOpenAI.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# ── Сборка контекста ───────────────────────────────────────────────

def build_context(results) -> str:
    """
    Собрать контекст из результатов поиска.

    Формат:
    ---
    [Источник: filename, страница N]
    Текст чанка...
    ---
    """
    context_parts = []

    for i, (doc, score) in enumerate(results, 1):
        filename = doc.metadata.get("filename", "неизвестно")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()

        context_parts.append(
            f"--- Фрагмент {i} ---\n"
            f"[Источник: {filename}, страница {page}]\n"
            f"Релевантность: {score:.2f}\n"
            f"Текст:\n{text}\n"
        )

    return "\n".join(context_parts)


# ── Главная функция ────────────────────────────────────────────────

def answer_question(question: str, k: int = SEARCH_TOP_K) -> Answer:
    """
    Ответить на вопрос пользователя через RAG-пайплайн.

    Шаги:
    1. Семантический поиск релевантных чанков
    2. Сборка контекста с метаданными источников
    3. Генерация ответа через LLM
    4. Формирование списка источников

    Args:
        question: вопрос пользователя
        k: сколько чанков достать из векторной БД

    Returns:
        Answer: ответ + источники + метаданные
    """
    # 1. Поиск релевантных чанков
    results = search_with_scores(question, k=k)

    if not results:
        return Answer(
            question=question,
            answer="В базе документов нет данных. Сначала загрузите и индексируйте документы.",
            sources=[],
            context_chunks=0,
        )

    # 2. Сборка контекста
    context = build_context(results)

    # 3. Формирование промпта
    prompt = SYSTEM_PROMPT.format(context=context)

    # 4. Вызов LLM
    llm = get_llm()

    try:
        # Пробуем основную модель
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ])
        answer_text = response.content
    except Exception as e:
        # Fallback на запасную модель
        print(f"  [rag] Ошибка с {LLM_MODEL}: {e}")
        print(f"  [rag] Переключаюсь на {LLM_FALLBACK_MODEL}")
        llm = ChatOpenAI(
            model=LLM_FALLBACK_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ])
        answer_text = response.content

    # 5. Формирование списка источников
    sources = []
    for doc, score in results:
        sources.append(Source(
            filename=doc.metadata.get("filename", "неизвестно"),
            page=doc.metadata.get("page", 0),
            fragment=doc.page_content[:200].replace("\n", " ") + "...",
            relevance=round(score, 3),
        ))

    return Answer(
        question=question,
        answer=answer_text,
        sources=sources,
        context_chunks=len(results),
    )


# ── Утилита индексации ─────────────────────────────────────────────

def index_documents(force_recreate: bool = False):
    """
    Полный цикл индексации: загрузка → чанкинг → векторная БД.

    Вызывается из UI или скрипта при первичной настройке.
    """
    from src.loader import load_all_documents
    from src.chunker import chunk_documents
    from src.vectorstore import add_documents

    print("\n=== Индексация документов ===")

    # 1. Загрузка
    print("\n[1/3] Загрузка документов...")
    docs = load_all_documents()
    if not docs:
        print("  Нет документов для индексации!")
        return 0

    # 2. Чанкинг
    print("\n[2/3] Разбиение на чанки...")
    chunks = chunk_documents(docs)

    # 3. Векторизация
    print("\n[3/3] Векторизация и сохранение...")
    # Пересоздаём БД если нужно
    init_vectorstore(force_recreate=force_recreate)
    add_documents(chunks)

    print(f"\n=== Готово! Индексировано {len(chunks)} чанков ===")
    return len(chunks)
