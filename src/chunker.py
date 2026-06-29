"""
chunker.py — Разбиение документов на чанки (фрагменты).

RAG-системы работают не с целыми документами, а с небольшими фрагментами.
Это нужно потому что:
1. LLM имеет ограниченный контекст — нельзя запихнуть весь документ
2. Поиск по маленьким чанкам точнее — находим конкретный абзац, а не весь документ

Используем RecursiveCharacterTextSplitter из LangChain:
- Сначала разбивает по абзацам (двойной перенос строки)
- Если чанк слишком большой — по одинарному переносу
- Если всё ещё большой — по предложениям (точка)
- Метаданные сохраняются в каждый чанк
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Разбить список документов на чанки.

    Args:
        documents: список Document (каждый = одна страница PDF)

    Returns:
        список чанков Document, каждый ~CHUNK_SIZE символов
        с overlap=CHUNK_OVERLAP и унаследованными метаданными
    """
    # Создаём сплиттер с иерархией разделителей
    # Сначала пытаемся разбить по абзацам, потом по строкам, потом по точкам
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,  # Сохраняем разделитель в начале чанка
    )

    # Разбиваем — метаданные копируются в каждый чанк автоматически
    chunks = splitter.split_documents(documents)

    print(f"  [chunker] {len(documents)} страниц → {len(chunks)} чанков")
    print(f"  [chunker] Средний размер чанка: "
          f"{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} символов")

    return chunks
