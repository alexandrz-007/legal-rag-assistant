"""
vectorstore.py — Векторное хранилище на базе ChromaDB + гибридный поиск.

Гибридный поиск = семантический (embeddings) + ключевые слова (BM25).
Это решает проблему: embeddings плохо находят конкретные суммы/проценты,
а BM25 — находит. Вместе они покрывают оба случая.

ChromaDB — локальная векторная БД, не требует внешних сервисов.
Embedding-модель: all-MiniLM-L6-v2 (через ONNX, работает на CPU).
"""
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from src.config import CHROMA_DIR, SEARCH_TOP_K

# Имя коллекции в ChromaDB
COLLECTION_NAME = "legal_documents"

# Глобальные кэши
_vectorstore: Optional[Chroma] = None
_bm25_retriever: Optional[BM25Retriever] = None


def get_embeddings():
    """all-MiniLM-L6-v2 — маленькая, быстрая, мультиязычная."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.join(CHROMA_DIR, "models"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_vectorstore(force_recreate: bool = False) -> Chroma:
    """
    Инициализировать векторное хранилище.
    При загрузке восстанавливает BM25-индекс из ChromaDB.

    Args:
        force_recreate: если True — удалить старую БД и создать заново
    """
    global _vectorstore, _bm25_retriever

    if force_recreate:
        if _vectorstore is not None:
            try:
                _vectorstore._client.reset()
            except Exception:
                pass
            _vectorstore = None
        _bm25_retriever = None

        # Удаляем папку с retry (Windows может держать файлы)
        if os.path.exists(CHROMA_DIR):
            import shutil
            import time
            for attempt in range(3):
                try:
                    shutil.rmtree(CHROMA_DIR)
                    break
                except PermissionError:
                    time.sleep(1)
            os.makedirs(CHROMA_DIR, exist_ok=True)

    if _vectorstore is not None:
        # Если BM25 ещё не построен — восстанавливаем из ChromaDB
        if _bm25_retriever is None:
            _rebuild_bm25_from_chroma()
        return _vectorstore

    embeddings = get_embeddings()
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Восстанавливаем BM25-индекс из ChromaDB
    _rebuild_bm25_from_chroma()

    return _vectorstore


def _rebuild_bm25_from_chroma():
    """
    Восстановить BM25-индекс из ChromaDB.
    Достаём все чанки из коллекции и строим BM25Retriever.
    Вызывается при каждом старте процесса, если BM25 ещё не построен.
    """
    global _bm25_retriever

    if _bm25_retriever is not None:
        return  # Уже построен

    if _vectorstore is None:
        return  # ChromaDB ещё не инициализирована

    try:
        collection = _vectorstore._collection
        result = collection.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        if chunks:
            _bm25_retriever = BM25Retriever.from_documents(chunks)
            _bm25_retriever.k = SEARCH_TOP_K
            print(f"  [vectorstore] BM25 восстановлен из ChromaDB: {len(chunks)} чанков")
    except Exception as e:
        print(f"  [vectorstore] BM25 восстановление не удалось: {e}")


def add_documents(chunks: List[Document]):
    """
    Добавить чанки в векторную БД + создать BM25-индекс.
    ChromaDB: embeddings + метаданные + векторный индекс
    BM25: текстовый индекс для поиска по ключевым словам
    """
    global _bm25_retriever

    vs = init_vectorstore()
    vs.add_documents(chunks)

    # Строим BM25-индекс
    _bm25_retriever = BM25Retriever.from_documents(chunks)
    _bm25_retriever.k = SEARCH_TOP_K

    print(f"  [vectorstore] Добавлено чанков: {len(chunks)} (ChromaDB + BM25)")


def search(query: str, k: int = SEARCH_TOP_K) -> List[Document]:
    """Простой семантический поиск (без BM25)."""
    vs = init_vectorstore()
    return vs.similarity_search(query, k=k)


def search_with_scores(query: str, k: int = SEARCH_TOP_K) -> List[Tuple[Document, float]]:
    """
    ГИБРИДНЫЙ ПОИСК: семантика (ChromaDB) + ключевые слова (BM25).

    1. Семантический поиск: top-k чанков по embedding-сходству
    2. BM25-поиск: top-k чанков по совпадению ключевых слов
    3. Объединение: дедупликация по тексту чанка (не по id объекта)

    Returns:
        список кортежей (Document, score)
    """
    vs = init_vectorstore()

    # 1. Семантический поиск
    semantic_results = vs.similarity_search_with_relevance_scores(query, k=k)

    # Дедупликация по тексту чанка (не по id объекта!)
    seen_texts = set()
    combined = []

    for doc, score in semantic_results:
        text_key = doc.page_content[:200]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            combined.append((doc, score))

    # 2. BM25-поиск (если индекс построен)
    if _bm25_retriever is not None:
        try:
            bm25_results = _bm25_retriever.invoke(query)
            avg_score = 0.3  # Базовая оценка для BM25-находок
            for doc in bm25_results:
                text_key = doc.page_content[:200]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    combined.append((doc, avg_score))
        except Exception as e:
            print(f"  [vectorstore] BM25 ошибка: {e}")

    # Ограничиваем общее количество
    return combined[:k + 4]  # Немного больше для контекста


def get_collection_count() -> int:
    """Быстрая проверка количества чанков без загрузки embeddings."""
    try:
        if not os.path.exists(CHROMA_DIR):
            return 0
        for item in os.listdir(CHROMA_DIR):
            item_path = os.path.join(CHROMA_DIR, item)
            if os.path.isdir(item_path) and item != "models":
                bin_files = [f for f in os.listdir(item_path) if f.endswith((".bin", ".arrow"))]
                if bin_files:
                    return len(bin_files)
        return 0
    except Exception:
        return 0
