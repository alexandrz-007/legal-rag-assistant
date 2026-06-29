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
_all_chunks: List[Document] = []  # Храним все чанки для BM25


def get_embeddings():
    """
    Получить embedding-модель.
    all-MiniLM-L6-v2 — маленькая, быстрая, мультиязычная.
    """
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

    Args:
        force_recreate: если True — удалить старую БД и создать заново
    """
    global _vectorstore, _bm25_retriever, _all_chunks

    if force_recreate:
        # Закрываем текущее соединение
        if _vectorstore is not None:
            try:
                _vectorstore._client.reset()
            except Exception:
                pass
            _vectorstore = None
        _bm25_retriever = None
        _all_chunks = []

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
        return _vectorstore

    embeddings = get_embeddings()
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return _vectorstore


def add_documents(chunks: List[Document]):
    """
    Добавить чанки в векторную БД + создать BM25-индекс.

    ChromaDB: embeddings + метаданные + векторный индекс
    BM25: текстовый индекс для поиска по ключевым словам
    """
    global _bm25_retriever, _all_chunks

    vs = init_vectorstore()
    vs.add_documents(chunks)

    # Сохраняем чанки и строим BM25-индекс
    _all_chunks = chunks
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
    3. Объединение: уникальные чанки, приоритет — семантическим,
       но BM25-находки добавляются если их нет в семантической выдаче

    Returns:
        список кортежей (Document, score)
    """
    vs = init_vectorstore()

    # 1. Семантический поиск
    semantic_results = vs.similarity_search_with_relevance_scores(query, k=k)
    semantic_docs = {id(doc): (doc, score) for doc, score in semantic_results}

    # 2. BM25-поиск (если индекс построен)
    bm25_docs = []
    if _bm25_retriever is not None:
        try:
            bm25_results = _bm25_retriever.invoke(query)
            # Присваиваем BM25-результатам средний score
            avg_score = 0.3  # Базовая оценка для BM25-находок
            for doc in bm25_results:
                doc_id = id(doc)
                if doc_id not in semantic_docs:
                    bm25_docs.append((doc, avg_score))
        except Exception as e:
            print(f"  [vectorstore] BM25 ошибка: {e}")

    # 3. Объединение: сначала семантические, потом BM25-дополнения
    combined = list(semantic_results)
    for doc, score in bm25_docs:
        if id(doc) not in semantic_docs:
            combined.append((doc, score))

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
