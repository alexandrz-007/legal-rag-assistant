"""
config.py — Центральная конфигурация проекта.
Все настройки в одном месте: пути, модель, параметры чанкинга.
"""
import os
from dotenv import load_dotenv

# Загружаем .env из корня проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── API ключи ──────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Модель для генерации ответов (быстрая и дешёвая)
LLM_MODEL = "google/gemini-2.5-flash"
LLM_FALLBACK_MODEL = "openai/gpt-4o-mini"

# ── Пути ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCS_DIR = os.path.join(DATA_DIR, "generated")        # PDF из generate_docs.py
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")          # Загруженные через UI
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")    # Векторная БД

# ── Параметры чанкинга ────────────────────────────────────────────
CHUNK_SIZE = 1000       # Размер чанка в символах (больше = больше контекста)
CHUNK_OVERLAP = 200     # Перекрытие между чанками (контекст)

# ── Параметры поиска ──────────────────────────────────────────────
SEARCH_TOP_K = 12       # Сколько чанков доставать из векторной БД

# ── Параметры LLM ─────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1   # Низкая температура = менее креативный, более точный
LLM_MAX_TOKENS = 2000   # Максимум токенов в ответе


def ensure_dirs():
    """Создать все нужные папки, если их нет."""
    for d in [DATA_DIR, DOCS_DIR, UPLOAD_DIR, CHROMA_DIR]:
        os.makedirs(d, exist_ok=True)


ensure_dirs()
