"""
app.py — Streamlit веб-интерфейс для Legal RAG Assistant.

Запуск:
    streamlit run app.py

Три вкладки:
1. 💬 Чат — диалог с RAG-системой, каждый ответ с источниками
2. 🏷 Сущности — загрузить документ → извлечь стороны, даты, суммы
3. 📊 Оценка — запустить тест-кейсы → таблица результатов

Кэширование: @st.cache_resource — embeddings грузятся ОДИН раз за сессию.
"""
import os
import sys
import json

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.config import OPENROUTER_API_KEY, DOCS_DIR, UPLOAD_DIR, CHROMA_DIR, SEARCH_TOP_K
from src.rag import answer_question, index_documents
from src.vectorstore import get_collection_count, init_vectorstore
from src.extractor import extract_from_file, extract_entities
from src.loader import load_document


# ── Настройка страницы ─────────────────────────────────────────────

st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Кэширование ресурсов (грузятся ОДИН раз) ──────────────────────

@st.cache_resource(show_spinner=False)
def get_cached_vectorstore():
    """
    Получить векторное хранилище.
    Кэшируется Streamlit'ом — embeddings грузятся один раз за сессию,
    а не при каждом нажатии кнопки.
    """
    from src.vectorstore import init_vectorstore
    return init_vectorstore()


def init_state():
    """Инициализировать состояние сессии Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "chunk_count" not in st.session_state:
        try:
            st.session_state.chunk_count = get_collection_count()
        except Exception:
            st.session_state.chunk_count = 0


init_state()

# Прогреваем векторное хранилище (кешируется)
_vs = get_cached_vectorstore()


# ── Боковая панель ─────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ Legal RAG Assistant")
    st.caption("RAG-система для юридических документов")

    st.divider()

    # Статус API
    if OPENROUTER_API_KEY:
        st.success("✅ OpenRouter API подключён")
    else:
        st.error("❌ OpenRouter API ключ не найден")
        st.caption("Добавьте ключ в .env файл")

    st.divider()

    # Статус БД
    st.subheader("📊 Статус базы")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Чанков в БД", st.session_state.chunk_count)
    with col2:
        # Считаем PDF файлы
        pdf_count = len([f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]) if os.path.exists(DOCS_DIR) else 0
        st.metric("Документов", pdf_count)

    st.divider()

    # Индексация
    st.subheader("📁 Индексация")
    if st.button("🔄 Переиндексировать", type="primary", use_container_width=True):
        # Сбрасываем кэш векторного хранилища
        get_cached_vectorstore.clear()
        with st.spinner("Индексация документов..."):
            count = index_documents(force_recreate=True)
            st.session_state.chunk_count = get_collection_count()
            st.session_state.indexed = True
        st.success(f"✅ Индексировано {count} чанков")
        st.rerun()

    # Загрузка файлов
    st.subheader("📤 Загрузить документ")
    uploaded = st.file_uploader(
        "PDF или DOCX",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filepath = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(filepath, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ {uploaded.name} загружен")
        st.caption("Нажмите «Переиндексировать» чтобы добавить в базу")


# ── Главная область ────────────────────────────────────────────────

tab_chat, tab_entities, tab_eval = st.tabs(["💬 Чат", "🏷 Сущности", "📊 Оценка"])


# ── Вкладка 1: Чат ─────────────────────────────────────────────────

with tab_chat:
    st.header("💬 Чат с юридическими документами")

    if st.session_state.chunk_count == 0:
        st.warning("⚠️ База пуста. Нажмите «Переиндексировать» в боковой панели.")
    else:
        st.caption(f"База: {st.session_state.chunk_count} чанков · Задайте вопрос по документам")

    # История чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Показываем источники для ответов ассистента
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(f"📚 Источники ({len(message['sources'])})"):
                    for src in message["sources"]:
                        st.write(
                            f"**{src['filename']}** · стр. {src['page']} "
                            f"· релевантность: {src['relevance']}"
                        )
                        st.caption(src["fragment"])

    # Поле ввода
    if prompt := st.chat_input("Напишите вопрос..."):
        # Добавляем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Генерируем ответ
        with st.chat_message("assistant"):
            with st.spinner("Поиск по документам..."):
                result = answer_question(prompt, k=SEARCH_TOP_K)

            st.markdown(result.answer)

            # Источники
            if result.sources:
                with st.expander(f"📚 Источники ({len(result.sources)})"):
                    for src in result.sources:
                        st.write(
                            f"**{src.filename}** · стр. {src.page} "
                            f"· релевантность: {src.relevance}"
                        )
                        st.caption(src.fragment)

            # Сохраняем в историю
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": [
                    {
                        "filename": s.filename,
                        "page": s.page,
                        "relevance": s.relevance,
                        "fragment": s.fragment,
                    }
                    for s in result.sources
                ],
            })


# ── Вкладка 2: Сущности ───────────────────────────────────────────

with tab_entities:
    st.header("🏷 Извлечение сущностей")
    st.caption("Загрузите документ — система извлечёт стороны, даты, суммы, сроки")

    # Выбор: загрузить новый или использовать существующий
    entity_method = st.radio(
        "Способ выбора документа:",
        ["Загрузить файл", "Выбрать из базы"],
        horizontal=True,
    )

    target_path = None
    target_name = None

    if entity_method == "Загрузить файл":
        entity_file = st.file_uploader(
            "PDF или DOCX для извлечения",
            type=["pdf", "docx"],
            key="entity_uploader",
        )
        if entity_file:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            target_path = os.path.join(UPLOAD_DIR, entity_file.name)
            with open(target_path, "wb") as f:
                f.write(entity_file.getbuffer())
            target_name = entity_file.name
    else:
        # Выбор из существующих документов
        if os.path.exists(DOCS_DIR):
            doc_files = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".docx"))])
            if doc_files:
                selected = st.selectbox("Документ:", doc_files)
                target_path = os.path.join(DOCS_DIR, selected)
                target_name = selected
            else:
                st.warning("Нет документов в базе")

    if target_path and target_name:
        st.info(f"📄 Выбран: **{target_name}**")

        if st.button("🔍 Извлечь сущности", type="primary"):
            with st.spinner("Извлечение сущностей..."):
                result = extract_from_file(target_path)

            # Тип документа
            st.subheader(f"Тип документа: {result.get('document_type', 'неизвестно')}")

            col1, col2 = st.columns(2)

            with col1:
                # Стороны
                st.markdown("### 👥 Стороны")
                parties = result.get("parties", [])
                if parties:
                    for p in parties:
                        st.write(f"**{p.get('name', '?')}**")
                        st.caption(f"Роль: {p.get('role', '?')} · Представитель: {p.get('representative', '?')}")
                else:
                    st.caption("Не найдены")

                # Суммы
                st.markdown("### 💰 Суммы")
                amounts = result.get("amounts", [])
                if amounts:
                    for a in amounts:
                        st.write(f"**{a.get('amount', '?')} {a.get('currency', 'руб.')}** — {a.get('description', '')}")
                else:
                    st.caption("Не найдены")

                # Клаузулы
                st.markdown("### 📋 Клаузулы")
                clauses = result.get("clauses", [])
                if clauses:
                    for c in clauses:
                        st.write(f"**{c.get('type', '?')}**: {c.get('description', '')}")
                else:
                    st.caption("Не найдены")

            with col2:
                # Даты
                st.markdown("### 📅 Даты")
                dates = result.get("dates", [])
                if dates:
                    for d in dates:
                        st.write(f"**{d.get('date', '?')}** — {d.get('description', '')}")
                else:
                    st.caption("Не найдены")

                # Сроки
                st.markdown("### ⏰ Сроки")
                deadlines = result.get("deadlines", [])
                if deadlines:
                    for dl in deadlines:
                        st.write(f"**{dl.get('period', '?')}** — {dl.get('description', '')}")
                else:
                    st.caption("Не найдены")

                # Ключевые термины
                st.markdown("### 🔑 Ключевые термины")
                terms = result.get("key_terms", [])
                if terms:
                    for t in terms:
                        st.write(f"`{t}`")
                else:
                    st.caption("Не найдены")


# ── Вкладка 3: Оценка ─────────────────────────────────────────────

with tab_eval:
    st.header("📊 Оценка качества RAG-системы")
    st.caption("10 тест-кейсов: 9 вопросов по документам + 1 вопрос вне базы")

    if st.button("▶️ Запустить оценку", type="primary"):
        from eval.evaluate import run_evaluation

        with st.spinner("Запуск тестов (это займёт ~1-2 минуты)..."):
            report = run_evaluation()

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pct = report["passed"] / report["total"] * 100
            st.metric("Пройдено", f"{report['passed']}/{report['total']}", f"{pct:.0f}%")
        with col2:
            st.metric("Source Recall", f"{report['source_recall_rate']*100:.0f}%")
        with col3:
            st.metric("Faithfulness", f"{report['faithfulness_rate']*100:.0f}%")
        with col4:
            st.metric("Refusal Accuracy", f"{report['refusal_accuracy']*100:.0f}%")

        st.divider()

        # Таблица результатов
        st.subheader("Детали")
        for r in report["results"]:
            # Иконка статуса
            icon = "✅" if r["passed"] else "❌"
            with st.expander(f"{icon} Тест {r['id']}: {r['question'][:70]}..."):
                st.write(f"**Вопрос:** {r['question']}")
                st.write(f"**Описание:** {r['description']}")
                st.write(f"**Должен отвечать:** {'Да' if r['should_answer'] else 'Нет'}")
                st.write(f"**Ответ:** {r['answer']}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Source Recall", "✅" if r["source_recall"] else "❌")
                with col_b:
                    st.metric("Keyword Match", "✅" if r["keyword_match"] else "❌")
                with col_c:
                    st.metric("Refusal Correct", "✅" if r["refusal_correct"] else "❌")

                if r["sources_found"]:
                    st.write("**Источники:**", ", ".join(r["sources_found"]))
