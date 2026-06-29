FROM python:3.11-slim

# Системные зависимости для PyMuPDF и ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements и ставим зависимости (кэш Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники
COPY . .

# Порт Streamlit
EXPOSE 8501

# Проверка здоровья
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
