"""
evaluate.py — Оценка качества RAG-системы.

Метрики:
1. Source Recall — найден ли правильный источник?
2. Answer Faithfulness — содержит ли ответ ожидаемые ключевые слова?
3. Refusal Accuracy — правильно ли отказывается, когда данных нет?

Запуск:
  python -m eval.evaluate           # из корня проекта
  или из UI: вкладка "Оценка"
"""
import json
import os
from typing import List, Dict
from dataclasses import dataclass, asdict

# Добавляем корень проекта в path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import answer_question


# ── Структура результата ───────────────────────────────────────────

@dataclass
class TestCaseResult:
    """Результат одного тест-кейса."""
    id: int
    question: str
    description: str
    should_answer: bool
    answer: str
    sources_found: List[str]
    source_recall: bool          # Найден ли ожидаемый источник
    keyword_match: bool          # Есть ли ключевые слова в ответе
    refusal_correct: bool        # Правильный ли отказ/ответ
    passed: bool                 # Тест пройден


@dataclass
class EvaluationReport:
    """Отчёт по всем тест-кейсам."""
    total: int
    passed: int
    failed: int
    source_recall_rate: float    # % тестов с правильным источником
    faithfulness_rate: float     # % тестов с верными ключевыми словами
    refusal_accuracy: float      # % правильных отказов
    results: List[Dict]


# ── Загрузка тест-кейсов ───────────────────────────────────────────

def load_test_cases() -> List[Dict]:
    """Загрузить тест-кейсы из JSON."""
    cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Проверка одного кейса ──────────────────────────────────────────

def evaluate_single(case: Dict) -> Dict:
    """
    Проверить один тест-кейс.

    Логика:
    - Если should_answer=True: проверяем источник и ключевые слова
    - Если should_answer=False: проверяем, что система отказалась отвечать
    """
    question = case["question"]
    expected_source = case.get("expected_source")
    expected_keywords = case.get("expected_keywords", [])
    should_answer = case["should_answer"]

    # Вызываем RAG
    result = answer_question(question, k=12)

    answer_text = result.answer.lower()
    sources_found = [s.filename for s in result.sources]

    # 1. Source Recall: есть ли ожидаемый источник в результатах?
    source_recall = False
    if expected_source:
        source_recall = any(expected_source in s for s in sources_found)

    # 2. Keyword Match: есть ли ключевые слова в ответе?
    # Проверяем каждое ключевое слово отдельно, хотя бы одно должно совпасть
    keyword_match = False
    if expected_keywords:
        keyword_match = any(kw.lower() in answer_text for kw in expected_keywords)

    # 3. Refusal Accuracy
    # Если не должен отвечать — проверяем что отказался
    # Если должен отвечать — проверяем что не отказался
    refusal_phrases = [
        "нет информации", "недостаточно данных", "не содержится",
        "нет данных", "не удалось найти", "нет сведений",
        "нет ответа", "нет информации по этому вопросу",
        "в предоставленных документах нет",
    ]
    has_refusal = any(phrase in answer_text for phrase in refusal_phrases)

    if should_answer:
        # Должен ответить — отказ НЕ правильный
        refusal_correct = not has_refusal
    else:
        # Не должен отвечать — отказ правильный
        refusal_correct = has_refusal

    # 4. Overall: тест пройден?
    if should_answer:
        # Должен ответить: источник найден + ключевые слова есть + не отказал
        passed = source_recall and keyword_match and refusal_correct
    else:
        # Не должен отвечать: отказался
        passed = refusal_correct

    return {
        "id": case["id"],
        "question": question,
        "description": case.get("description", ""),
        "should_answer": should_answer,
        "answer": result.answer[:500],  # Обрезаем для читаемости
        "sources_found": sources_found[:3],
        "source_recall": source_recall,
        "keyword_match": keyword_match,
        "refusal_correct": refusal_correct,
        "passed": passed,
    }


# ── Запуск всех тестов ─────────────────────────────────────────────

def run_evaluation() -> Dict:
    """
    Запустить оценку по всем тест-кейсам.

    Returns:
        Dict с метриками и результатами
    """
    cases = load_test_cases()
    results = []

    print(f"\n=== Оценка качества: {len(cases)} тест-кейсов ===\n")

    for case in cases:
        print(f"  [{case['id']}/{len(cases)}] {case['question'][:60]}...")
        try:
            result = evaluate_single(case)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"       → {status}")
        except Exception as e:
            print(f"       → ERROR: {e}")
            results.append({
                "id": case["id"],
                "question": case["question"],
                "description": case.get("description", ""),
                "should_answer": case["should_answer"],
                "answer": f"ERROR: {str(e)}",
                "sources_found": [],
                "source_recall": False,
                "keyword_match": False,
                "refusal_correct": False,
                "passed": False,
            })

    # Считаем метрики
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    # Source recall: только для кейсов, где должен отвечать
    answer_cases = [r for r in results if r["should_answer"]]
    source_recall_rate = (
        sum(1 for r in answer_cases if r["source_recall"]) / len(answer_cases)
        if answer_cases else 0
    )
    faithfulness_rate = (
        sum(1 for r in answer_cases if r["keyword_match"]) / len(answer_cases)
        if answer_cases else 0
    )
    refusal_accuracy = (
        sum(1 for r in results if r["refusal_correct"]) / total
        if total else 0
    )

    report = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "source_recall_rate": round(source_recall_rate, 2),
        "faithfulness_rate": round(faithfulness_rate, 2),
        "refusal_accuracy": round(refusal_accuracy, 2),
        "results": results,
    }

    print(f"\n=== Результаты ===")
    print(f"  Пройдено: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  Source Recall: {source_recall_rate*100:.0f}%")
    print(f"  Faithfulness: {faithfulness_rate*100:.0f}%")
    print(f"  Refusal Accuracy: {refusal_accuracy*100:.0f}%")

    return report


if __name__ == "__main__":
    report = run_evaluation()
    # Сохраняем отчёт
    output_path = os.path.join(os.path.dirname(__file__), "report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт сохранён: {output_path}")
