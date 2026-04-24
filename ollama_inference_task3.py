import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import requests


DEFAULT_PROMPTS = [
    "Объясни простыми словами, что такое градиентный спуск.",
    "Составь план изучения Python на 2 недели для новичка.",
    "Чем precision отличается от recall?",
    "Напиши 3 идеи pet-проектов по машинному обучению.",
    "Как работает свертка в CNN?",
    "Объясни разницу между overfitting и underfitting.",
    "Для чего нужен train/validation/test split?",
    "Что такое token в LLM?",
    "Приведи пример задачи, где лучше использовать классификацию, а не регрессию.",
    "Дай краткий чек-лист подготовки данных перед обучением модели.",
]


def load_prompts(prompts_file: str = "") -> List[str]:
    """Load prompts from JSON file or return default list of 10 prompts."""
    if not prompts_file:
        return DEFAULT_PROMPTS
    path = Path(prompts_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("prompts file must be a JSON array of strings")
    if len(data) < 10:
        raise ValueError("provide at least 10 prompts")
    return data[:10]


def ask_ollama(prompt: str, model: str, base_url: str) -> str:
    """Send one prompt to Ollama HTTP API and return model response text."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def run_inference(prompts: List[str], model: str, base_url: str) -> List[Tuple[str, str]]:
    """Run sequential inference for provided prompts and collect pairs."""
    results: List[Tuple[str, str]] = []
    for prompt in prompts:
        answer = ask_ollama(prompt, model=model, base_url=base_url)
        results.append((prompt, answer))
    return results


def save_reports(results: List[Tuple[str, str]], output_dir: Path) -> None:
    """Save inference report in CSV and Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "inference_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "llm_output"])
        writer.writerows(results)

    md_path = output_dir / "inference_report.md"
    lines = [
        "# Отчет инференса (ЛР2, оценка 3)",
        "",
        "| Запрос к LLM | Вывод LLM |",
        "|---|---|",
    ]
    for q, a in results:
        q_clean = q.replace("|", "\\|").replace("\n", " ")
        a_clean = a.replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {q_clean} | {a_clean} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved CSV report: {csv_path}")
    print(f"Saved Markdown report: {md_path}")


def main() -> None:
    """Entry point for lab task 3 NLP inference runner."""
    parser = argparse.ArgumentParser(description="Ollama inference task for grade 3")
    parser.add_argument("--model", type=str, default="qwen2.5:0.5b", help="Ollama model name")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="Optional JSON file with prompts",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    results = run_inference(prompts, model=args.model, base_url=args.base_url)
    save_reports(results, Path("nlp/reports"))


if __name__ == "__main__":
    main()
