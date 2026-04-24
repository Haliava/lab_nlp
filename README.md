# ЛР2 NLP (оценка 3): Ollama + Qwen2.5:0.5b

## Что реализовано

Скрипт `ollama_inference_task3.py`:

1. Отправляет HTTP-запросы к локальному серверу Ollama.
2. Прогоняет 10 запросов (можно передать свои).
3. Сохраняет отчет с двумя столбцами:
   - запрос к LLM
   - вывод LLM

## Предусловия

1. Установлен Ollama.
2. Загружена модель:

```bash
ollama pull qwen2.5:0.5b
```

3. Запущен сервер Ollama (обычно автоматически на `http://localhost:11434`).

## Запуск

```bash
python nlp/ollama_inference_task3.py --model qwen2.5:0.5b
```

Опционально можно передать свои запросы через файл JSON:

```bash
python nlp/ollama_inference_task3.py --prompts-file nlp/custom_prompts.json
```

## Выходные файлы

- `nlp/reports/inference_report.csv`
- `nlp/reports/inference_report.md`
