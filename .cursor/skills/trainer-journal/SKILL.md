---
name: trainer-journal
description: >-
  Workout journal with two modes: self (the user trains alone) and trainer (they coach
  a group, each person with a separate schedule). Compose programmes, structure reports
  (exercises, sets, weights) and export CSV. Persists in SQLite. Use when the user
  mentions тренировки, зал, расписание, программу, отчёт о тренировке, жим, присед,
  веса, спортсменов, «я тренер» or asks to keep a training log.
---

# Журнал тренировок

Данные живут в SQLite через MCP `training_*`. Сессия Cursor не память.

## Два режима

Сначала `training_profile_get`. Если режима нет — определи и вызови `training_profile_set`.

**self** — пользователь тренируется сам.
- «сегодня присед 80», «моя программа», «запиши тренировку» без чужих имён.
- Отчёт и расписание без имени идут на него (`is_self`).
- Не спрашивай «для кого», пока не появятся другие люди.

**trainer** — представился тренером, ведёт группу, называет нескольких.
- «я тренер», «подопечные», «группа», «расписание для Васи и Маши».
- Сразу `training_profile_set` с `mode=trainer`.
- Каждый человек — своя запись, своя программа, своё расписание.
- Группа в один день: несколько `training_schedule_upsert`, разные `athlete_id`.
- Отчёт без имени не писать: спроси, кто это. Не угадывай.
- Свои тренировки тренера — явно «я» / `is_self`.

Появление второго человека (не `is_self`) само переключает в `trainer`. Обратно в `self`
только по просьбе.

## Память

- `training_athlete_upsert`, затем программа и логи на `athlete_id`.
- Сначала `training_athlete_list`. Несколько совпадений — спроси.
- Первая запись кладёт факт в `memory` (категория `training`). Не дублируй `memory_remember`.

## Отчёт

Структурируй текст/голос и вызови `training_log_save`:

```json
{
  "local_date": "2026-08-30",
  "athlete_name": "Вася",
  "raw_text": "оригинал",
  "exercises": [
    {"name": "Присед", "sets": [{"reps": 5, "weight_kg": 100, "rpe": 7}]}
  ]
}
```

Вес в кг, если не сказали иначе. В чат — коротко: кто, дата, упражнения и веса.
В ответе всегда скажи, сколько тренировок проведено и сколько осталось (`progress.label`).

## Прогресс

`training_progress` — сколько сделано и сколько осталось. После `training_log_save`
числа уже в результате, отдельно вызывать не обязательно.

При сохранении программы передай длину: `total_sessions` или `weeks` + `days_per_week`
(тогда всего = недели × дни). Без длины «осталось» считается по запланированным сессиям.

В режиме trainer прогресс у каждого свой. Не смешивай.

## Таблица

`training_export` (`logs` / `schedule` / `both`). CSV в чат не вставляй.

## Запрещено

- Держать программу только в ответе.
- Смешивать людей в одной записи.
- Путать с вечерним дневником `journal_*`.
