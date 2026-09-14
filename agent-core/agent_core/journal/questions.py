"""The evening check-in: a short work + personal survey, then two 1–5 scales.

Kept as data so the flow in ``service.py`` is just "ask the current step".
"""

from __future__ import annotations

from pa_protocol import methods

OFFER = "offer"
WORK = "work"
PERSONAL = "personal"
MOOD = "mood"
PROGRESS = "progress"
TOMORROW = "tomorrow"
DONE = "done"

ORDER = (OFFER, WORK, PERSONAL, MOOD, PROGRESS, TOMORROW)
TEXT_STEPS = frozenset({WORK, PERSONAL, TOMORROW})
CHOICE_STEPS = frozenset({OFFER, MOOD, PROGRESS})

MOOD_LABELS = {1: "плохо", 2: "так себе", 3: "нормально", 4: "хорошо", 5: "отлично"}
PROGRESS_LABELS = {
    1: "стою",
    2: "чуть-чуть",
    3: "есть движение",
    4: "заметно",
    5: "сильно",
}

_FILL = methods.ConfirmAction(id="fill", label="Заполнить", style="primary")
_SKIP = methods.ConfirmAction(id="skip", label="Пропустить", style="secondary")
_SKIP_OPTIONAL = methods.ConfirmAction(id="skip", label="Пропустить", style="secondary")

_MOOD = tuple(
    methods.ConfirmAction(id=str(n), label=str(n), style="primary" if n == 3 else "secondary")
    for n in range(1, 6)
)
_PROGRESS = _MOOD


def next_step(current: str) -> str:
    try:
        index = ORDER.index(current)
    except ValueError:
        return DONE
    if index + 1 >= len(ORDER):
        return DONE
    return ORDER[index + 1]


def prompt_for(step: str, *, date_label: str) -> str:
    if step == OFFER:
        return (
            f"Дневник за {date_label}.\n\n"
            "Короткий вечерний опрос: работа и личное, потом две оценки. "
            "Ответы соберу и в конце месяца подведу итог."
        )
    if step == WORK:
        return (
            "Работа — что сделал сегодня, что сдвинулось, где застрял?\n"
            "Можно текстом или голосом."
        )
    if step == PERSONAL:
        return (
            "Личное — как день, что важного, как самочувствие?\n"
            "Можно текстом или голосом."
        )
    if step == MOOD:
        return "Настроение сегодня? 1 — плохо, 5 — отлично."
    if step == PROGRESS:
        return "Ощущение прогресса по жизни? 1 — стою, 5 — сильно сдвинулось."
    if step == TOMORROW:
        return "Что забрать в завтра? Можно пропустить."
    return "Дневник."


def actions_for(step: str) -> list[methods.ConfirmAction]:
    if step == OFFER:
        return [_FILL, _SKIP]
    if step == MOOD:
        return list(_MOOD)
    if step == PROGRESS:
        return list(_PROGRESS)
    if step in TEXT_STEPS:
        return [_SKIP_OPTIONAL]
    return []
