"""Табличное представление результата (фрагменты-источники) для UI."""

import json
from typing import Any


def _sort_key_index(key: str) -> tuple[int, str]:
    try:
        return (0, int(key))
    except ValueError:
        return (1, key)


def sources_dict_to_rows(sources: Any) -> list[dict[str, str]]:
    """
    Преобразует объект «источники» в строки таблицы: номер ссылки и текст фрагмента.
    """
    if not isinstance(sources, dict):
        return []
    rows: list[dict[str, str]] = []
    for k in sorted(sources.keys(), key=lambda x: _sort_key_index(str(x))):
        v = sources[k]
        if isinstance(v, str):
            text = v
        else:
            text = json.dumps(v, ensure_ascii=False)
        rows.append({"№": str(k), "Фрагмент текста": text})
    return rows


def pick_sources_for_table(result: dict[str, Any]) -> dict[str, Any] | None:
    """Берёт блок источников из агрегата или, при отсутствии, из генераторов."""
    agg = result.get("aggregated_result") or {}
    src = agg.get("источники")
    if isinstance(src, dict) and len(src) > 0:
        return src
    for key in ("generator_1_output", "generator_2_output"):
        block = result.get(key) or {}
        s = block.get("источники")
        if isinstance(s, dict) and len(s) > 0:
            return s
    return None
