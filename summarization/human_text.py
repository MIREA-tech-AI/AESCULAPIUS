"""Преобразование агрегированного JSON в человекочитаемый текст для UI и экспорта."""

from __future__ import annotations

import json
from typing import Any


def _sort_source_keys(keys: list[Any]) -> list[Any]:
    def key_fn(k: Any) -> tuple[int, str]:
        s = str(k)
        try:
            return (0, f"{int(s):010d}")
        except ValueError:
            return (1, s)

    return sorted(keys, key=key_fn)


def _format_nested(val: Any, indent: str = "") -> list[str]:
    """Рекурсивно форматирует значение в строки с отступами."""
    lines: list[str] = []
    if val is None:
        lines.append(f"{indent}—")
        return lines
    if isinstance(val, str):
        for part in val.strip().splitlines() or [""]:
            lines.append(f"{indent}{part}".rstrip())
        return lines
    if isinstance(val, (int, float, bool)):
        lines.append(f"{indent}{val}")
        return lines
    if isinstance(val, list):
        if not val:
            lines.append(f"{indent}(пусто)")
            return lines
        for i, item in enumerate(val, start=1):
            if isinstance(item, dict):
                lines.append(f"{indent}{i}.")
                lines.extend(_format_nested(item, indent + "   "))
            else:
                sub = _format_nested(item, indent + "   ")
                if sub:
                    lines.append(f"{indent}• {sub[0].strip()}")
                    lines.extend(sub[1:])
                else:
                    lines.append(f"{indent}•")
        return lines
    if isinstance(val, dict):
        if not val:
            lines.append(f"{indent}(пусто)")
            return lines
        for k, v in val.items():
            lines.append(f"{indent}{k}:")
            lines.extend(_format_nested(v, indent + "  "))
        return lines
    lines.append(f"{indent}{val!s}")
    return lines


def _skip_key(key: str) -> bool:
    return key.endswith("_source") or key == "metadata"


def aggregated_to_plain_text(aggregated: dict[str, Any] | None) -> str:
    """
    Итоговая суммаризация: названия полей — заголовки разделов, ниже — текст.
    Служебные ключи *_source и metadata не включаются.
    """
    if not aggregated:
        return ""

    blocks: list[str] = []
    for key, value in aggregated.items():
        if _skip_key(key):
            continue
        title = str(key)
        if key == "источники" and isinstance(value, dict):
            blocks.append(title)
            blocks.append("─" * min(max(len(title), 3), 72))
            for num in _sort_source_keys(list(value.keys())):
                chunk = value[num]
                if not isinstance(chunk, str):
                    chunk = json.dumps(chunk, ensure_ascii=False)
                blocks.append(f"[{num}] {chunk}")
            blocks.append("")
            continue

        blocks.append(title)
        blocks.append("─" * min(max(len(title), 3), 72))
        body = "\n".join(_format_nested(value)).strip()
        blocks.append(body if body else "—")
        blocks.append("")

    return "\n".join(blocks).strip()


def aggregated_to_markdown(aggregated: dict[str, Any] | None) -> str:
    """Тот же контент с заголовками ## для отображения в Streamlit."""
    if not aggregated:
        return "_Нет агрегированных данных._"

    parts: list[str] = []
    for key, value in aggregated.items():
        if _skip_key(key):
            continue
        title = str(key)
        parts.append(f"## {title}")
        if key == "источники" and isinstance(value, dict):
            for num in _sort_source_keys(list(value.keys())):
                chunk = value[num]
                if not isinstance(chunk, str):
                    chunk = json.dumps(chunk, ensure_ascii=False)
                parts.append(f"**[{num}]** {chunk}")
            parts.append("")
            continue

        body_lines = _format_nested(value)
        body = "\n".join(body_lines).strip()
        parts.append(body if body else "—")
        parts.append("")

    return "\n".join(parts).strip()
