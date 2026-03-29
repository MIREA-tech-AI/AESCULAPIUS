"""Утилиты разбора JSON из ответа модели."""

import json
from typing import Any


def parse_json_from_response(content: str) -> dict[str, Any]:
    """Выделяет первый JSON-объект из текста и парсит его."""
    try:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        return {"raw_output": content}
    except json.JSONDecodeError:
        return {"raw_output": content, "parse_error": True}
