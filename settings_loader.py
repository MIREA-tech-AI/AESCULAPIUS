import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Корень проекта (рядом с app.py)
ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float
    max_tokens: int
    top_k: int
    enable_thinking: bool


@dataclass(frozen=True)
class PromptPaths:
    system_file: str
    user_intro_file: str


@dataclass(frozen=True)
class AppSettings:
    llm: LLMSettings
    generation: GenerationSettings
    prompts: PromptPaths
    system_prompt: str
    user_intro: str


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивное слияние override в base."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _resolve_api_key(raw_llm: dict[str, Any]) -> str:
    key = (raw_llm.get("api_key") or "").strip()
    env_name = raw_llm.get("api_key_env")
    if env_name:
        key = os.environ.get(env_name, key).strip()
    return key


def load_app_settings() -> AppSettings:
    """Загрузка config/settings.json и опционально config/settings.local.json."""
    cfg_path = ROOT_DIR / "config" / "settings.json"
    raw = _read_json(cfg_path)
    local_path = ROOT_DIR / "config" / "settings.local.json"
    if local_path.is_file():
        raw = _deep_merge(raw, _read_json(local_path))

    llm_raw = raw["llm"]
    gen_raw = raw["generation"]
    pr_raw = raw.get("prompts", {})

    system_rel = pr_raw.get("system_file", "prompts/system.txt")
    user_rel = pr_raw.get("user_intro_file", "prompts/user_intro.txt")

    system_path = ROOT_DIR / system_rel
    user_path = ROOT_DIR / user_rel

    system_prompt = system_path.read_text(encoding="utf-8").strip()
    user_intro = user_path.read_text(encoding="utf-8").strip()

    llm = LLMSettings(
        base_url=str(llm_raw["base_url"]).strip(),
        api_key=_resolve_api_key(llm_raw),
        model=str(llm_raw["model"]).strip(),
    )
    generation = GenerationSettings(
        temperature=float(gen_raw["temperature"]),
        max_tokens=int(gen_raw["max_tokens"]),
        top_k=int(gen_raw["top_k"]),
        enable_thinking=bool(gen_raw["enable_thinking"]),
    )
    prompts = PromptPaths(
        system_file=system_rel,
        user_intro_file=user_rel,
    )

    return AppSettings(
        llm=llm,
        generation=generation,
        prompts=prompts,
        system_prompt=system_prompt,
        user_intro=user_intro,
    )
