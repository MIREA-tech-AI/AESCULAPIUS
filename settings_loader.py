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
class SummarizationModels:
    generator_1: str
    generator_2: str
    aggregator: str
    validator: str


@dataclass(frozen=True)
class SummarizationTemperatures:
    generator_1: float
    generator_2: float
    aggregator: float
    validator: float


@dataclass(frozen=True)
class SummarizationPromptFiles:
    generator_1_system: str
    generator_2_system: str


@dataclass(frozen=True)
class SummarizationValidation:
    required_fields: tuple[str, ...]
    validation_threshold: float
    min_field_length: int


@dataclass(frozen=True)
class SummarizationGeneration:
    """Общие параметры вызова LLM в пайплайне (температуры — в SummarizationTemperatures)."""

    max_tokens: int
    top_k: int
    enable_thinking: bool


@dataclass(frozen=True)
class SummarizationSettings:
    """Настройки двух генераторов, агрегации и валидации (модели из конфига)."""

    models: SummarizationModels
    temperatures: SummarizationTemperatures
    generation: SummarizationGeneration
    prompt_files: SummarizationPromptFiles
    system_prompt_generator_1: str
    system_prompt_generator_2: str
    user_template_generator_1: str
    user_template_generator_2: str
    validation: SummarizationValidation


@dataclass(frozen=True)
class AppSettings:
    llm: LLMSettings
    generation: GenerationSettings
    prompts: PromptPaths
    system_prompt: str
    user_intro: str
    summarization: SummarizationSettings


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

    sum_raw = raw.get("summarization") or {}
    sum_models = sum_raw.get("models", {})
    sum_temps = sum_raw.get("temperatures", {})
    sum_gen = sum_raw.get("generation", {})
    sum_pf = sum_raw.get("prompt_files", {})
    sum_ut = sum_raw.get("user_templates", {})
    sum_val = sum_raw.get("validation", {})

    g1_rel = sum_pf.get(
        "generator_1_system", "prompts/summarization/generator_1_system.txt"
    )
    g2_rel = sum_pf.get(
        "generator_2_system", "prompts/summarization/generator_2_system.txt"
    )
    g1_path = ROOT_DIR / g1_rel
    g2_path = ROOT_DIR / g2_rel

    summarization = SummarizationSettings(
        models=SummarizationModels(
            generator_1=str(sum_models.get("generator_1", llm.model)).strip(),
            generator_2=str(sum_models.get("generator_2", llm.model)).strip(),
            aggregator=str(sum_models.get("aggregator", llm.model)).strip(),
            validator=str(sum_models.get("validator", llm.model)).strip(),
        ),
        temperatures=SummarizationTemperatures(
            generator_1=float(sum_temps.get("generator_1", 0.5)),
            generator_2=float(sum_temps.get("generator_2", 0.5)),
            aggregator=float(sum_temps.get("aggregator", 0.6)),
            validator=float(sum_temps.get("validator", 0.1)),
        ),
        generation=SummarizationGeneration(
            max_tokens=int(sum_gen.get("max_tokens", generation.max_tokens)),
            top_k=int(sum_gen.get("top_k", generation.top_k)),
            enable_thinking=bool(
                sum_gen.get("enable_thinking", generation.enable_thinking)
            ),
        ),
        prompt_files=SummarizationPromptFiles(
            generator_1_system=g1_rel,
            generator_2_system=g2_rel,
        ),
        system_prompt_generator_1=g1_path.read_text(encoding="utf-8").strip(),
        system_prompt_generator_2=g2_path.read_text(encoding="utf-8").strip(),
        user_template_generator_1=str(
            sum_ut.get("generator_1", "Суммаризируй медицинский текст:\n\n{text}")
        ),
        user_template_generator_2=str(
            sum_ut.get("generator_2", "Проанализируй медицинскую карту:\n\n{text}")
        ),
        validation=SummarizationValidation(
            required_fields=tuple(
                sum_val.get("required_fields")
                or (
                    "пациент",
                    "жалобы",
                    "анамнез_заболевания",
                    "анамнез_жизни",
                    "лабораторные_данные",
                    "инструментальные_данные",
                    "цель_исследования",
                    "источники",
                )
            ),
            validation_threshold=float(sum_val.get("validation_threshold", 0.7)),
            min_field_length=int(sum_val.get("min_field_length", 10)),
        ),
    )

    return AppSettings(
        llm=llm,
        generation=generation,
        prompts=prompts,
        system_prompt=system_prompt,
        user_intro=user_intro,
        summarization=summarization,
    )
