"""
Пайплайн: два генератора → агрегатор → валидатор (LangGraph).
Модели и промпты задаются через AppSettings / config.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

import openai
from langgraph.graph import END, START, StateGraph

from settings_loader import AppSettings, SummarizationValidation
from summarization.json_utils import parse_json_from_response
from summarization.state import GraphState


def _norm_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return len(val.strip()) == 0
    if isinstance(val, (list, dict)):
        return len(val) == 0
    return False


def _serialize_len(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    return json.dumps(val, ensure_ascii=False, sort_keys=True)


def _merge_sources(a: dict[Any, Any], b: dict[Any, Any]) -> dict[Any, Any]:
    """Объединяет два словаря цитат; при конфликте — более длинная строка."""
    out: dict[Any, Any] = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va, vb = a.get(k), b.get(k)
        e1, e2 = _norm_empty(va), _norm_empty(vb)
        if e1 and e2:
            continue
        if e1:
            out[k] = vb
        elif e2:
            out[k] = va
        else:
            sa = va if isinstance(va, str) else json.dumps(va, ensure_ascii=False)
            sb = vb if isinstance(vb, str) else json.dumps(vb, ensure_ascii=False)
            out[k] = va if len(sa) >= len(sb) else vb
    return out


def _field_nonempty(value: Any, min_len: int) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > min_len
    if isinstance(value, (dict, list)):
        return len(value) > 0
    return True


def build_aggregate_node(
    val_cfg: SummarizationValidation,
) -> Callable[[GraphState], dict[str, Any]]:
    def aggregate_results(state: GraphState) -> dict[str, Any]:
        aggregated: dict[str, Any] = {}
        gen1 = state.get("generator_1_output") or {}
        gen2 = state.get("generator_2_output") or {}

        for field_name in val_cfg.required_fields:
            val1, val2 = gen1.get(field_name), gen2.get(field_name)

            if field_name == "источники":
                if isinstance(val1, dict) and isinstance(val2, dict):
                    aggregated[field_name] = _merge_sources(val1, val2)
                    aggregated[f"{field_name}_source"] = "merged"
                    continue
                if isinstance(val1, dict) and _norm_empty(val2):
                    aggregated[field_name] = val1
                    aggregated[f"{field_name}_source"] = "generator_1"
                    continue
                if isinstance(val2, dict) and _norm_empty(val1):
                    aggregated[field_name] = val2
                    aggregated[f"{field_name}_source"] = "generator_2"
                    continue

            e1, e2 = _norm_empty(val1), _norm_empty(val2)
            if e1 and e2:
                aggregated[field_name] = ""
                aggregated[f"{field_name}_source"] = "none"
            elif e1:
                aggregated[field_name] = val2
                aggregated[f"{field_name}_source"] = "generator_2"
            elif e2:
                aggregated[field_name] = val1
                aggregated[f"{field_name}_source"] = "generator_1"
            else:
                l1, l2 = len(_serialize_len(val1)), len(_serialize_len(val2))
                if l1 >= l2:
                    aggregated[field_name] = val1
                    aggregated[f"{field_name}_source"] = "generator_1"
                else:
                    aggregated[field_name] = val2
                    aggregated[f"{field_name}_source"] = "generator_2"

        aggregated["metadata"] = {
            "aggregation_method": "voting_by_completeness",
            "timestamp": datetime.now().isoformat(),
            "generators_used": ["generator_1", "generator_2"],
        }
        return {
            "aggregated_result": aggregated,
            "timestamp_aggregator": datetime.now().isoformat(),
        }

    return aggregate_results


def build_validate_node(
    val_cfg: SummarizationValidation,
) -> Callable[[GraphState], dict[str, Any]]:
    def validate_result(state: GraphState) -> dict[str, Any]:
        validation: dict[str, Any] = {
            "checks": {},
            "score": 0.0,
            "is_valid": False,
            "issues": [],
        }
        aggregated = state.get("aggregated_result") or {}
        source_text = (state.get("source_emr") or "").lower()
        min_len = val_cfg.min_field_length
        threshold = val_cfg.validation_threshold

        required_present = 0
        for field_name in val_cfg.required_fields:
            value = aggregated.get(field_name)
            is_present = _field_nonempty(value, min_len)
            validation["checks"][f"field_{field_name}"] = is_present
            if is_present:
                required_present += 1
            else:
                validation["issues"].append(
                    f"Поле '{field_name}' отсутствует или пустое"
                )

        n_req = len(val_cfg.required_fields) or 1
        validation["checks"]["required_fields_coverage"] = required_present / n_req

        factual_accuracy = 0.0
        complaints = aggregated.get("жалобы")
        if complaints:
            if isinstance(complaints, str):
                text_c = complaints.lower()
            else:
                text_c = json.dumps(complaints, ensure_ascii=False).lower()
            words = [w for w in text_c.split() if len(w) > 4]
            total_words = len(words)
            keywords_found = sum(1 for word in words if word in source_text)
            factual_accuracy = (
                keywords_found / total_words if total_words > 0 else 0.0
            )

        validation["checks"]["factual_accuracy"] = factual_accuracy
        if factual_accuracy < 0.5 and complaints:
            validation["issues"].append("Низкая фактологическая точность")

        critical_info_missing = False
        if "кт" in source_text or "компьютерная томография" in source_text:
            inst = aggregated.get("инструментальные_данные")
            if _norm_empty(inst):
                critical_info_missing = True
                validation["issues"].append(
                    "Отсутствуют данные инструментальных исследований"
                )

        validation["checks"]["critical_info_present"] = not critical_info_missing
        validation["score"] = (
            validation["checks"]["required_fields_coverage"] * 0.4
            + factual_accuracy * 0.4
            + (0.2 if not critical_info_missing else 0.0)
        )
        validation["is_valid"] = (
            validation["score"] >= threshold and not critical_info_missing
        )
        validation["timestamp"] = datetime.now().isoformat()

        return {
            "validation_result": validation,
            "is_valid": bool(validation["is_valid"]),
            "timestamp_validator": datetime.now().isoformat(),
        }

    return validate_result


def build_generator_node(
    client: openai.OpenAI,
    settings: AppSettings,
    which: int,
) -> Callable[[GraphState], dict[str, Any]]:
    s = settings.summarization
    gen = s.generation

    def run(state: GraphState) -> dict[str, Any]:
        try:
            if which == 1:
                model = s.models.generator_1
                temp = s.temperatures.generator_1
                system = s.system_prompt_generator_1
                user_tmpl = s.user_template_generator_1
                out_key = "generator_1_output"
                ts_key = "timestamp_gen1"
            else:
                model = s.models.generator_2
                temp = s.temperatures.generator_2
                system = s.system_prompt_generator_2
                user_tmpl = s.user_template_generator_2
                out_key = "generator_2_output"
                ts_key = "timestamp_gen2"

            user_content = user_tmpl.replace("{text}", state["source_emr"])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
            extra_body: dict[str, Any] = {
                "top_k": gen.top_k,
                "chat_template_kwargs": {"enable_thinking": gen.enable_thinking},
            }
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=gen.max_tokens,
                extra_body=extra_body,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_json_from_response(content)
            return {
                out_key: parsed,
                ts_key: datetime.now().isoformat(),
            }
        except Exception as e:
            err = f"Generator {which} error: {e}"
            out_key = "generator_1_output" if which == 1 else "generator_2_output"
            return {
                out_key: {"error": str(e)},
                "errors": [err],
            }

    return run


def create_voting_graph(client: openai.OpenAI, settings: AppSettings):
    """Собирает и компилирует граф."""
    val_cfg = settings.summarization.validation
    workflow = StateGraph(GraphState)
    workflow.add_node(
        "generator_1", build_generator_node(client, settings, 1)
    )
    workflow.add_node(
        "generator_2", build_generator_node(client, settings, 2)
    )
    workflow.add_node("aggregator", build_aggregate_node(val_cfg))
    workflow.add_node("validator", build_validate_node(val_cfg))
    workflow.add_edge(START, "generator_1")
    workflow.add_edge(START, "generator_2")
    workflow.add_edge("generator_1", "aggregator")
    workflow.add_edge("generator_2", "aggregator")
    workflow.add_edge("aggregator", "validator")
    workflow.add_edge("validator", END)
    return workflow.compile()


def run_summarization_pipeline(
    client: openai.OpenAI,
    settings: AppSettings,
    source_emr_text: str,
) -> dict[str, Any]:
    """Запуск полного пайплайна по тексту ЭМК."""
    app = create_voting_graph(client, settings)
    initial_state: GraphState = {
        "source_emr": source_emr_text,
        "generator_1_output": {},
        "generator_2_output": {},
        "aggregated_result": {},
        "validation_result": {},
        "is_valid": False,
        "errors": [],
        "timestamp_gen1": "",
        "timestamp_gen2": "",
        "timestamp_aggregator": "",
        "timestamp_validator": "",
    }
    return app.invoke(initial_state)
