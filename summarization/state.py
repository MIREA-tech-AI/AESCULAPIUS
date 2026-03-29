"""Состояние графа LangGraph для пайплайна суммаризации."""

from operator import add
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict):
    """
    Состояние графа: два параллельных генератора, агрегатор, валидатор.
    errors — с редюсером add для параллельных нод.
    """

    source_emr: str
    generator_1_output: dict[str, Any]
    generator_2_output: dict[str, Any]
    aggregated_result: dict[str, Any]
    validation_result: dict[str, Any]
    is_valid: bool
    errors: Annotated[list[str], add]
    timestamp_gen1: str
    timestamp_gen2: str
    timestamp_aggregator: str
    timestamp_validator: str
