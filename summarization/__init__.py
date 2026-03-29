"""Пайплайн суммаризации ЭМК (два генератора, агрегация, валидация)."""

from summarization.human_text import aggregated_to_markdown, aggregated_to_plain_text
from summarization.pipeline import create_voting_graph, run_summarization_pipeline
from summarization.tables import pick_sources_for_table, sources_dict_to_rows

__all__ = [
    "aggregated_to_markdown",
    "aggregated_to_plain_text",
    "create_voting_graph",
    "run_summarization_pipeline",
    "pick_sources_for_table",
    "sources_dict_to_rows",
]
