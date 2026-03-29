import pandas as pd
import streamlit as st

from llm_service import make_client
from settings_loader import load_app_settings
from summarization import (
    aggregated_to_markdown,
    aggregated_to_plain_text,
    pick_sources_for_table,
    run_summarization_pipeline,
    sources_dict_to_rows,
)


@st.cache_resource
def get_app_settings():
    """Кэш настроек и промптов на время сессии сервера."""
    return load_app_settings()


@st.cache_resource
def get_llm_client():
    return make_client(get_app_settings())


def main():
    st.set_page_config(
        page_title="Суммаризатор медицинских карт YulaStar",
        page_icon="🏥",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        h1.yulastar-title { text-align: center; margin-bottom: 0.35rem; }
        div.yulastar-desc { text-align: center; color: rgba(49, 51, 63, 0.85); margin-bottom: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1 class="yulastar-title">Суммаризатор медицинских карт YulaStar</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="yulastar-desc">
        Вставьте текст медицинской карты слева или загрузите <strong>.txt</strong> файл. Справа — итоговая
        суммаризация в текстовом виде (разделы по полям JSON), таблица цитат и вспомогательные JSON-вкладки.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "source_text" not in st.session_state:
        st.session_state.source_text = ""
    if "_upload_sig" not in st.session_state:
        st.session_state._upload_sig = None

    work_status = st.empty()

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Исходный текст")
        uploaded = st.file_uploader(
            "Загрузить .txt",
            type=["txt"],
            help="Содержимое сразу появится в поле ниже",
        )
        if uploaded is not None:
            file_id = getattr(uploaded, "file_id", None)
            sig = (file_id, uploaded.name, uploaded.size)
            if sig != st.session_state._upload_sig:
                st.session_state.source_text = uploaded.getvalue().decode(
                    "utf-8", errors="replace"
                )
                st.session_state._upload_sig = sig
        st.text_area(
            "Текст карты",
            height=420,
            placeholder="Текст медицинской карты...",
            key="source_text",
            label_visibility="collapsed",
        )
        run = st.button(
            "Суммаризировать",
            type="primary",
            use_container_width=True,
            help="Запустить пайплайн (2 модели → агрегация → валидация)",
        )

    source = (st.session_state.source_text or "").strip()

    settings = get_app_settings()
    client = get_llm_client()

    if run:
        if not source:
            st.warning("Добавьте текст или загрузите .txt файл.")
        else:
            with work_status:
                with st.spinner("Пайплайн суммаризации (генераторы и валидация)…"):
                    try:
                        st.session_state.pipeline_result = run_summarization_pipeline(
                            client, settings, source
                        )
                    except Exception as e:
                        st.session_state.pipeline_result = None
                        st.error(f"Ошибка при обращении к API: {e}")

    result = st.session_state.pipeline_result

    with right:
        st.subheader("Результат")
        if result is None:
            st.caption("Нажмите «Суммаризировать», чтобы запустить пайплайн.")
        else:
            errs = result.get("errors") or []
            if errs:
                st.error("Ошибки: " + "; ".join(errs))

            agg = result.get("aggregated_result") or {}
            summary_plain = aggregated_to_plain_text(agg if isinstance(agg, dict) else {})
            summary_bytes = summary_plain.encode("utf-8-sig")
            st.download_button(
                label="Скачать суммаризацию (.txt)",
                data=summary_bytes,
                file_name="yulastar_summary.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key="download_summary_txt",
                disabled=len(summary_plain.strip()) == 0,
            )

            sources = pick_sources_for_table(result)
            rows = sources_dict_to_rows(sources)

            tab_summary, tab_citations, tab_agg, tab_g1, tab_g2, tab_val = st.tabs(
                [
                    "Итоговая суммаризация",
                    "Таблица цитат",
                    "Агрегат (JSON)",
                    "Генератор 1",
                    "Генератор 2",
                    "Валидация",
                ]
            )

            with tab_summary:
                if summary_plain.strip():
                    st.markdown(aggregated_to_markdown(agg if isinstance(agg, dict) else {}))
                else:
                    st.info(
                        "Агрегированный результат пуст. Проверьте ответы моделей на вкладках генераторов."
                    )

            with tab_citations:
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info(
                        "Блок «источники» пуст или отсутствует в ответе моделей. "
                        "Смотрите сырые JSON на других вкладках."
                    )

            with tab_agg:
                st.json(result.get("aggregated_result") or {})

            with tab_g1:
                st.json(result.get("generator_1_output") or {})

            with tab_g2:
                st.json(result.get("generator_2_output") or {})

            with tab_val:
                vr = result.get("validation_result") or {}
                st.json(vr)
                valid = result.get("is_valid")
                if valid is not None:
                    st.caption(f"Итог валидации: {'пройдено' if valid else 'не пройдено'}")


if __name__ == "__main__":
    main()
