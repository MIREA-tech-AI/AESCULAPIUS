import streamlit as st

from llm_service import make_client, summarize_medical_text
from settings_loader import load_app_settings


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
        Вставьте текст медицинской карты слева или загрузите <strong>.txt</strong> файл. Справа отобразится
        структурированное резюме для врача-рентгенолога (КТ органов брюшной полости) —
        жалобы, анамнез, сопутствующие данные, лаборатория и инструментальные методы.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "summary" not in st.session_state:
        st.session_state.summary = ""
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
            help="Отправить текст в модель и показать резюме справа",
        )

    source = (st.session_state.source_text or "").strip()

    settings = get_app_settings()
    client = get_llm_client()

    if run:
        if not source:
            st.warning("Добавьте текст или загрузите .txt файл.")
        else:
            with work_status:
                with st.spinner("Модель формирует резюме…"):
                    try:
                        st.session_state.summary = summarize_medical_text(
                            client, settings, source
                        )
                    except Exception as e:
                        st.session_state.summary = ""
                        st.error(f"Ошибка при обращении к API: {e}")

    summary_text = st.session_state.summary or ""
    summary_bytes = summary_text.encode("utf-8-sig")

    with right:
        st.subheader("Суммаризация")
        st.download_button(
            label="Экспортировать (.txt)",
            data=summary_bytes,
            file_name="yulastar_summary.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=len(summary_text.strip()) == 0,
            key="download_summary_txt",
            help="Скачать текущее резюме в UTF-8 (с BOM для Блокнота Windows)",
        )
        if summary_text.strip():
            st.markdown(summary_text)
        else:
            st.caption("Результат появится здесь после нажатия «Суммаризировать».")


if __name__ == "__main__":
    main()
