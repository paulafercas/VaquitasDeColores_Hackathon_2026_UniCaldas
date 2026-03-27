from __future__ import annotations

import streamlit as st

from App_Streamlit.components import render_chat_message, render_sidebar
from LLM_Workflow.response_service import ResponseService


st.set_page_config(page_title="Copiloto Analitico", page_icon=":bar_chart:", layout="wide")


@st.cache_resource(show_spinner=False)
def get_response_service() -> ResponseService:
    return ResponseService()


def _dataset_info(service: ResponseService) -> dict[str, str | int]:
    sessions = service.query_service.bundle.canonical_sessions
    return {
        "min_date": str(sessions["timestamp"].dt.date.min()),
        "max_date": str(sessions["timestamp"].dt.date.max()),
        "total_sessions": int(sessions["session_id"].nunique()),
    }


def main() -> None:
    service = get_response_service()
    llm_runtime = st.session_state.get("last_llm_runtime")
    render_sidebar(
        dataset_info=_dataset_info(service),
        api_configured=service.llm_client.is_configured,
        llm_runtime=llm_runtime,
    )

    st.title("Copiloto Conversacional de Analitica Web")
    st.write(
        "Haz preguntas sobre navegacion, paginas mas vistas, salidas, abandono o segmentos. "
        "El motor calcula la metrica y Gemini redacta la respuesta final."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Estoy listo para ayudarte a analizar el comportamiento web. "
                    "Puedes preguntarme, por ejemplo, cual fue la pagina mas vista hoy o cual fue el flujo principal."
                ),
                "support_table": None,
            }
        ]

    for message in st.session_state.messages:
        render_chat_message(message)

    question = st.chat_input("Escribe tu pregunta analitica")
    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_chat_message(user_message)

        with st.spinner("Consultando el motor analitico..."):
            result = service.generate_answer(question)

        assistant_message = {
            "role": "assistant",
            "content": result["final_answer"],
            "support_table": result["analytics_result"].get("support_table"),
            "analytics_result": result["analytics_result"],
            "route": result["route"],
            "llm_status": result["llm_status"],
            "llm_error_type": result.get("llm_error_type"),
            "llm_error_message": result.get("llm_error_message"),
            "used_fallback": result.get("used_fallback", False),
        }
        st.session_state.last_llm_runtime = {
            "llm_status": result["llm_status"],
            "model": result["model"],
            "llm_error_type": result.get("llm_error_type"),
            "llm_error_message": result.get("llm_error_message"),
            "used_fallback": result.get("used_fallback", False),
        }
        st.session_state.messages.append(assistant_message)
        render_chat_message(assistant_message)


if __name__ == "__main__":
    main()
