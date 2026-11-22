"""Chat UI for the multimodal RAG pipeline.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

from mmrag.config import PROJECT_ROOT, get_settings
from mmrag.generation.rag import ask_rag
from mmrag.indexing.vectorstore import get_embeddings, get_index
from mmrag.schemas import RAGResponse

EXAMPLE_QUESTIONS = [
    "Which campus region has the highest PUE?",
    "What was the Atlas-M incident rate in October 2025?",
    "Which process step receives units that fail QA?",
    "How much did Frontier AI Labs spend on R&D in FY2025?",
]

MODALITY_ICONS = {"text": ":material/notes:", "table": ":material/table_chart:", "visual": ":material/bar_chart:"}

st.set_page_config(page_title="Multimodal RAG", page_icon=":material/search:", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and connecting to the index...")
def warm_up() -> None:
    """Load the embedding model and open the index once, before the first question."""
    get_embeddings()
    get_index()


def render_details(response: RAGResponse) -> None:
    """Show how an answer was produced: route, model, timings, tokens and sources."""
    route_color = "violet" if response.route == "visual" else "blue"
    badges = [
        f":{route_color}-badge[:material/alt_route: {response.route} route]",
        f":gray-badge[{response.model}]",
        f":gray-badge[:material/timer: {response.timings['total']}s]",
        f":gray-badge[{response.usage.get('total_tokens', '?')} tokens]",
    ]
    st.markdown(" ".join(badges))

    with st.expander(f"Sources ({len(response.sources)})", icon=":material/description:"):
        st.caption(
            f"Retrieval {response.timings['retrieval']}s · Generation {response.timings['generation']}s"
        )
        for s in response.sources:
            with st.container(border=True):
                st.markdown(
                    f"**[{s.number}]** {MODALITY_ICONS.get(s.modality, '')} {s.modality} · "
                    f"{s.source} · page {s.page} · score `{s.score:.3f}`"
                )
                if s.image_path:
                    st.image(str(PROJECT_ROOT / s.image_path), caption=f"Image sent to the VLM for [{s.number}]")
                st.text(s.content)


def answer(question: str, top_k: int) -> None:
    """Run the pipeline for one question and add both messages to the history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and generating..."):
                response = ask_rag(question, k=top_k)
        except Exception as exc:  # show API/network errors in the chat instead of crashing the app
            st.error(f"Something went wrong: {exc}")
            return
        st.markdown(response.answer)
        render_details(response)

    st.session_state.messages.append({"role": "assistant", "content": response.answer, "response": response})


# ---- Sidebar ----
settings = get_settings()
with st.sidebar:
    st.header("Multimodal RAG")
    st.caption("Ask questions about PDFs containing text, tables, charts and diagrams.")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=settings.top_k)
    st.caption(f"Text model: `{settings.text_model}`  \nVision model: `{settings.vision_model}`")
    if st.button("Clear chat", icon=":material/delete:"):
        st.session_state.messages = []

# ---- Chat ----
warm_up()
st.session_state.setdefault("messages", [])

st.title("Ask your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "response" in message:
            render_details(message["response"])

clicked_example = None
if not st.session_state.messages:
    st.caption("Try an example:")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example, use_container_width=True):
            clicked_example = example

question = st.chat_input("Ask a question about the documents") or clicked_example
if question:
    answer(question, top_k)
