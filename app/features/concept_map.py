"""Concept map (F9): 6–12 concepts from one answer, drawn with Graphviz (st.graphviz_chart).

Nodes without source support are dashed. Node labels double as pills that ask a follow-up.
"""

import streamlit as st

from . import export
from .llm_extras import ConceptMap, LLMExtrasError, llm_json
from .records import sources_context

PROMPT = (
    "Build a concept map of this answer: 6 to 12 key concepts (labels of at most 4 words) and the "
    "relationships between them (edge labels of at most 3 words, such as \"improves\", \"measured by\", "
    "\"part of\"). Use node ids n1, n2… and give each node `src`: the numbers of the sources that support it "
    "(an empty list if no source does). Every edge's `from` and `to` must be node ids."
)
INK, INK2, SURFACE, PAPER = "#1f1d1a", "#5b574f", "#ffffff", "#f7f6f3"


def generate(record):
    data = llm_json(PROMPT, ConceptMap, sources_context(record), feature="concept_map", answer_id=record["id"])
    valid = {source["n"] for source in record.get("sources", [])}
    ids = set()
    nodes = []
    for node in data["nodes"]:
        if node["id"] in ids:
            continue
        ids.add(node["id"])
        nodes.append(dict(node, src=[n for n in node.get("src") or [] if n in valid]))
    edges = [edge for edge in data.get("edges", []) if edge["from"] in ids and edge["to"] in ids and edge["from"] != edge["to"]]
    return {"nodes": nodes[:12], "edges": edges}


def _quote(text):
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def to_dot(concept_map):
    lines = [
        "digraph concepts {",
        f'  graph [rankdir=LR, bgcolor="transparent", pad=0.2, nodesep=0.35, ranksep=0.55];',
        f'  node [shape=box, style="rounded,filled", fillcolor="{SURFACE}", color="{INK}", fontcolor="{INK}", '
        'fontname="IBM Plex Sans, Helvetica, Arial", fontsize=12, margin="0.18,0.08"];',
        f'  edge [color="{INK2}", fontcolor="{INK2}", fontname="IBM Plex Sans, Helvetica, Arial", fontsize=10, arrowsize=0.7];',
    ]
    for node in concept_map["nodes"]:
        style = "" if node.get("src") else f', style="rounded,dashed,filled", color="{INK2}", fontcolor="{INK2}"'
        lines.append(f"  {_quote(node['id'])} [label={_quote(node['label'])}{style}];")
    for edge in concept_map.get("edges", []):
        label = f" [label={_quote(' ' + edge['label'] + ' ')}]" if edge.get("label") else ""
        lines.append(f"  {_quote(edge['from'])} -> {_quote(edge['to'])}{label};")
    lines.append("}")
    return "\n".join(lines)


def render_panel(record, mid, ask):
    try:
        with st.spinner("Mapping the concepts…"):
            concept_map = generate(record)
    except LLMExtrasError as error:
        st.error(str(error))
        return
    st.html("<div class='panel-h'><b>Concept map</b></div>")
    dot = to_dot(concept_map)
    st.graphviz_chart(dot, width="stretch")
    if any(not node.get("src") for node in concept_map["nodes"]):
        st.caption("Dashed boxes are concepts no single source states directly.")
    labels = [node["label"] for node in concept_map["nodes"]]

    def explore():
        label = st.session_state.get(f"map_{mid}_explore")
        if label:
            ask(f"Explain {label} in the context of: {record['question']}")
        st.session_state[f"map_{mid}_explore"] = None

    st.pills("Explore a concept as a new question", labels, key=f"map_{mid}_explore", on_change=explore)
    svg = export.svg(dot)
    with st.container(horizontal=True, key=f"map_{mid}_tools"):
        if svg:
            st.download_button("Download map (.svg)", svg, file_name=export.slug(record["question"], 50) + "-map.svg",
                               mime="image/svg+xml", key=f"map_{mid}_svg")
    with st.expander("Mermaid code for your notes"):
        st.code(export.mermaid(concept_map), language="text")
