"""Collections (F14): named groups of saved answers in localStorage ("ps_collections_v1").

{name: [answer snapshots]}; included in the review deck's JSON backup.
"""

import streamlit as st

from app.storage import local
from research_assistant.presentation import format_date, shorten

from . import export
from .records import now_iso, without_upload_text

MAX_NAME = 40


def all_collections():
    return local.get(local.COLLECTIONS)


def snapshot(record):
    """What a collection keeps of an answer (uploaded excerpts are dropped: uploads are never persisted)."""
    sources = without_upload_text(record.get("sources", []))
    return {"id": record["id"], "question": record["question"], "answer_md": record["answer_md"],
            "followups": record.get("followups", []), "sources": sources, "created_at": record.get("created_at"),
            "saved_at": now_iso(), "kind": record.get("kind", "answer"), "verified": record.get("verified")}


def clean_name(name):
    return " ".join((name or "").split())[:MAX_NAME]


def save(name, record):
    """Add an answer to a collection (created if new); returns False if it was already there."""
    name = clean_name(name)
    if not name:
        raise ValueError("Give the collection a name.")
    current = dict(all_collections())
    items = list(current.get(name, []))
    if any(item["id"] == record["id"] for item in items):
        return False
    current[name] = items + [snapshot(record)]
    local.put(local.COLLECTIONS, current)
    return True


def remove(name, record_id):
    current = dict(all_collections())
    current[name] = [item for item in current.get(name, []) if item["id"] != record_id]
    local.put(local.COLLECTIONS, current)


def rename(old, new):
    new = clean_name(new)
    current = dict(all_collections())
    if not new or new == old or old not in current:
        return old
    if new in current:
        raise ValueError("A collection with that name already exists.")
    current = {(new if key == old else key): value for key, value in current.items()}
    local.put(local.COLLECTIONS, current)
    return new


def delete(name):
    current = dict(all_collections())
    current.pop(name, None)
    local.put(local.COLLECTIONS, current)


def collection_markdown(name, items):
    parts = [f"# {name}", ""]
    for item in items:
        parts.append(export.answer_markdown(item).replace("# ", "## ", 1).replace("\n## References", "\n### References"))
    return "\n".join(parts).strip() + "\n"


def collection_bibtex(items):
    return export.bibtex([source for item in items for source in item.get("sources", [])])


# ------------------------------------------------------------------ UI
def render_panel(record, mid):
    state = st.session_state
    st.html("<div class='panel-h'><b>Save to a collection</b></div>")
    names = list(all_collections())
    choice = st.pills("Your collections", names, key=f"save_{mid}_pick") if names else None
    new_name = st.text_input("New collection name", max_chars=MAX_NAME, key=f"save_{mid}_new",
                             placeholder="e.g. Thesis reading")
    target = clean_name(new_name) or choice
    if st.button("Save answer", key=f"save_{mid}_go", type="primary", disabled=not target):
        if save(target, record):
            st.toast(f"Saved to “{target}”")
        else:
            st.toast(f"Already in “{target}”")
        state[f"save_{mid}_saved"] = target
    if not names and not new_name:
        st.caption("Collections live in this browser. Name your first one above.")


def render_view(name, open_record):
    """The page for one collection (state['view'] == 'collection')."""
    state = st.session_state
    items = all_collections().get(name)
    if items is None:
        st.info("This collection no longer exists.")
        return
    st.markdown(f"# {name}")
    st.caption(f"{len(items)} saved answer{'s' if len(items) != 1 else ''}")
    for item in reversed(items):
        with st.container(border=True, key=f"coll_item_{item['id']}"):
            st.markdown(f"**{item['question']}**")
            stamp = item.get("saved_at") or item.get("created_at")
            saved = format_date(local.local_date(stamp)) if stamp else ""
            st.caption(f"Saved {saved}" if saved else "")
            with st.container(horizontal=True):
                if st.button("Open", key=f"coll_open_{item['id']}"):
                    open_record(item)
                    state["view"] = None
                    st.rerun()
                if st.button("Remove", key=f"coll_remove_{item['id']}", type="tertiary"):
                    remove(name, item["id"])
                    st.rerun()
    st.divider()
    with st.container(horizontal=True, key="coll_tools"):
        base = export.slug(name, 50)
        st.download_button("Download collection (.md)", collection_markdown(name, items), file_name=f"{base}.md",
                           mime="text/markdown", key="coll_md", disabled=not items)
        st.download_button("Download BibTeX", collection_bibtex(items) or "% No citable sources\n", file_name=f"{base}.bib",
                           mime="application/x-bibtex", key="coll_bib", disabled=not items)
    with st.expander("Rename or delete"):
        new = st.text_input("New name", value=name, max_chars=MAX_NAME, key=f"coll_rename_{name}")
        if st.button("Rename", key="coll_rename_go"):
            try:
                state["collection"] = rename(name, new)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        if state.get("coll_confirm_delete") == name:
            st.warning(f"Delete “{name}” and its {len(items)} saved answers from this browser?")
            with st.container(horizontal=True):
                if st.button("Delete collection", key="coll_delete_yes", type="primary"):
                    delete(name)
                    state["view"], state["coll_confirm_delete"] = None, None
                    st.rerun()
                if st.button("Cancel", key="coll_delete_no"):
                    state["coll_confirm_delete"] = None
                    st.rerun()
        elif st.button("Delete collection…", key="coll_delete"):
            state["coll_confirm_delete"] = name
            st.rerun()


def sidebar_list():
    """Collection buttons for the sidebar; returns the clicked name or None."""
    clicked = None
    collections = all_collections()
    if collections:
        st.html("<div class='ps-side-label'>Collections</div>")
        for index, (name, items) in enumerate(collections.items()):
            if st.button(f"{shorten(name, 40)} ({len(items)})", key=f"coll_nav_{index}", type="tertiary", width="stretch"):
                clicked = name
    return clicked
