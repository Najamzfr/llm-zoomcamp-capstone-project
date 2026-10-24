"""Browser localStorage for history, review deck, collections and review stats.

One `st.components.v2` component (it runs in the page, not an iframe) reads every key once per
session and writes changed keys. Python keeps the parsed values in session_state and is the
source of truth after the first load. Everything degrades to in-memory storage when
localStorage is blocked (private mode, AppTest, old browsers): `get()` still works, data just
doesn't survive a refresh.

Keys are versioned: bump the suffix when a stored shape changes.
"""

import json
import uuid

import streamlit as st

HISTORY = "ps_history_v1"
DECK = "ps_deck_v1"
COLLECTIONS = "ps_collections_v1"
REVIEW_META = "ps_review_meta_v1"  # streak and per-day review counts
DEFAULTS = {HISTORY: list, DECK: dict, COLLECTIONS: dict, REVIEW_META: dict}

_JS = """
export default function (component) {
  const { data, setStateValue } = component;
  if (!data) return;
  let store = null;
  try { store = window.localStorage; store.getItem('ps_probe'); } catch (e) { store = null; }
  if (data.writes && data.nonce && window.__psStoreNonce !== data.nonce) {
    window.__psStoreNonce = data.nonce;
    if (store) {
      for (const [key, value] of Object.entries(data.writes)) {
        try { value === null ? store.removeItem(key) : store.setItem(key, value); }
        catch (e) { console.warn('Paper Scout: could not save', key, e); }
      }
    }
  }
  if (window.__psStoreSession !== data.session) {
    window.__psStoreSession = data.session;
    const snapshot = { available: !!store, values: {}, tz: -new Date().getTimezoneOffset() };
    if (store) for (const key of data.keys) { try { snapshot.values[key] = store.getItem(key); } catch (e) {} }
    setStateValue('snapshot', snapshot);
  }
}
"""
_component = None


def _mount_component():
    global _component
    if _component is None:
        _component = st.components.v2.component("paper_scout_store", js=_JS)
    return _component


def _state():
    state = st.session_state
    if "ps_local" not in state:
        state["ps_local"] = {key: factory() for key, factory in DEFAULTS.items()}
        state["ps_local_loaded"] = False
        state["ps_local_available"] = False
        state["ps_local_pending"] = {}
        state["ps_local_nonce"] = 0
        state["ps_local_session"] = uuid.uuid4().hex
    return state


def get(key):
    return _state()["ps_local"][key]


def put(key, value):
    """Replace a stored value; it is written to the browser on the next sync()."""
    state = _state()
    state["ps_local"][key] = value
    state["ps_local_pending"][key] = value
    state["ps_local_nonce"] += 1


def loaded():
    return _state()["ps_local_loaded"]


def available():
    return _state()["ps_local_available"]


def tz_minutes():
    """The browser's UTC offset in minutes (0 until the first load or outside a browser)."""
    return int(st.session_state.get("ps_tz_minutes", 0))


def local_date(iso_timestamp):
    """ISO date (YYYY-MM-DD) of a UTC timestamp in the visitor's time zone."""
    from datetime import datetime, timedelta

    try:
        moment = datetime.fromisoformat(str(iso_timestamp))
    except ValueError:
        return str(iso_timestamp or "")[:10]
    return (moment + timedelta(minutes=tz_minutes())).date().isoformat()


def pending():
    return bool(_state()["ps_local_pending"])


def _merge(key, stored, current):
    """Values changed in this session before the browser copy arrived win; lists are unioned."""
    if key == HISTORY:
        ids = {item.get("id") for item in current}
        return [item for item in stored if item.get("id") not in ids] + current
    if isinstance(stored, dict) and isinstance(current, dict):
        return {**stored, **current}
    return current


def sync():
    """Mount the storage component once per run (call it early, inside the sidebar)."""
    state = _state()
    writes = {}
    if state["ps_local_pending"]:
        writes = {key: json.dumps(value, ensure_ascii=False) for key, value in state["ps_local_pending"].items()}
        state["ps_local_pending"] = {}
    try:
        result = _mount_component()(
            key="ps_store",
            data={"keys": list(DEFAULTS), "session": state["ps_local_session"], "nonce": state["ps_local_nonce"],
                  "writes": writes},
            default={"snapshot": None},
            on_snapshot_change=lambda: None,
            height=0,
        )
        snapshot = result.get("snapshot") if hasattr(result, "get") else getattr(result, "snapshot", None)
    except Exception:  # components unavailable (e.g. tests): stay in memory
        snapshot = None
    if snapshot and not state["ps_local_loaded"]:
        state["ps_local_loaded"] = True
        state["ps_local_available"] = bool(snapshot.get("available"))
        state["ps_tz_minutes"] = int(snapshot.get("tz") or 0)
        for key, factory in DEFAULTS.items():
            raw = (snapshot.get("values") or {}).get(key)
            try:
                stored = json.loads(raw) if raw else factory()
            except (TypeError, ValueError):
                stored = factory()
            if not isinstance(stored, type(factory())):
                stored = factory()
            current = state["ps_local"][key]
            state["ps_local"][key] = _merge(key, stored, current) if current else stored
            if current:  # write the merged value back
                state["ps_local_pending"][key] = state["ps_local"][key]
                state["ps_local_nonce"] += 1
        return True  # caller reruns so the loaded values render
    return False
