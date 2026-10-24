"""Read aloud (F13) with the browser's Web Speech API: zero cost, nothing leaves the browser.

A `st.components.v2` component (runs in the page) with Play / Pause / Stop, speed 0.8–1.5× and a voice
picker. Text is read in sentence-sized utterances (long utterances stall in Chrome). The short answer is
read first, then the body; citations and Markdown are stripped beforehand.
"""

import re

import streamlit as st

from research_assistant.presentation import split_short_answer, strip_for_speech

HTML = """
<div class="tts">
  <div class="row">
    <button type="button" data-act="play" class="primary">▶ Play</button>
    <button type="button" data-act="pause">Pause</button>
    <button type="button" data-act="stop">Stop</button>
  </div>
  <div class="row">
    <label>Speed <input type="range" min="0.8" max="1.5" step="0.1" value="1" data-speed> <span data-speed-label>1.0×</span></label>
    <label>Voice <select data-voice></select></label>
  </div>
  <p class="note" data-status>Reads the short answer first, then the rest.</p>
</div>
"""
CSS = """
.tts { font-family: inherit; color: var(--st-text-color, #1f1d1a); }
.tts .row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 10px; }
.tts button { height: 40px; padding: 0 14px; border-radius: 8px; border: 1px solid #e2dfd8; background: #fff;
              font: inherit; font-size: 14px; color: #1f1d1a; cursor: pointer; }
.tts button.primary { background: #1f1d1a; border-color: #1f1d1a; color: #fff; }
.tts button:focus-visible, .tts select:focus-visible, .tts input:focus-visible { outline: 2px solid #1d4ed8; outline-offset: 2px; }
.tts label { font-size: 14px; display: inline-flex; gap: 8px; align-items: center; }
.tts select { max-width: 230px; height: 34px; border-radius: 6px; border: 1px solid #e2dfd8; font: inherit; font-size: 14px; }
.tts .note { color: #5b574f; font-size: 13.5px; margin: 0; }
"""
JS = """
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement;
  const synth = window.speechSynthesis;
  const status = root.querySelector('[data-status]');
  if (!synth || typeof SpeechSynthesisUtterance === 'undefined') {
    root.querySelector('.tts').innerHTML = '<p class="note">Read-aloud isn\\'t supported in this browser.</p>';
    return;
  }
  const select = root.querySelector('[data-voice]');
  const speed = root.querySelector('[data-speed]');
  const speedLabel = root.querySelector('[data-speed-label]');
  const fillVoices = () => {
    const voices = synth.getVoices();
    if (!voices.length || select.options.length) return;
    voices.forEach((voice, index) => {
      const option = document.createElement('option');
      option.value = String(index); option.textContent = voice.name + ' (' + voice.lang + ')';
      if (voice.default || (!select.value && voice.lang && voice.lang.startsWith('en'))) option.selected = true;
      select.appendChild(option);
    });
  };
  fillVoices();
  synth.addEventListener('voiceschanged', fillVoices);
  speed.oninput = () => { speedLabel.textContent = Number(speed.value).toFixed(1) + '×'; };
  const sentences = (data.text || '').match(/[^.!?\\n]+[.!?]*\\s*/g) || [];
  const play = () => {
    if (synth.paused && window.__psSpeaking === data.id) { synth.resume(); status.textContent = 'Reading…'; return; }
    synth.cancel();
    window.__psSpeaking = data.id;
    const voice = synth.getVoices()[Number(select.value)];
    sentences.forEach((sentence, index) => {
      const utterance = new SpeechSynthesisUtterance(sentence.trim());
      if (voice) utterance.voice = voice;
      utterance.rate = Number(speed.value);
      if (index === sentences.length - 1) utterance.onend = () => { status.textContent = 'Finished.'; window.__psSpeaking = null; };
      synth.speak(utterance);
    });
    status.textContent = 'Reading…';
  };
  root.querySelector('[data-act="play"]').onclick = play;
  root.querySelector('[data-act="pause"]').onclick = () => { synth.pause(); status.textContent = 'Paused.'; };
  root.querySelector('[data-act="stop"]').onclick = () => { synth.cancel(); window.__psSpeaking = null; status.textContent = 'Stopped.'; };
}
"""


@st.cache_resource
def _player():
    return st.components.v2.component("paper_scout_tts", html=HTML, css=CSS, js=JS)


def speech_text(answer_md):
    """Short answer first, then the body, without citations, tables or Markdown."""
    short, rest = split_short_answer(answer_md)
    rest = re.sub(r"(?im)^#{1,6}\s*", "", rest)
    return "\n".join(part for part in (strip_for_speech(short), strip_for_speech(rest)) if part)


def render_panel(record, mid):
    st.html("<div class='panel-h'><b>Listen</b></div>")
    _player()(key=f"listen_{mid}_player", data={"id": mid, "text": speech_text(record["answer_md"])})
