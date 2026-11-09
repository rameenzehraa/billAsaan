"""Page styling for the public app: one stylesheet, two token sets (light/dark).

Direction: "the meter, made legible". Calm paper and slate ink; the one bold element is the meter-LCD panel that holds
the amount due and due date. Atkinson Hyperlegible everywhere (set in .streamlit/config.toml); Share Tech Mono only for
the two LCD readouts. Motion: results fade/rise once after processing; buttons press to 0.97. Nothing else moves.
"""

TOKENS = {
    "light": {"ink": "#18242C", "muted": "#56656D", "line": "#D3DCD4", "accent": "#1D6B74", "rust": "#B5541F",
              "lcd": "#D5E2C6", "lcd_ink": "#1E2A16", "lcd_line": "#B9C9A8", "bar": "#A9B7AE", "surface": "#FFFFFF"},
    "dark": {"ink": "#E3EBE6", "muted": "#9AABA3", "line": "#2A3A36", "accent": "#5BB8C2", "rust": "#E08A5A",
             "lcd": "#141F17", "lcd_ink": "#B9E07C", "lcd_line": "#2E4231", "bar": "#4A5A55", "surface": "#132027"},
}


def tokens(theme_type: str | None) -> dict:
    return TOKENS["dark" if theme_type == "dark" else "light"]


def css(theme_type: str | None, animate_results: bool = False) -> str:
    """The page stylesheet as one <style> block. Its body must never contain '<': st.html sanitises with DOMPurify,
    which silently drops the whole style element if the CSS text looks like markup (e.g. a comment mentioning <p>)."""
    t = tokens(theme_type)
    entrance = """
@media (prefers-reduced-motion: no-preference) {
  .st-key-results { animation: ba-rise 260ms var(--ba-ease-out) both; }
  .st-key-results .st-key-hero { animation: ba-rise 260ms var(--ba-ease-out) 40ms both; }
}
@media (prefers-reduced-motion: reduce) {
  .st-key-results { animation: ba-fade 200ms ease both; }
}
@keyframes ba-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes ba-fade { from { opacity: 0; } to { opacity: 1; } }
""" if animate_results else ""
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Noto+Naskh+Arabic:wght@400;600&display=swap');
:root {{
  --ba-ink: {t['ink']}; --ba-muted: {t['muted']}; --ba-line: {t['line']}; --ba-accent: {t['accent']};
  --ba-rust: {t['rust']}; --ba-lcd: {t['lcd']}; --ba-lcd-ink: {t['lcd_ink']}; --ba-lcd-line: {t['lcd_line']};
  --ba-surface: {t['surface']};
  --ba-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
}}

/* ---------- platform layer (mobile-native) ---------- */
html {{ -webkit-tap-highlight-color: transparent; -webkit-text-size-adjust: 100%; }}
input, textarea, select, [data-testid="stChatInputTextArea"] {{ font-size: 16px !important; }}
button, [role="button"], [data-testid="stCheckbox"] label {{
  touch-action: manipulation; user-select: none; -webkit-user-select: none;
}}
.stButton button, [data-testid="stFileUploaderDropzone"] button, [data-testid="stChatInputSubmitButton"] {{
  min-height: 44px;
}}
[data-testid="stCheckbox"] label {{ min-height: 44px; align-items: center; }}
[data-testid="stChatInputTextArea"] {{ min-height: 44px; }}
@media (pointer: coarse) {{  /* Streamlit's small icon buttons: 44px hit areas on touch screens */
  [data-testid="stElementToolbar"] button {{ min-width: 44px; min-height: 44px; }}
  [data-testid="stTooltipIcon"], [data-testid="stTooltipHoverTarget"], [data-testid="stTooltipHoverTarget"] button {{
    min-width: 44px; min-height: 44px; display: inline-flex; align-items: center; justify-content: center;
  }}
}}
.stMainBlockContainer {{ max-width: 1120px; padding-top: 2.5rem; }}

/* ---------- press feedback (emil-design-eng): 0.97 on press, hover only for fine pointers ---------- */
@media (prefers-reduced-motion: no-preference) {{
  .stButton button, [data-testid="stFileUploaderDropzone"] button {{ transition: transform 140ms var(--ba-ease-out); }}
  .stButton button:active:not(:disabled), [data-testid="stFileUploaderDropzone"] button:active {{ transform: scale(0.97); }}
}}
@media (hover: hover) and (pointer: fine) {{
  .st-key-chips .stButton button:hover:not(:disabled) {{ border-color: var(--ba-accent); color: var(--ba-accent); }}
}}

/* ---------- header ---------- */
.st-key-brand h1 {{ letter-spacing: -0.02em; line-height: 1.05; margin-bottom: 0.15rem; }}
.st-key-brand .ba-tagline p {{ font-size: 1.2rem; font-weight: 700; color: var(--ba-accent); margin: 0; }}
.st-key-brand [data-testid="stCaptionContainer"] p {{ font-size: 1rem; color: var(--ba-muted); max-width: 62ch; }}
.st-key-consent_box {{
  border-left: 3px solid var(--ba-accent); padding: 0.25rem 0 0.25rem 0.9rem; max-width: 62ch;
}}

/* ---------- results: bill image ---------- */
.st-key-billimg img {{ max-height: 640px; object-fit: contain; object-position: top; }}
@media (max-width: 640px) {{ .st-key-billimg img {{ max-height: 340px; }} }}

/* ---------- results: LCD hero (amount due + due date) ---------- */
.st-key-hero {{
  background: var(--ba-lcd); border: 1px solid var(--ba-lcd-line); border-radius: 14px; padding: 1rem 1.15rem 0.9rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.35), inset 0 -2px 6px rgba(0,0,0,0.06);
}}
.st-key-hero [data-testid="stMetric"] {{ background: transparent; border: 0; padding: 0; box-shadow: none; }}
.st-key-hero [data-testid="stMetricLabel"] p {{ color: var(--ba-lcd-ink); opacity: 0.78; font-weight: 700; font-size: 0.95rem; }}
.st-key-hero [data-testid="stMetricValue"] {{
  font-family: 'Share Tech Mono', ui-monospace, monospace; color: var(--ba-lcd-ink);
  font-size: clamp(1.75rem, 1.1rem + 2.6vw, 2.75rem); line-height: 1.1; letter-spacing: 0.01em;
  font-variant-numeric: tabular-nums;
}}
.st-key-hero .st-key-hero_date [data-testid="stMetricValue"] {{ font-size: clamp(1.35rem, 1rem + 1.4vw, 1.8rem); }}
.st-key-hero_date {{ border-top: 1px dashed var(--ba-lcd-line); padding-top: 0.55rem; margin-top: 0.15rem; }}
/* Never truncate amounts or dates (break-ui): Streamlit sets nowrap + ellipsis on the inner p element; wrap it. */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] p,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  overflow-wrap: anywhere;
}}

/* ---------- results: secondary facts ---------- */
.st-key-facts [data-testid="stMetric"] {{ padding: 0.6rem 0.8rem; }}
/* Keep the facts a 2x2 grid on phones instead of four stacked cards; values wrap inside each half. */
@media (max-width: 640px) {{
  .st-key-facts [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; gap: 0.6rem; }}
  .st-key-facts [data-testid="stColumn"] {{ width: auto !important; flex: 1 1 0 !important; min-width: 0 !important; }}
  .st-key-facts [data-testid="stMetricValue"] {{ font-size: 1.15rem; }}
}}
.st-key-facts [data-testid="stMetricValue"] {{ font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.st-key-facts [data-testid="stMetricLabel"] p {{ color: var(--ba-muted); }}

/* ---------- checks: compact rows, reasons readable ---------- */
.st-key-checks [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
.st-key-checks .ba-check-name p {{ margin: 0; font-weight: 700; }}
.st-key-checks [data-testid="stCaptionContainer"] p {{ color: var(--ba-muted); font-size: 0.92rem; margin: -0.1rem 0 0.35rem; }}
.st-key-checks hr {{ margin: 0.25rem 0; }}
/* Badge stays put; a long check name wraps beside it instead of dropping onto its own line (phones). */
.st-key-checks [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; align-items: flex-start; }}
.st-key-checks [data-testid="stHorizontalBlock"] > [data-testid="stElementContainer"] {{ flex-shrink: 0; }}
.st-key-checks [data-testid="stHorizontalBlock"] > [data-testid="stElementContainer"]:last-child {{
  flex: 1 1 0; flex-shrink: 1; min-width: 0;
}}

/* ---------- chat ---------- */
/* Per-paragraph direction: Urdu-script lines read right-to-left, English and Roman Urdu left-to-right. */
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] [data-testid="stText"],
[data-testid="stChatMessage"] [data-testid="stText"] * {{
  line-height: 1.6; max-width: 70ch; unicode-bidi: plaintext; text-align: start;
  overflow-wrap: anywhere; font-family: 'Atkinson Hyperlegible', 'Noto Naskh Arabic', sans-serif;
}}
[data-testid="stChatMessage"] [data-testid="stText"],
[data-testid="stChatMessage"] [data-testid="stText"] * {{ text-align: start !important; }}
.st-key-chips .stButton button {{ min-height: 44px; border-radius: 999px; padding: 0.35rem 0.95rem; }}
.st-key-chips {{ flex-wrap: wrap; }}
{entrance}</style>"""
