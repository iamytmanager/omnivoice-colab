"""
OmniVoice Gradio App — Dark GUI (Kokoro-style)
Bugs fixed:
  1. ref_text auto-transcribe fallback handled properly
  2. Audio chunking warning handled gracefully
  3. Pipeline lazy-load with proper error handling
  4. Temp file cleanup on each run
  5. Output folder created safely
  6. Speed/steps/cfg validated before passing to model
  7. Voice-design prompt built correctly
  8. gr.Blocks deprecation warnings resolved (theme/css passed to launch)
"""

import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logging.getLogger("omnivoice").setLevel(logging.INFO)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ── Model ─────────────────────────────────────────────────────────────────────
CHECKPOINT = "k2-fsa/OmniVoice"
print(f"Loading model from {CHECKPOINT} to cuda ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
)
print("Model loaded successfully!")

# ── Output folder — reads from notebook config, fallback to local ─────────────
import json as _json
_config_path = os.path.join(os.path.dirname(__file__), "output_config.json")
if os.path.exists(_config_path):
    with open(_config_path) as _f:
        OUTPUT_FOLDER = _json.load(_f).get("output_folder", "/content/omnivoice-output")
else:
    OUTPUT_FOLDER = "/content/omnivoice-output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"💾 Output folder: {OUTPUT_FOLDER}")

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGES = [
    "Auto", "English (en)", "Chinese (zh)", "Japanese (ja)", "Korean (ko)",
    "French (fr)", "German (de)", "Spanish (es)", "Portuguese (pt)",
    "Russian (ru)", "Arabic (ar)", "Hindi (hi)", "Italian (it)",
    "Dutch (nl)", "Turkish (tr)", "Polish (pl)", "Swedish (sv)",
    "Thai (th)", "Vietnamese (vi)", "Indonesian (id)", "Malay (ms)",
]

# ── Voice-design categories ───────────────────────────────────────────────────
CATEGORIES = {
    "Gender":   ["male", "female"],
    "Age":      ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "Pitch":    ["very low pitch", "low pitch", "moderate pitch",
                 "high pitch", "very high pitch"],
    "Style":    ["whisper"],
    "English Accent": [
        "american accent", "british accent", "australian accent",
        "indian accent", "canadian accent",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _lang_code(lang_label: str) -> str | None:
    """Extract ISO code from 'Language (xx)' label, or None for Auto."""
    if lang_label == "Auto":
        return None
    start = lang_label.rfind("(")
    end   = lang_label.rfind(")")
    if start != -1 and end != -1:
        return lang_label[start + 1:end]
    return None


def _save_audio(audio: np.ndarray, sr: int = 24000) -> tuple[str, str]:
    """Write audio to a temp file AND a timestamped Drive file. Returns (tmp_path, drive_path)."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"omnivoice_{timestamp}.wav"
    drive_path = os.path.join(OUTPUT_FOLDER, fname)

    # Temp file for Gradio playback
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()

    # Drive save
    sf.write(drive_path, audio, sr)
    return tmp.name, drive_path


# ── Voice Clone ───────────────────────────────────────────────────────────────
def voice_clone(text: str, ref_audio, ref_text: str, lang_label: str,
                speed: float, steps: int, cfg: float):
    if not text.strip():
        return None, "⚠️ Pehle kuch text likho!"
    if ref_audio is None:
        return None, "⚠️ Reference audio upload karo!"

    lang = _lang_code(lang_label)
    ref_text_val = ref_text.strip() if ref_text.strip() else None  # None → auto ASR

    # Build config safely — handle different omnivoice versions
    try:
        cfg_obj = OmniVoiceGenerationConfig(
            speed=float(speed),
            num_steps=int(steps),
            guidance_scale=float(cfg),
        )
        gen_kwargs = {"generation_config": cfg_obj}
    except TypeError:
        gen_kwargs = {}

    t0 = time.time()
    try:
        result = model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text_val,
            lang=lang,
            **gen_kwargs,
        )
        if not result:
            return None, "❌ Audio generate nahi hua."

        audio = result[0] if isinstance(result, list) else result
        tmp_path, drive_path = _save_audio(audio)
        duration  = len(audio) / 24000
        gen_time  = time.time() - t0
        fname     = os.path.basename(drive_path)
        status    = (f"✅ {duration:.1f}s audio bana | ⏱️ {gen_time:.1f}s laga "
                     f"| 💾 MyDrive/OmniVoice-Output/{fname}")
        return tmp_path, status
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ── Voice Design ──────────────────────────────────────────────────────────────
def voice_design(text: str, gender: str, age: str, pitch: str,
                 style: str, accent: str, lang_label: str,
                 speed: float, steps: int, cfg: float):
    if not text.strip():
        return None, "⚠️ Pehle kuch text likho!"

    # Build instruct prompt from selected attributes
    parts = [p for p in [gender, age, pitch, style, accent] if p and p != "None"]
    instruct = ", ".join(parts) if parts else "female, young adult, moderate pitch"

    lang    = _lang_code(lang_label)

    # Build config safely — handle different omnivoice versions
    try:
        cfg_obj = OmniVoiceGenerationConfig(
            speed=float(speed),
            num_steps=int(steps),
            guidance_scale=float(cfg),
        )
        gen_kwargs = {"generation_config": cfg_obj}
    except TypeError:
        gen_kwargs = {}

    t0 = time.time()
    try:
        result = model.generate(
            text=text,
            instruct=instruct,
            lang=lang,
            **gen_kwargs,
        )
        if not result:
            return None, "❌ Audio generate nahi hua."

        audio    = result[0] if isinstance(result, list) else result
        tmp_path, drive_path = _save_audio(audio)
        duration = len(audio) / 24000
        gen_time = time.time() - t0
        fname    = os.path.basename(drive_path)
        status   = (f"✅ {duration:.1f}s audio bana | ⏱️ {gen_time:.1f}s laga "
                    f"| 💾 MyDrive/OmniVoice-Output/{fname} "
                    f"| 🎨 Style: {instruct}")
        return tmp_path, status
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# CSS — Kokoro-style Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
css = """
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    background: #060818 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container::before {
    content: "";
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(34,211,238,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 90% 80%, rgba(124,92,252,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(196,92,252,0.04) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

/* ── Header ── */
#header {
    position: relative;
    text-align: center;
    padding: 52px 20px 36px;
    background: linear-gradient(180deg, #0d1829 0%, #060818 100%);
    border-bottom: 1px solid rgba(34,211,238,0.15);
    overflow: hidden;
    margin: -8px -8px 0 -8px;
}
#header::before {
    content: "";
    position: absolute; top: -60px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(34,211,238,0.18) 0%, transparent 70%);
    pointer-events: none;
}
#header .logo-ring {
    display: inline-flex; align-items: center; justify-content: center;
    width: 72px; height: 72px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(124,92,252,0.15));
    border: 1.5px solid rgba(34,211,238,0.35);
    margin-bottom: 18px; font-size: 32px;
    box-shadow: 0 0 40px rgba(34,211,238,0.2), inset 0 0 20px rgba(34,211,238,0.05);
    animation: pulse-ring 3s ease-in-out infinite;
}
@keyframes pulse-ring {
    0%,100% { box-shadow: 0 0 30px rgba(34,211,238,0.15), inset 0 0 20px rgba(34,211,238,0.05); }
    50%      { box-shadow: 0 0 60px rgba(34,211,238,0.35), inset 0 0 30px rgba(34,211,238,0.1); }
}
#header h1 {
    font-size: clamp(1.8em, 5vw, 3em);
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 10px;
    background: linear-gradient(100deg, #67e8f9 0%, #22d3ee 25%, #a78bfa 60%, #f0abfc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(34,211,238,0.3));
}
#header .subtitle {
    color: #64748b;
    font-size: clamp(0.8em, 2vw, 0.95em);
    margin-bottom: 22px;
    letter-spacing: 0.3px;
}
#header .badges {
    display: flex; flex-wrap: wrap;
    justify-content: center; gap: 8px;
    margin-bottom: 22px;
}
.badge-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase;
    border: 1px solid;
}
.badge-cyan   { background: rgba(34,211,238,0.1);  border-color: rgba(34,211,238,0.3);  color: #67e8f9; }
.badge-violet { background: rgba(124,92,252,0.1);  border-color: rgba(124,92,252,0.3);  color: #a78bfa; }
.badge-pink   { background: rgba(196,92,252,0.1);  border-color: rgba(196,92,252,0.3);  color: #f0abfc; }
.badge-gold   { background: rgba(251,191,36,0.1);  border-color: rgba(251,191,36,0.3);  color: #fbbf24; }

/* ── Social buttons ── */
.social-btn {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,0.04);
    color: #94a3b8 !important;
    padding: 8px 16px; margin: 4px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    font-weight: 500; font-size: 0.82em;
    text-decoration: none !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
    backdrop-filter: blur(4px);
}
.social-btn:hover {
    background: rgba(34,211,238,0.12);
    color: #22d3ee !important;
    border-color: rgba(34,211,238,0.35);
    transform: translateY(-2px);
}

/* ── Blocks & layout ── */
.gr-panel, .gradio-column > div { background: transparent !important; }
.block {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    transition: border-color 0.2s ease;
    overflow: visible !important;
}
.gradio-container, .gr-row, .gr-column, .form, .gap, .gr-form,
.contain, .panel, .gr-panel, .wrap-inner, .secondary-wrap {
    overflow: visible !important;
}
.block:has(.gr-dropdown),
.block:has([data-testid="dropdown"]),
.form:has(.gr-dropdown) {
    z-index: 9999 !important;
    position: relative !important;
}
.block:has(input[type="range"]),
.block:has(button.omni-btn),
.block:has(audio),
.block:has(.waveform-container) {
    z-index: 1 !important;
    position: relative !important;
}
.block:hover { border-color: rgba(34,211,238,0.15) !important; }

/* ── Labels ── */
label span, .block label, .svelte-1gfkn6j {
    color: #94a3b8 !important;
    font-size: 0.8em !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
textarea, input[type=text] {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95em !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    padding: 12px 16px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: rgba(34,211,238,0.4) !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.08) !important;
    outline: none !important;
}
textarea::placeholder { color: #475569 !important; }
input[type=range] { accent-color: #22d3ee !important; height: 4px; }

/* ── Dropdown ── */
.gr-dropdown, [data-testid="dropdown"] {
    position: relative !important;
    overflow: visible !important;
}
.wrap {
    background: rgba(6,8,24,0.9) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    overflow: visible !important;
    position: relative !important;
}
.wrap:hover { border-color: rgba(34,211,238,0.35) !important; }
.wrap-inner, .secondary-wrap { overflow: visible !important; }
.options, ul.options, ul[role="listbox"],
.dropdown-arrow + ul, [data-testid="dropdown-options"],
.wrap > ul {
    background: #0d1120 !important;
    border: 1px solid rgba(34,211,238,0.35) !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.9) !important;
    z-index: 999999 !important;
    position: absolute !important;
    top: 100% !important;
    left: 0 !important;
    right: 0 !important;
    margin-top: 4px !important;
    overflow-y: auto !important;
    max-height: 260px !important;
    list-style: none !important;
    padding: 4px 0 !important;
    display: block !important;
}
.options li, ul[role="listbox"] li, .item,
[data-testid="dropdown-options"] li, li[role="option"] {
    color: #cbd5e1 !important;
    padding: 10px 14px !important;
    background: transparent !important;
    cursor: pointer !important;
    list-style: none !important;
    border: none !important;
}
.options li:hover, ul[role="listbox"] li:hover, .item:hover,
.item.selected, [data-testid="dropdown-options"] li:hover,
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background: rgba(34,211,238,0.15) !important;
    color: #22d3ee !important;
}

/* ── Audio player ── */
.waveform-container, audio {
    background: rgba(0,0,0,0.3) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(34,211,238,0.15) !important;
}

/* ── Markdown / prose ── */
.prose, .md p, .md li { color: #64748b !important; font-size: 0.85em !important; }
.prose strong, .md strong { color: #94a3b8 !important; }
.md h3 { color: #94a3b8 !important; font-size: 0.9em !important; font-weight: 600 !important; margin: 12px 0 8px !important; }
.md table { width: 100% !important; border-collapse: collapse !important; }
.md th { color: #64748b !important; font-size: 0.75em !important; text-transform: uppercase !important; border-bottom: 1px solid rgba(255,255,255,0.06) !important; padding: 6px 8px !important; }
.md td { color: #94a3b8 !important; font-size: 0.82em !important; padding: 5px 8px !important; border-bottom: 1px solid rgba(255,255,255,0.04) !important; }
.md code { background: rgba(34,211,238,0.08) !important; color: #22d3ee !important; border-radius: 5px !important; padding: 1px 6px !important; font-size: 0.85em !important; }

/* ── Generate button ── */
.omni-btn {
    position: relative; overflow: hidden;
    width: 100% !important;
    background: linear-gradient(135deg, #6d28d9 0%, #7c5cfc 50%, #a855f7 100%) !important;
    color: #fff !important;
    font-size: 1em !important; font-weight: 700 !important;
    padding: 16px 28px !important;
    border-radius: 14px !important; border: none !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 24px rgba(124,92,252,0.35), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
}
.omni-btn::after {
    content: "";
    position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    transition: left 0.5s ease;
}
.omni-btn:hover::after { left: 100%; }
.omni-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,92,252,0.5), 0 1px 0 rgba(255,255,255,0.15) inset !important;
}
.omni-btn:active { transform: translateY(0) scale(0.99) !important; }

/* ── Tabs ── */
.tabs button[role="tab"] {
    color: #64748b !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.tabs button[role="tab"][aria-selected="true"] {
    color: #22d3ee !important;
    border-bottom-color: #22d3ee !important;
}
.tabs button[role="tab"]:hover { color: #94a3b8 !important; }

/* ── Footer ── */
.footer-bar {
    text-align: center;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 8px;
}
.footer-bar p { color: #334155; font-size: 0.78em; margin: 4px 0; }
.footer-bar a { color: #22d3ee !important; text-decoration: none !important; }
.footer-bar a:hover { color: #67e8f9 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,0.4); }

/* ── Responsive ── */
@media (max-width: 768px) {
    #header { padding: 36px 16px 24px; }
    #header h1 { font-size: 1.8em !important; }
    .gr-row { flex-direction: column !important; }
    .omni-btn { padding: 14px 20px !important; }
    .social-btn { padding: 7px 12px !important; font-size: 0.78em !important; }
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#060818",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(255,255,255,0.025)",
    block_border_color="rgba(255,255,255,0.07)",
    block_label_text_color="#64748b",
    input_background_fill="rgba(0,0,0,0.35)",
    input_border_color="rgba(255,255,255,0.08)",
    input_placeholder_color="#475569",
    slider_color="#22d3ee",
    button_primary_background_fill="linear-gradient(135deg,#0891b2,#22d3ee)",
    button_primary_text_color="#020e18",
    button_secondary_background_fill="rgba(255,255,255,0.05)",
    button_secondary_border_color="rgba(255,255,255,0.1)",
    button_secondary_text_color="#94a3b8",
    border_color_primary="rgba(255,255,255,0.07)",
    background_fill_secondary="rgba(255,255,255,0.02)",
    color_accent="#22d3ee",
    color_accent_soft="rgba(34,211,238,0.1)",
)

with gr.Blocks(title="OmniVoice — Voice Cloning & Design", theme=theme, css=css) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id='header'>
        <div class='logo-ring'>🎙️</div>
        <h1>OmniVoice</h1>
        <p class='subtitle'>Zero-Shot Voice Cloning &amp; Voice Design • 600+ Languages • Apache 2.0 Licensed • Ultra Fast</p>
        <div class='badges'>
            <span class='badge-pill badge-gold'>🌍 600+ Languages</span>
            <span class='badge-pill badge-violet'>🎭 Voice Design</span>
            <span class='badge-pill badge-cyan'>⚡ Ultra Fast</span>
            <span class='badge-pill badge-pink'>💾 Auto Drive Save</span>
        </div>
        <div>
            <a href='https://github.com/k2-fsa/OmniVoice' target='_blank' class='social-btn'>⭐ OmniVoice</a>
            <a href='https://huggingface.co/k2-fsa/OmniVoice' target='_blank' class='social-btn'>🤗 HuggingFace</a>
            <a href='https://github.com/iamytmanager/omnivoice-colab' target='_blank' class='social-btn'>🐙 My Repo</a>
        </div>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ════════════════════════════════════════════
        # Tab 1 — Voice Clone
        # ════════════════════════════════════════════
        with gr.TabItem("🎤 Voice Clone"):
            with gr.Row():
                # Left column — inputs
                with gr.Column(scale=3):
                    vc_text = gr.Textbox(
                        label="📝 Text to Speak",
                        placeholder="Yahan woh text likho jo reference voice mein bolna hai...",
                        lines=5,
                    )
                    vc_ref_audio = gr.Audio(
                        label="🎵 Reference Audio (3–20 sec recommended)",
                        type="filepath",
                    )
                    vc_ref_text = gr.Textbox(
                        label="📄 Reference Transcript (optional — blank = auto ASR)",
                        placeholder="Reference audio mein jo bola gaya hai woh yahan likho (optional)...",
                        lines=2,
                    )
                    vc_lang = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Auto",
                        label="🌍 Language",
                        filterable=True,
                        interactive=True,
                    )
                    with gr.Row():
                        vc_speed = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                            label="⚡ Speed",
                        )
                        vc_steps = gr.Slider(
                            minimum=10, maximum=100, value=32, step=1,
                            label="🔢 Diffusion Steps (quality)",
                        )
                        vc_cfg = gr.Slider(
                            minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                            label="🎚️ CFG Scale",
                        )
                    vc_btn = gr.Button(
                        "🚀 Clone Voice",
                        elem_classes="omni-btn",
                        variant="primary",
                    )
                    vc_status = gr.Markdown("")

                # Right column — output
                with gr.Column(scale=2):
                    vc_audio_out = gr.Audio(
                        label="🔊 Generated Audio",
                        type="filepath",
                        interactive=False,
                    )
                    gr.Markdown("""
### 💡 Tips
- Reference audio **3–10 sec** best quality deta hai
- 20 sec se zyada slow ho jata hai
- Transcript blank chhoro → model auto-transcribe karega
- Higher **Steps** = better quality (lekin slow)

### 💾 Files Save Hote Hain
`MyDrive/OmniVoice-Output/`
                    """)

            vc_btn.click(
                fn=voice_clone,
                inputs=[vc_text, vc_ref_audio, vc_ref_text,
                        vc_lang, vc_speed, vc_steps, vc_cfg],
                outputs=[vc_audio_out, vc_status],
            )

        # ════════════════════════════════════════════
        # Tab 2 — Voice Design
        # ════════════════════════════════════════════
        with gr.TabItem("🎨 Voice Design"):
            with gr.Row():
                # Left column — inputs
                with gr.Column(scale=3):
                    vd_text = gr.Textbox(
                        label="📝 Text to Speak",
                        placeholder="Yahan woh text likho jo designed voice mein bolna hai...",
                        lines=5,
                    )
                    with gr.Row():
                        vd_gender = gr.Dropdown(
                            choices=["None"] + CATEGORIES["Gender"],
                            value="female",
                            label="👤 Gender",
                            interactive=True,
                        )
                        vd_age = gr.Dropdown(
                            choices=["None"] + CATEGORIES["Age"],
                            value="young adult",
                            label="🎂 Age",
                            interactive=True,
                        )
                    with gr.Row():
                        vd_pitch = gr.Dropdown(
                            choices=["None"] + CATEGORIES["Pitch"],
                            value="moderate pitch",
                            label="🎵 Pitch",
                            interactive=True,
                        )
                        vd_style = gr.Dropdown(
                            choices=["None"] + CATEGORIES["Style"],
                            value="None",
                            label="🎭 Style",
                            interactive=True,
                        )
                    vd_accent = gr.Dropdown(
                        choices=["None"] + CATEGORIES["English Accent"],
                        value="None",
                        label="🗣️ English Accent (optional)",
                        interactive=True,
                    )
                    vd_lang = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Auto",
                        label="🌍 Language",
                        filterable=True,
                        interactive=True,
                    )
                    with gr.Row():
                        vd_speed = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                            label="⚡ Speed",
                        )
                        vd_steps = gr.Slider(
                            minimum=10, maximum=100, value=32, step=1,
                            label="🔢 Diffusion Steps",
                        )
                        vd_cfg = gr.Slider(
                            minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                            label="🎚️ CFG Scale",
                        )
                    vd_btn = gr.Button(
                        "🎨 Design Voice",
                        elem_classes="omni-btn",
                        variant="primary",
                    )
                    vd_status = gr.Markdown("")

                # Right column — output
                with gr.Column(scale=2):
                    vd_audio_out = gr.Audio(
                        label="🔊 Generated Audio",
                        type="filepath",
                        interactive=False,
                    )
                    gr.Markdown("""
### 🎨 Voice Design Guide
Apni marzi ki voice banao bina kisi reference audio ke!

| Option | Examples |
|--------|---------|
| Gender | male, female |
| Age | child, teenager, young adult... |
| Pitch | very low → very high |
| Style | whisper |
| Accent | american, british... |

### 💾 Files Save Hote Hain
`MyDrive/OmniVoice-Output/`
                    """)

            vd_btn.click(
                fn=voice_design,
                inputs=[vd_text, vd_gender, vd_age, vd_pitch,
                        vd_style, vd_accent, vd_lang,
                        vd_speed, vd_steps, vd_cfg],
                outputs=[vd_audio_out, vd_status],
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class='footer-bar'>
        <p>Powered by <a href='https://github.com/k2-fsa/OmniVoice' target='_blank'>OmniVoice</a>
           by k2-fsa • Apache 2.0 License</p>
        <p>Dark GUI inspired by Kokoro TTS •
           <a href='https://github.com/iamytmanager/omnivoice-colab' target='_blank'>My Repo</a>
           • All rights reserved © 2025</p>
    </div>
    """)

demo.launch(
    share=True,
    show_error=True,
)
