import gradio as gr
import os
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
import threading
import queue

# ── Load config ────────────────────────────────────────────────────────────────
CONFIG_PATH = '/content/omnivoice-colab/output_config.json'
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    OUTPUT_FOLDER = cfg.get('output_folder', '/content/omnivoice-output')
    MODEL_CACHE_FOLDER = cfg.get('model_cache_folder', '/content/omnivoice-model-cache')
else:
    OUTPUT_FOLDER = '/content/omnivoice-output'
    MODEL_CACHE_FOLDER = '/content/omnivoice-model-cache'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_CACHE_FOLDER, exist_ok=True)

# ── Load OmniVoice model ───────────────────────────────────────────────────────
print("🔄 Model load ho raha hai...")
import torch
from omnivoice import OmniVoice

# Correct API: OmniVoice.from_pretrained() — OmniVoice() directly nahi hota
# HuggingFace cache folder environment variable se set karo
os.environ['HF_HOME'] = MODEL_CACHE_FOLDER

tts = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
print("✅ Model ready!")


# ══════════════════════════════════════════════════════════════════════════════
# SILENCE REMOVAL  (pydub-free, pure numpy+soundfile)
# Logic:
#   1. Audio ko frames mein divide karo
#   2. Har frame ka RMS energy check karo
#   3. Silence threshold se neeche wale frames remove karo
#   4. Natural pauses (short silences < min_silence_keep_ms) preserve karo
#   5. Leading/trailing silence trim karo
# ══════════════════════════════════════════════════════════════════════════════
def remove_silence(
    audio_path: str,
    silence_thresh_db: float = -40.0,   # dB threshold — isse neeche = silence
    min_silence_ms: int = 400,           # silence isse zyada lamba ho to remove
    keep_silence_ms: int = 80,           # har silence chunk ke start/end pe yeh ms rakhna (natural breath)
    frame_ms: int = 20,                  # analysis frame size
) -> str:
    """
    Remove long silence gaps from audio while keeping natural pauses.
    Returns path to cleaned audio file.
    """
    audio, sr = sf.read(audio_path)

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    frame_size = int(sr * frame_ms / 1000)
    min_silence_frames = int(min_silence_ms / frame_ms)
    keep_frames = int(keep_silence_ms / frame_ms)

    # Compute RMS per frame
    frames = []
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) == 0:
            break
        rms = np.sqrt(np.mean(frame ** 2))
        rms_db = 20 * np.log10(rms + 1e-9)
        frames.append((i, frame, rms_db))

    # Mark silence frames
    is_silence = [db < silence_thresh_db for (_, _, db) in frames]

    # Find silence runs and decide what to keep
    keep_mask = [True] * len(frames)
    i = 0
    while i < len(is_silence):
        if is_silence[i]:
            # Find run end
            j = i
            while j < len(is_silence) and is_silence[j]:
                j += 1
            run_len = j - i
            if run_len > min_silence_frames:
                # Remove middle, keep `keep_frames` at each end
                start_keep = i + keep_frames
                end_keep = j - keep_frames
                if start_keep < end_keep:
                    for k in range(start_keep, end_keep):
                        keep_mask[k] = False
            i = j
        else:
            i += 1

    # Reconstruct audio
    chunks = [frames[k][1] for k in range(len(frames)) if keep_mask[k]]
    if not chunks:
        return audio_path  # kuch nahi kiya, original return karo

    cleaned = np.concatenate(chunks)

    # Save
    out_name = Path(audio_path).stem + '_cleaned.wav'
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    sf.write(out_path, cleaned, sr)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def generate_voice_clone(
    text: str,
    ref_audio,
    ref_transcript: str,
    steps: int,
    speed_factor: float,
    remove_sil: bool,
    sil_thresh_db: float,
    min_sil_ms: int,
    auto_dl: bool,
):
    if not text.strip():
        raise gr.Error("❌ Text to Speak khali hai!")

    if ref_audio is None or (isinstance(ref_audio, str) and not ref_audio.strip()):
        raise gr.Error("❌ Reference audio upload karo!")

    if isinstance(ref_audio, str) and not os.path.exists(ref_audio):
        raise gr.Error("❌ Reference audio file mil nahi rahi — dobara upload karo!")

    yield None, "", "⏳ Shuruaat ho rahi hai..."

    result_q = queue.Queue()

    def _run():
        try:
            timestamp = int(time.time())
            out_filename = f"omnivoice_{timestamp}.wav"
            out_path = os.path.join(OUTPUT_FOLDER, out_filename)

            ref_audio_path = ref_audio if isinstance(ref_audio, str) else ref_audio
            generate_kwargs = dict(
                text=text,
                ref_audio=ref_audio_path,
                num_step=steps,
                speed=speed_factor,
            )
            if ref_transcript and ref_transcript.strip():
                generate_kwargs['ref_text'] = ref_transcript.strip()

            audio_list = tts.generate(**generate_kwargs)
            if not audio_list:
                result_q.put(("error", "Audio generate nahi hui — model ne kuch return nahi kiya!"))
                return

            audio_np = audio_list[0]
            sf.write(out_path, audio_np, 24000)

            if remove_sil:
                result_q.put(("status", "✂️ Silence remove ho rahi hai..."))
                final_path = remove_silence(
                    out_path,
                    silence_thresh_db=sil_thresh_db,
                    min_silence_ms=min_sil_ms,
                )
                # Raw file delete karo — sirf clean version rakhna hai
                if final_path != out_path and os.path.exists(out_path):
                    os.remove(out_path)
            else:
                final_path = out_path

            saved_name = os.path.basename(final_path)
            dl_trigger = saved_name if auto_dl else ""
            result_q.put(("done", final_path, dl_trigger, f"✅ Done! Saved: {saved_name}"))

        except Exception as e:
            result_q.put(("error", str(e)))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # UI responsive rakhne ke liye — har 0.5 sec pe status update karo
    dots = 0
    while t.is_alive():
        dots = (dots + 1) % 4
        dot_str = "●" * (dots + 1) + "○" * (3 - dots)
        yield None, "", f"🔄 Voice generate ho rahi hai {dot_str} — please wait..."
        time.sleep(0.5)

    # Thread khatam — result lo
    while not result_q.empty():
        item = result_q.get()
        if item[0] == "status":
            yield None, "", item[1]
        elif item[0] == "done":
            _, filepath, dl_trigger, status_msg = item
            yield filepath, dl_trigger, status_msg
        elif item[0] == "error":
            raise gr.Error(f"❌ Error: {item[1]}")


def generate_voice_design(
    text: str,
    gender: str,
    age: str,
    emotion: str,
    steps: int,
    speed_factor: float,
    remove_sil: bool,
    sil_thresh_db: float,
    min_sil_ms: int,
    auto_dl: bool,
):
    if not text.strip():
        raise gr.Error("❌ Text to Speak khali hai!")

    yield None, "", "⏳ Shuruaat ho rahi hai..."

    result_q = queue.Queue()

    def _run():
        try:
            timestamp = int(time.time())
            out_filename = f"omnivoice_design_{timestamp}.wav"
            out_path = os.path.join(OUTPUT_FOLDER, out_filename)

            instruct_parts = [gender.lower(), age.lower()]
            if emotion.lower() != "neutral":
                instruct_parts.append(emotion.lower())
            instruct_str = ", ".join(instruct_parts)

            audio_list = tts.generate(
                text=text,
                instruct=instruct_str,
                num_step=steps,
                speed=speed_factor,
            )
            if not audio_list:
                result_q.put(("error", "Audio generate nahi hui!"))
                return

            audio_np = audio_list[0]
            sf.write(out_path, audio_np, 24000)

            if remove_sil:
                result_q.put(("status", "✂️ Silence remove ho rahi hai..."))
                final_path = remove_silence(
                    out_path,
                    silence_thresh_db=sil_thresh_db,
                    min_silence_ms=min_sil_ms,
                )
                # Raw file delete karo — sirf clean version rakhna hai
                if final_path != out_path and os.path.exists(out_path):
                    os.remove(out_path)
            else:
                final_path = out_path

            saved_name = os.path.basename(final_path)
            dl_trigger = saved_name if auto_dl else ""
            result_q.put(("done", final_path, dl_trigger, f"✅ Done! Saved: {saved_name}"))

        except Exception as e:
            result_q.put(("error", str(e)))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    dots = 0
    while t.is_alive():
        dots = (dots + 1) % 4
        dot_str = "●" * (dots + 1) + "○" * (3 - dots)
        yield None, "", f"🔄 Voice design ho rahi hai {dot_str} — please wait..."
        time.sleep(0.5)

    while not result_q.empty():
        item = result_q.get()
        if item[0] == "status":
            yield None, "", item[1]
        elif item[0] == "done":
            _, filepath, dl_trigger, status_msg = item
            yield filepath, dl_trigger, status_msg
        elif item[0] == "error":
            raise gr.Error(f"❌ Error: {item[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Base reset ─────────────────────────────────────────────────────────── */
body, .gradio-container, .main, footer {
    background: #0D0D0D !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #F0F0F0 !important;
}
footer { display: none !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ── Hide Gradio header chrome ──────────────────────────────────────────── */
.app-header, .share-btn, .svelte-1f354aw { display: none !important; }

/* ── Tab navigation ─────────────────────────────────────────────────────── */
.tab-nav { 
    background: #141414 !important;
    border-bottom: 2px solid #FF6B35 !important;
    padding: 10px 20px 0 !important;
    gap: 4px !important;
}
.tab-nav button {
    background: transparent !important;
    color: #888888 !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 9px 20px !important;
    margin: 0 !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
.tab-nav button:hover {
    color: #FF6B35 !important;
    background: rgba(255,107,53,0.06) !important;
}
.tab-nav button.selected {
    background: #0D0D0D !important;
    color: #FF6B35 !important;
    border-color: #333 !important;
    border-bottom: 2px solid #0D0D0D !important;
    margin-bottom: -2px !important;
}

/* ── Panels & forms ─────────────────────────────────────────────────────── */
.contain, .gap, .form, .block, .padded, .tabs, .tabitem {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    gap: 14px !important;
}

/* ── Labels ──────────────────────────────────────────────────────────────── */
label span, .label-wrap span {
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #FF6B35 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Inputs & textareas ──────────────────────────────────────────────────── */
textarea, input[type='text'], input[type='number'] {
    background: #1A1A1A !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 10px !important;
    color: #F0F0F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.75 !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input:focus {
    border-color: #FF6B35 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.15) !important;
}
textarea::placeholder, input::placeholder {
    color: #555 !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
input[type='range'] { accent-color: #FF6B35 !important; }
.wrap.svelte-1cl284s { color: #FF6B35 !important; font-weight: 700 !important; }

/* ── Dropdowns / selects ─────────────────────────────────────────────────── */
.wrap-inner, select, .multiselect {
    background: #1A1A1A !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 10px !important;
    color: #F0F0F0 !important;
}

/* ── Checkboxes ──────────────────────────────────────────────────────────── */
input[type='checkbox'] { accent-color: #FF6B35 !important; }
.checkbox-wrap label { 
    color: #CCCCCC !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Primary Button — orange gradient, glowing ──────────────────────────── */
button.primary, .gr-button.primary, button[variant='primary'] {
    background: linear-gradient(135deg, #FF6B35 0%, #FF4500 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 14px 24px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(255,107,53,0.35) !important;
    text-transform: uppercase !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #FF7F50 0%, #FF5500 100%) !important;
    box-shadow: 0 6px 28px rgba(255,107,53,0.5) !important;
    transform: translateY(-1px) !important;
}
button.secondary, .gr-button.secondary {
    background: #1A1A1A !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 8px !important;
    color: #AAAAAA !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    border-color: #FF6B35 !important;
    color: #FF6B35 !important;
}

/* ── Audio component — force dark everywhere ─────────────────────────────── */
.audio-container, .audio-wrap, [data-testid="audio"],
[data-testid="audio"] > div, [data-testid="audio"] > div > div,
.waveform-wrap, .waveform-container,
div[class*="waveform"], div[class*="audio"],
.component-wrapper {
    background: #141414 !important;
    background-color: #141414 !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 12px !important;
    color: #F0F0F0 !important;
}
/* Inner waveform canvas dark karo */
[data-testid="audio"] canvas,
[data-testid="audio"] wave,
[data-testid="audio"] wave canvas,
.waveform-wrap canvas, .waveform-wrap wave,
wave { background: #141414 !important; background-color: #141414 !important; }
/* Audio player */
.audio-container audio, audio {
    width: 100% !important;
    height: 36px !important;
    border-radius: 6px !important;
    accent-color: #FF6B35 !important;
    background: #141414 !important;
}
.waveform-wrap { background: #141414 !important; border-radius: 8px !important; }
/* Timestamps */
[data-testid="audio"] span, [data-testid="audio"] p { color: #888888 !important; }
/* Buttons inside audio */
[data-testid="audio"] button { color: #FF6B35 !important; background: transparent !important; }
/* Block wrappers */
.block.padded { background: #141414 !important; }

/* ── File upload zone ────────────────────────────────────────────────────── */
.upload-btn, .wrap.svelte-r2cif8 {
    background: #1A1A1A !important;
    border: 2px dashed #333333 !important;
    border-radius: 10px !important;
    color: #666666 !important;
    transition: all 0.2s !important;
}
.upload-btn:hover {
    border-color: #FF6B35 !important;
    color: #FF6B35 !important;
    background: rgba(255,107,53,0.05) !important;
}

/* ── Accordion ───────────────────────────────────────────────────────────── */
.accordion {
    background: #1A1A1A !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 10px !important;
}
.accordion-header {
    color: #AAAAAA !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
}

/* ── Status / HTML boxes ─────────────────────────────────────────────────── */
#status-box {
    background: #1A1A1A !important;
    border: 1px solid #2E2E2E !important;
    border-left: 3px solid #FF6B35 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    color: #FF6B35 !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* ── Tip box ─────────────────────────────────────────────────────────────── */
.tip-box {
    background: #141414 !important;
    border: 1px solid #1F2E1A !important;
    border-left: 3px solid #22C55E !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 12px 14px !important;
    font-size: 11.5px !important;
    color: #86EFAC !important;
    line-height: 2 !important;
}

/* ── Save path box ───────────────────────────────────────────────────────── */
.save-path {
    background: #141414 !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    color: #888888 !important;
}

/* ── Kill any remaining white in audio player ───────────────────────────── */
* { scrollbar-color: #333 #111 !important; }
.svelte-1oiin9d, .svelte-1oiin9d * { background: #141414 !important; }
/* WaveSurfer white canvas override */
canvas { background: #141414 !important; }
/* Timer box that shows duration */
.time, .duration, span.time, span.duration {
    color: #FF6B35 !important;
    background: transparent !important;
}
/* Any leftover white divs inside audio block */
[data-testid="audio"] * { 
    background-color: transparent !important; 
}
[data-testid="audio"] > div {
    background-color: #141414 !important;
}

/* ── Button disabled state during generation ─────────────────────────────── */
button.primary:disabled, button[variant="primary"]:disabled {
    background: #3A2A1A !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
    opacity: 0.6 !important;
}

/* ── Row layout ──────────────────────────────────────────────────────────── */
.row { gap: 18px !important; }

/* ── Waveform animation bars ─────────────────────────────────────────────── */
@keyframes ovWave {
    0%, 100% { transform: scaleY(0.6); opacity: 0.6; }
    50% { transform: scaleY(1.4); opacity: 1; }
}
.ov-bar {
    display: inline-block;
    width: 3px;
    background: #FF6B35;
    border-radius: 2px;
    margin: 0 1px;
    vertical-align: middle;
    animation: ovWave 1.2s ease-in-out infinite;
}

/* ── Info text / hints ───────────────────────────────────────────────────── */
.info-text, .gr-form .info {
    color: #666 !important;
    font-size: 11px !important;
}

/* ── Number input in sliders ─────────────────────────────────────────────── */
.wrap.svelte-1cl284s input {
    color: #FF6B35 !important;
    background: #1A1A1A !important;
    border-color: #333 !important;
}
"""

SILENCE_ACCORDION_LABEL = "✂️ Silence Removal Settings"

def silence_controls():
    """Reusable silence removal controls."""
    remove_sil = gr.Checkbox(
        label="Remove Silence (default ON — natural pauses rahenge, extra gaps hatengy)",
        value=True,
        elem_id="remove-sil-checkbox",
    )
    with gr.Accordion(SILENCE_ACCORDION_LABEL, open=False):
        sil_thresh = gr.Slider(
            label="Silence Threshold (dB) — zyada negative = sirf very quiet gaps remove hongy",
            minimum=-60,
            maximum=-20,
            value=-40,
            step=1,
        )
        min_sil_ms = gr.Slider(
            label="Min Silence Duration to Remove (ms) — chhote pauses safe rahenge",
            minimum=100,
            maximum=2000,
            value=400,
            step=50,
        )
    return remove_sil, sil_thresh, min_sil_ms


HEADER_HTML = """
<div style="
    background:linear-gradient(90deg,#141414 0%,#1A1208 100%);
    border-bottom:2px solid #FF6B35;
    padding:14px 24px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin:-8px -8px 0 -8px;
">
    <div style="display:flex;align-items:center;gap:12px;">
        <div style="
            width:40px;height:40px;
            background:linear-gradient(135deg,#FF6B35,#FF4500);
            border-radius:10px;
            display:flex;align-items:center;justify-content:center;
            box-shadow:0 4px 16px rgba(255,107,53,0.4);
        ">
            <svg width="20" height="20" viewBox="0 0 18 18" fill="none">
                <path d="M2 13 Q5 5 9 9 Q13 13 16 5" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" fill="none"/>
                <circle cx="9" cy="9" r="2.2" fill="#FFFFFF"/>
            </svg>
        </div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:19px;font-weight:700;color:#FFFFFF;letter-spacing:-0.5px;">OmniVoice</div>
            <div style="font-size:11px;color:#FF6B35;margin-top:1px;font-weight:500;letter-spacing:0.5px;text-transform:uppercase;">Voice Cloning & Design Studio</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:3px;height:32px;">
        <span class="ov-bar" style="height:8px;animation-delay:0s;"></span>
        <span class="ov-bar" style="height:18px;animation-delay:0.1s;"></span>
        <span class="ov-bar" style="height:26px;animation-delay:0.2s;"></span>
        <span class="ov-bar" style="height:14px;animation-delay:0.3s;"></span>
        <span class="ov-bar" style="height:22px;animation-delay:0.4s;"></span>
        <span class="ov-bar" style="height:10px;animation-delay:0.5s;"></span>
        <span class="ov-bar" style="height:20px;animation-delay:0.6s;"></span>
        <span class="ov-bar" style="height:12px;animation-delay:0.7s;"></span>
        <span class="ov-bar" style="height:24px;animation-delay:0.8s;"></span>
    </div>
    <div style="font-size:12px;display:flex;gap:12px;align-items:center;">
        <a href="https://www.facebook.com/iamyourshahzaib" style="
            color:#FFFFFF;text-decoration:none;
            background:rgba(255,255,255,0.08);
            border:1px solid rgba(255,255,255,0.15);
            padding:5px 12px;border-radius:20px;
            font-family:'Inter',sans-serif;font-weight:500;
            transition:all 0.2s;
        " target="_blank">Facebook</a>
        <a href="https://wa.me/923363854956" style="
            color:#FFFFFF;text-decoration:none;
            background:rgba(34,197,94,0.15);
            border:1px solid rgba(34,197,94,0.3);
            padding:5px 12px;border-radius:20px;
            font-family:'Inter',sans-serif;font-weight:500;
        " target="_blank">WhatsApp</a>
    </div>
</div>
"""

VC_TIPS_HTML = """
<div class="tip-box">
    <b style="color:#22C55E;font-size:11.5px;letter-spacing:1px;text-transform:uppercase;">💡 Studio Tips</b><br>
    ▸ Reference audio 3–10 sec best quality deta hai<br>
    ▸ 20 sec se zyada slow ho jata hai<br>
    ▸ Transcript blank → auto-transcribe hoga<br>
    ▸ Steps 20–30 = speed/quality sweet spot<br>
    ▸ Silence ON = cleaner narration output
</div>
"""

# ── Auto-download JS ──────────────────────────────────────────────────────
AUTO_DL_JS = """
<script>
// Auto-download: audio src ready hote hi download trigger karo
(function() {
    function triggerDownload(src, filename) {
        if (!src || src === window._lastAutoDownload) return;
        window._lastAutoDownload = src;
        var a = document.createElement('a');
        a.href = src;
        a.download = filename || 'omnivoice.wav';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function waitForAudioAndDownload(audioContainerId, filename) {
        // Audio element appear hone ka wait karo (max 10 sec)
        var attempts = 0;
        var interval = setInterval(function() {
            attempts++;
            var audioBox = document.getElementById(audioContainerId);
            if (audioBox) {
                var audio = audioBox.querySelector('audio');
                if (audio && audio.src && audio.src.startsWith('http')) {
                    clearInterval(interval);
                    // Audio load hone ka thoda wait karo phir download
                    audio.addEventListener('canplay', function() {
                        triggerDownload(audio.src, filename);
                    }, { once: true });
                    // Fallback — agar canplay already fire ho chuka
                    if (audio.readyState >= 3) {
                        triggerDownload(audio.src, filename);
                    }
                }
            }
            if (attempts > 100) clearInterval(interval); // 10 sec timeout
        }, 100);
    }

    function setupWatcher(triggerId, audioContainerId) {
        var triggerEl = document.getElementById(triggerId);
        if (!triggerEl) return;
        var textarea = triggerEl.querySelector('textarea');
        if (!textarea) return;
        var lastVal = '';
        var obs = new MutationObserver(function() {
            var val = textarea.value.trim();
            if (val && val.length > 3 && val !== lastVal) {
                lastVal = val;
                // val = filename (e.g. omnivoice_1234567890_cleaned.wav)
                waitForAudioAndDownload(audioContainerId, val);
            }
        });
        obs.observe(textarea, { attributes: true, childList: true, subtree: true, characterData: true });
    }

    function init() {
        setupWatcher('vc-dl-trigger', 'vc-audio-out');
        setupWatcher('vd-dl-trigger', 'vd-audio-out');
    }

    // DOM ready hone ka wait
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { setTimeout(init, 2000); });
    } else {
        setTimeout(init, 2000);
    }
})();
</script>
"""

with gr.Blocks(css=STUDIO_CSS, title="OmniVoice — Voice Cloning & Design") as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(AUTO_DL_JS)

    with gr.Tabs():

        # ── TAB 1: Voice Clone ────────────────────────────────────────────
        with gr.Tab("🎤  Voice Clone"):
            with gr.Row():
                with gr.Column(scale=3):
                    vc_text = gr.Textbox(
                        label="Text to speak",
                        placeholder="Yahan apna text paste karo...",
                        lines=12,
                        max_lines=40,
                    )
                    vc_ref_audio = gr.Audio(
                        label="Reference audio  (3–20 sec recommended)",
                        type="filepath",
                        format="wav",
                        sources=["upload", "microphone"],
                    )
                    vc_ref_transcript = gr.Textbox(
                        label="Reference transcript  (optional — blank = auto-detect)",
                        placeholder="Reference audio ka text likho ya blank chhoro...",
                        lines=2,
                    )
                    vc_steps = gr.Slider(
                        label="Generation steps",
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=2,
                        info="10–16 = fast · 20–24 = best balance · 32–50 = best quality",
                    )
                    vc_speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="1.0 = normal · >1.0 = faster · <1.0 = slower",
                    )
                    vc_remove_sil, vc_sil_thresh, vc_min_sil_ms = silence_controls()
                    vc_auto_dl = gr.Checkbox(
                        label="⬇️ Auto Download — voice bante hi automatic download ho jaye (default ON)",
                        value=True,
                        elem_id="vc-auto-dl",
                    )
                    vc_btn = gr.Button("✦  Generate Voice Clone", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML(VC_TIPS_HTML)
                    vc_status = gr.HTML(
                        value='<span style="color:#FF6B35;">● Ready — Generate dabao</span>',
                        elem_id="status-box",
                    )
                    vc_audio_out = gr.Audio(
                        label="Generated audio",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                        autoplay=True,
                        elem_id="vc-audio-out",
                    )
                    # Hidden textbox — auto download trigger ke liye JS use karta hai isko
                    vc_dl_trigger = gr.Textbox(
                        value="",
                        visible=False,
                        elem_id="vc-dl-trigger",
                    )
                    gr.HTML(f'<div class="save-path">📁 &nbsp;{OUTPUT_FOLDER}</div>')

            vc_btn.click(
                fn=generate_voice_clone,
                inputs=[vc_text, vc_ref_audio, vc_ref_transcript, vc_steps,
                        vc_speed, vc_remove_sil, vc_sil_thresh, vc_min_sil_ms, vc_auto_dl],
                outputs=[vc_audio_out, vc_dl_trigger, vc_status],
                concurrency_limit=1,
                show_progress="full",
            )

        # ── TAB 2: Voice Design ───────────────────────────────────────────
        with gr.Tab("🎨  Voice Design"):
            with gr.Row():
                with gr.Column(scale=3):
                    vd_text = gr.Textbox(
                        label="Text to speak",
                        placeholder="Yahan apna text paste karo...",
                        lines=10,
                        max_lines=40,
                    )
                    with gr.Row():
                        vd_gender = gr.Dropdown(
                            label="Gender",
                            choices=["Male", "Female"],
                            value="Male",
                        )
                        vd_age = gr.Dropdown(
                            label="Age range",
                            choices=["Young", "Middle-aged", "Old"],
                            value="Middle-aged",
                        )
                        vd_emotion = gr.Dropdown(
                            label="Emotion / tone",
                            choices=["Neutral", "Happy", "Sad", "Angry", "Excited", "Calm"],
                            value="Neutral",
                        )
                    vd_steps = gr.Slider(
                        label="Generation steps",
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=2,
                        info="10–16 = fast · 20–24 = best balance · 32–50 = best quality",
                    )
                    vd_speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="1.0 = normal · >1.0 = faster · <1.0 = slower",
                    )
                    vd_remove_sil, vd_sil_thresh, vd_min_sil_ms = silence_controls()
                    vd_auto_dl = gr.Checkbox(
                        label="⬇️ Auto Download — voice bante hi automatic download ho jaye (default ON)",
                        value=True,
                        elem_id="vd-auto-dl",
                    )
                    vd_btn = gr.Button("✦  Generate Voice Design", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML(VC_TIPS_HTML)
                    vd_status = gr.HTML(
                        value='<span style="color:#FF6B35;">● Ready — Generate dabao</span>',
                        elem_id="status-box",
                    )
                    vd_audio_out = gr.Audio(
                        label="Generated audio",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                        autoplay=True,
                        elem_id="vd-audio-out",
                    )
                    vd_dl_trigger = gr.Textbox(
                        value="",
                        visible=False,
                        elem_id="vd-dl-trigger",
                    )
                    gr.HTML(f'<div class="save-path">📁 &nbsp;{OUTPUT_FOLDER}</div>')

            vd_btn.click(
                fn=generate_voice_design,
                inputs=[vd_text, vd_gender, vd_age, vd_emotion, vd_steps,
                        vd_speed, vd_remove_sil, vd_sil_thresh, vd_min_sil_ms, vd_auto_dl],
                outputs=[vd_audio_out, vd_dl_trigger, vd_status],
                concurrency_limit=1,
            )


if __name__ == "__main__":
    demo.queue(max_size=1)  # Page freeze prevent — queue enable karo
    demo.launch(
        share=True,
        show_error=True,
        server_port=7860,
        max_file_size="50mb",
        allowed_paths=[OUTPUT_FOLDER, "/content"],
    )
