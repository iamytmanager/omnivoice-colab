import gradio as gr
import os
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path

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
):
    if not text.strip():
        raise gr.Error("❌ Text to Speak khali hai!")
    if ref_audio is None:
        raise gr.Error("❌ Reference audio upload karo!")

    status_msg = "⏳ Voice generate ho rahi hai..."
    yield None, status_msg

    try:
        # Generate with OmniVoice
        timestamp = int(time.time())
        out_filename = f"omnivoice_{timestamp}.wav"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)

        # OmniVoice correct API: model.generate() with ref_audio for cloning
        # num_step (not num_steps), ref_text (not reference_transcript)
        generate_kwargs = dict(
            text=text,
            ref_audio=ref_audio,
            num_step=steps,          # ← correct param name
            speed=speed_factor,
        )
        if ref_transcript and ref_transcript.strip():
            generate_kwargs['ref_text'] = ref_transcript.strip()
        # if ref_text omitted, Whisper auto-transcribes

        audio_list = tts.generate(**generate_kwargs)
        # Returns list of np.ndarray at 24000 Hz
        if not audio_list:
            raise gr.Error("❌ Audio generate nahi hui — model ne kuch return nahi kiya!")
        
        audio_np = audio_list[0]
        sf.write(out_path, audio_np, 24000)

        # Silence removal
        if remove_sil:
            status_msg = "✂️ Silence remove ho rahi hai..."
            yield None, status_msg
            final_path = remove_silence(
                out_path,
                silence_thresh_db=sil_thresh_db,
                min_silence_ms=min_sil_ms,
            )
        else:
            final_path = out_path

        # Audio already numpy array hai — seedha return karo (instant render)
        if final_path != out_path:
            # silence removal ne naya file banaya — us se read karo
            audio_data, sr = sf.read(final_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
        else:
            audio_data = audio_np.astype(np.float32)
            sr = 24000

        status_msg = f"✅ Done! Saved: {os.path.basename(final_path)}"
        yield (sr, audio_data.astype(np.float32)), status_msg

    except Exception as e:
        raise gr.Error(f"❌ Error: {str(e)}")


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
):
    if not text.strip():
        raise gr.Error("❌ Text to Speak khali hai!")

    status_msg = "⏳ Voice design ho rahi hai..."
    yield None, status_msg

    try:
        timestamp = int(time.time())
        out_filename = f"omnivoice_design_{timestamp}.wav"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)

        # Voice design: instruct string banao attributes se
        instruct_parts = [gender.lower(), age.lower()]
        if emotion.lower() != "neutral":
            instruct_parts.append(emotion.lower())
        instruct_str = ", ".join(instruct_parts)

        audio_list = tts.generate(
            text=text,
            instruct=instruct_str,
            num_step=steps,          # ← correct param name
            speed=speed_factor,
        )
        if not audio_list:
            raise gr.Error("❌ Audio generate nahi hui!")

        audio_np = audio_list[0]
        sf.write(out_path, audio_np, 24000)

        if remove_sil:
            status_msg = "✂️ Silence remove ho rahi hai..."
            yield None, status_msg
            final_path = remove_silence(
                out_path,
                silence_thresh_db=sil_thresh_db,
                min_silence_ms=min_sil_ms,
            )
        else:
            final_path = out_path

        if final_path != out_path:
            audio_data, sr = sf.read(final_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
        else:
            audio_data = audio_np.astype(np.float32)
            sr = 24000

        status_msg = f"✅ Done! Saved: {os.path.basename(final_path)}"
        yield (sr, audio_data.astype(np.float32)), status_msg

    except Exception as e:
        raise gr.Error(f"❌ Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Inter:wght@400;500&display=swap');

/* ── Base reset ─────────────────────────────────────────────────────────── */
body, .gradio-container, .main, footer {
    background: #080C12 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #C8D6E8 !important;
}
footer { display: none !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ── Hide Gradio header chrome ──────────────────────────────────────────── */
.app-header, .share-btn, .svelte-1f354aw { display: none !important; }

/* ── Tab navigation ─────────────────────────────────────────────────────── */
.tab-nav { 
    background: #0C1220 !important;
    border-bottom: 1px solid #1A2535 !important;
    padding: 10px 20px 0 !important;
    gap: 2px !important;
}
.tab-nav button {
    background: transparent !important;
    color: #4A6A8A !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    border-radius: 6px 6px 0 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    margin: 0 !important;
    transition: all 0.15s !important;
}
.tab-nav button.selected {
    background: #080C12 !important;
    color: #4DA6FF !important;
    border-color: #1A2535 !important;
    border-bottom: 1px solid #080C12 !important;
    margin-bottom: -1px !important;
}

/* ── Panels & forms ─────────────────────────────────────────────────────── */
.contain, .gap, .form, .block, .padded, .tabs, .tabitem {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    gap: 12px !important;
}

/* ── Labels ──────────────────────────────────────────────────────────────── */
label span, .label-wrap span {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: #2E5080 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Inputs & textareas ──────────────────────────────────────────────────── */
textarea, input[type='text'], input[type='number'] {
    background: #0E1520 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 8px !important;
    color: #B8CCDE !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12.5px !important;
    line-height: 1.7 !important;
    padding: 10px 12px !important;
    transition: border-color 0.15s !important;
}
textarea:focus, input:focus {
    border-color: #2A5FA0 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
input[type='range'] { accent-color: #3B82F6 !important; }
.wrap.svelte-1cl284s { color: #3B82F6 !important; font-weight: 600 !important; }

/* ── Dropdowns / selects ─────────────────────────────────────────────────── */
.wrap-inner, select, .multiselect {
    background: #0E1520 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 8px !important;
    color: #B8CCDE !important;
}

/* ── Checkboxes ──────────────────────────────────────────────────────────── */
input[type='checkbox'] { accent-color: #3B82F6 !important; }
.checkbox-wrap label { 
    color: #7AAAD8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
button.primary, .gr-button.primary, button[variant='primary'] {
    background: #1A4A8A !important;
    border: 1px solid #2A5FA0 !important;
    border-radius: 8px !important;
    color: #E8F4FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    padding: 12px 20px !important;
    transition: background 0.15s !important;
}
button.primary:hover { background: #2050A0 !important; }
button.secondary, .gr-button.secondary {
    background: #0A1018 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 8px !important;
    color: #4A6A8A !important;
    font-size: 12px !important;
    padding: 8px 14px !important;
}

/* ── Audio component ─────────────────────────────────────────────────────── */
.audio-container, .audio-wrap, [data-testid="audio"] {
    background: #0A1018 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
.audio-container audio, audio {
    width: 100% !important;
    height: 36px !important;
    border-radius: 6px !important;
    accent-color: #3B82F6 !important;
}
/* Remove default waveform, show clean player */
.waveform-wrap { background: #0C1525 !important; border-radius: 6px !important; }

/* ── File upload zone ────────────────────────────────────────────────────── */
.upload-btn, .wrap.svelte-r2cif8 {
    background: #0A1018 !important;
    border: 1px dashed #1E3550 !important;
    border-radius: 8px !important;
    color: #4A6A8A !important;
}
.upload-btn:hover { border-color: #3B82F6 !important; color: #4DA6FF !important; }

/* ── Accordion ───────────────────────────────────────────────────────────── */
.accordion {
    background: #0A1018 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 8px !important;
}
.accordion-header {
    color: #4A6A8A !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
}

/* ── Status / HTML boxes ─────────────────────────────────────────────────── */
#status-box {
    background: #0A1018 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    color: #3B82F6 !important;
    font-size: 12px !important;
    min-height: 36px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* ── Tip box ─────────────────────────────────────────────────────────────── */
.tip-box {
    background: #08100E !important;
    border: 1px solid #0E2A22 !important;
    border-left: 3px solid #0F6E56 !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 10px 12px !important;
    font-size: 11px !important;
    color: #3A6A58 !important;
    line-height: 1.8 !important;
}

/* ── Save path box ───────────────────────────────────────────────────────── */
.save-path {
    background: #0A1018 !important;
    border: 1px solid #1A2535 !important;
    border-radius: 6px !important;
    padding: 7px 12px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    color: #2E5080 !important;
}

/* ── Row layout ──────────────────────────────────────────────────────────── */
.row { gap: 16px !important; }

/* ── Waveform animation bars (in header HTML) ────────────────────────────── */
@keyframes ovWave {
    0%, 100% { transform: scaleY(1); opacity: 0.5; }
    50% { transform: scaleY(1.7); opacity: 1; }
}
.ov-bar {
    display: inline-block;
    width: 3px;
    background: #2A5F9E;
    border-radius: 2px;
    margin: 0 1px;
    vertical-align: middle;
    animation: ovWave 1.4s ease-in-out infinite;
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
    background:#0C1220;
    border-bottom:1px solid #1A2535;
    padding:14px 20px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin:-8px -8px 0 -8px;
">
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="
            width:36px;height:36px;
            background:#0E1E32;
            border:1px solid #2A4A6F;
            border-radius:8px;
            display:flex;align-items:center;justify-content:center;
        ">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <path d="M2 13 Q5 5 9 9 Q13 13 16 5" stroke="#4DA6FF" stroke-width="1.8" stroke-linecap="round" fill="none"/>
                <circle cx="9" cy="9" r="2" fill="#3B82F6"/>
            </svg>
        </div>
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:600;color:#E8F2FF;letter-spacing:-0.3px;">OmniVoice</div>
            <div style="font-size:11px;color:#2E5080;margin-top:1px;">Voice Cloning & Design Studio</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:3px;height:28px;">
        <span class="ov-bar" style="height:7px;animation-delay:0s;"></span>
        <span class="ov-bar" style="height:15px;animation-delay:0.1s;"></span>
        <span class="ov-bar" style="height:22px;animation-delay:0.2s;background:#3B82F6;"></span>
        <span class="ov-bar" style="height:13px;animation-delay:0.3s;"></span>
        <span class="ov-bar" style="height:19px;animation-delay:0.4s;background:#3B82F6;"></span>
        <span class="ov-bar" style="height:9px;animation-delay:0.5s;"></span>
        <span class="ov-bar" style="height:17px;animation-delay:0.6s;"></span>
        <span class="ov-bar" style="height:11px;animation-delay:0.7s;"></span>
    </div>
    <div style="font-size:11px;color:#1A3550;">
        <a href="https://www.facebook.com/iamyourshahzaib" style="color:#2A5080;text-decoration:none;" target="_blank">Facebook</a>
        &nbsp;·&nbsp;
        <a href="https://wa.me/923363854956" style="color:#2A5080;text-decoration:none;" target="_blank">WhatsApp</a>
    </div>
</div>
"""

VC_TIPS_HTML = """
<div class="tip-box">
    <b style="color:#0F9E7A;font-size:11px;">STUDIO TIPS</b><br>
    ▸ Reference audio 3–10 sec best quality deta hai<br>
    ▸ 20 sec se zyada slow ho jata hai<br>
    ▸ Transcript blank → auto-transcribe hoga<br>
    ▸ Steps 20–30 = speed/quality sweet spot<br>
    ▸ Silence ON = cleaner narration output
</div>
"""

with gr.Blocks(css=STUDIO_CSS, title="OmniVoice — Voice Cloning & Design") as demo:

    gr.HTML(HEADER_HTML)

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
                        value=16,
                        step=2,
                        info="10–16 = fast · 32–50 = best quality",
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
                    vc_btn = gr.Button("✦  Generate Voice Clone", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML(VC_TIPS_HTML)
                    vc_status = gr.HTML(
                        value='<span style="color:#2E5080;">● Ready — Generate dabao</span>',
                        elem_id="status-box",
                    )
                    vc_audio_out = gr.Audio(
                        label="Generated audio",
                        type="numpy",
                        interactive=False,
                        show_download_button=True,
                        autoplay=True,
                    )
                    gr.HTML(f'<div class="save-path">📁 &nbsp;{OUTPUT_FOLDER}</div>')

            vc_btn.click(
                fn=generate_voice_clone,
                inputs=[vc_text, vc_ref_audio, vc_ref_transcript, vc_steps,
                        vc_speed, vc_remove_sil, vc_sil_thresh, vc_min_sil_ms],
                outputs=[vc_audio_out, vc_status],
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
                        value=16,
                        step=2,
                        info="10–16 = fast · 32–50 = best quality",
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
                    vd_btn = gr.Button("✦  Generate Voice Design", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML(VC_TIPS_HTML)
                    vd_status = gr.HTML(
                        value='<span style="color:#2E5080;">● Ready — Generate dabao</span>',
                        elem_id="status-box",
                    )
                    vd_audio_out = gr.Audio(
                        label="Generated audio",
                        type="numpy",
                        interactive=False,
                        show_download_button=True,
                        autoplay=True,
                    )
                    gr.HTML(f'<div class="save-path">📁 &nbsp;{OUTPUT_FOLDER}</div>')

            vd_btn.click(
                fn=generate_voice_design,
                inputs=[vd_text, vd_gender, vd_age, vd_emotion, vd_steps,
                        vd_speed, vd_remove_sil, vd_sil_thresh, vd_min_sil_ms],
                outputs=[vd_audio_out, vd_status],
            )


if __name__ == "__main__":
    demo.launch(
        share=True,
        show_error=True,
        server_port=7860,
    )
