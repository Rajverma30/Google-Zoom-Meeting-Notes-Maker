"""
Zoom auto-recorder with VB-Audio Virtual Cable (system audio capture).

Flow:
1) Wait for Zoom meeting window (pygetwindow).
2) Start recording system audio from VB-Audio Cable Output.
3) Optional: also record microphone, mix to mono.
4) Stop when meeting window disappears for a grace period.
5) Save WAV (16k mono) + Whisper large-v3 transcription to TXT.

Requirements:
- Windows, VB-Audio Virtual Cable installed.
- Zoom speaker set to: CABLE Input (VB-Audio Virtual Cable).
"""

import os
import time
import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import pygetwindow as gw
import sounddevice as sd
import wavio
import whisper
import msvcrt

# ================== CONFIG ==================
FS = 16_000
CHANNELS = 1
MODEL_SIZE = "large-v3"  # Whisper model size
LANGUAGE = "en"  # change to "hi" for Hinglish/Hindi, or None for auto-detect
OUTPUT_DIR = "zoom_meetings"

# Device selection
VB_CABLE_KEYWORDS = ["cable output", "vb-audio"]  # searched in device name (lower)
MIX_MIC = True               # True: mix mic + system audio; False: system audio only
MIC_DEVICE_KEYWORDS = None   # e.g., ["microphone", "usb"] or None for default input

# Meeting handling
POLL_SEC = 1
# Meeting handling
POLL_SEC = 1
# Start: when a Zoom meeting window appears.
# Stop: when meeting window is gone for this many seconds.
NO_MEETING_GRACE_SEC = 6

# Manual override:
# Press "p" anytime to stop recording + save + transcribe (if auto stop fails).
MANUAL_STOP_KEY = "p"

# Audio buffering
BLOCK_DURATION_SEC = 0.5   # block size for callbacks
# ============================================


# -------- Device helpers --------
def list_audio_devices() -> None:
    """Print available audio devices for debugging."""
    print("---- Available audio devices ----")
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        print(f"[{idx}] {d['name']} (in:{d['max_input_channels']} out:{d['max_output_channels']})")
    print("---- End of list ----")


def _match_device(keywords: List[str], require_input: bool = True) -> Optional[int]:
    """Return first device index whose name contains all keywords (case-insensitive)."""
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        name = d["name"].lower()
        if require_input and d["max_input_channels"] < 1:
            continue
        if all(k in name for k in keywords):
            return idx
    return None


def find_vb_cable_device() -> int:
    """Locate VB-Audio Cable output device; raise if not found."""
    device_idx = _match_device([k.lower() for k in VB_CABLE_KEYWORDS], require_input=True)
    if device_idx is None:
        list_audio_devices()
        raise RuntimeError(
            "VB-Audio Cable input device not found. Ensure 'CABLE Output (VB-Audio Virtual Cable)' exists "
            "and Zoom speaker is set to 'CABLE Input'."
        )
    return device_idx


def find_mic_device() -> Optional[int]:
    """Find mic device by keywords or use default input (None)."""
    if MIC_DEVICE_KEYWORDS:
        device_idx = _match_device([k.lower() for k in MIC_DEVICE_KEYWORDS], require_input=True)
        if device_idx is None:
            list_audio_devices()
            raise RuntimeError("Microphone device with given keywords not found.")
        return device_idx
    return None  # None uses default input


def _wasapi_loopback_setting(device_idx: int) -> Optional[sd.WasapiSettings]:
    """Enable WASAPI loopback if the device host API is WASAPI (Windows)."""
    info = sd.query_devices(device_idx)
    hostapi_idx = info["hostapi"]
    hostapi_name = sd.query_hostapis()[hostapi_idx]["name"].lower()
    if "wasapi" in hostapi_name:
        return sd.WasapiSettings(loopback=False)  # VB Cable is already capture side
    return None


# -------- Zoom detection (window-based only) --------
def zoom_meeting_titles() -> List[str]:
    """Return titles that look like Zoom meeting windows."""
    titles = []
    browser_markers = ["chrome", "brave", "edge", "firefox", "safari", "opera"]
    for title in gw.getAllTitles():
        if not title:
            continue
        low = title.lower()
        if "zoom meeting" not in low:
            continue
        if any(b in low for b in browser_markers):
            # Ignore browser tabs/windows with "Zoom Meeting" in title.
            continue
        titles.append(title)
    return titles


def is_zoom_meeting_window_open() -> bool:
    """Detect ONLY actual Zoom meeting window (title contains 'Zoom Meeting')."""
    return bool(zoom_meeting_titles())


# -------- Manual key handling (Windows console) --------
def manual_stop_pressed() -> bool:
    """
    Non-blocking check for manual stop key.
    Uses msvcrt so it works in a normal Windows terminal/console.
    """
    try:
        while msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                key = ch.decode("utf-8", errors="ignore").lower()
            except Exception:
                key = ""
            if key == MANUAL_STOP_KEY.lower():
                return True
        return False
    except Exception:
        # If the terminal doesn't support key polling, just disable manual stop.
        return False


# -------- Recording helpers --------
def build_stream(
    device: Optional[int],
    frames_store: List[np.ndarray],
    name: str,
    loopback: bool = False,
) -> sd.InputStream:
    """Create a configured InputStream."""
    extra = None
    if loopback:
        extra = sd.WasapiSettings(loopback=True)
    else:
        extra = _wasapi_loopback_setting(device) if device is not None else None

    return sd.InputStream(
        device=device,
        samplerate=FS,
        channels=CHANNELS,
        blocksize=int(BLOCK_DURATION_SEC * FS),
        dtype="float32",
        callback=lambda indata, frames, time_info, status: _frame_cb(
            indata, status, frames_store, name
        ),
        extra_settings=extra,
    )


def _frame_cb(indata, status, store: List[np.ndarray], name: str) -> None:
    if status:
        print(f"[{name}] {status}")
    store.append(indata.copy())


def mix_or_pick(system_audio: Optional[np.ndarray], mic_audio: Optional[np.ndarray]) -> np.ndarray:
    """Mix mic with system or return available track; always mono float32."""
    if system_audio is None and mic_audio is None:
        return np.array([], dtype=np.float32)
    if not MIX_MIC:
        return system_audio if system_audio is not None else mic_audio

    if system_audio is None:
        return mic_audio
    if mic_audio is None:
        return system_audio

    # Pad shorter to match length
    max_len = max(len(system_audio), len(mic_audio))
    def _pad(x): return np.pad(x, ((0, max_len - len(x)), (0, 0)), mode="constant")
    sys_p = _pad(system_audio)
    mic_p = _pad(mic_audio)
    mixed = (sys_p + mic_p) * 0.5
    return mixed.astype(np.float32)


def concat_frames(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    if not frames:
        return None
    return np.concatenate(frames, axis=0)


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------- Main flow --------
def main() -> None:
    print("---- Zoom Auto Recorder (VB-Audio Cable) ----")
    list_audio_devices()

    while True:
        # Wait for meeting window to appear
        print("👀 Waiting for Zoom meeting window...")
        while not is_zoom_meeting_window_open():
            # Debug: show candidate titles occasionally
            candidates = zoom_meeting_titles()
            if candidates:
                print(f"ℹ️ Detected meeting-like titles (will start if persist): {candidates}")
            time.sleep(POLL_SEC)

        detected_titles = zoom_meeting_titles()
        print(f"✅ Meeting window detected with titles: {detected_titles}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_output_dir()
        audio_path = os.path.join(OUTPUT_DIR, f"zoom_{timestamp}.wav")
        text_path = os.path.join(OUTPUT_DIR, f"zoom_{timestamp}.txt")

        # Resolve devices
        system_device = find_vb_cable_device()
        mic_device = find_mic_device() if MIX_MIC else None

        print(f"🚀 Zoom meeting detected! Recording started...\n"
              f"   Audio: {audio_path}\n   Text : {text_path}\n"
              f"   System device: {system_device}, Mic: {mic_device if MIX_MIC else 'disabled'}")
        print(f"⌨️ Manual stop: press '{MANUAL_STOP_KEY}' to stop recording anytime.")

        sys_frames: List[np.ndarray] = []
        mic_frames: List[np.ndarray] = []

        streams = []
        streams.append(build_stream(system_device, sys_frames, "system", loopback=False))
        if MIX_MIC:
            streams.append(build_stream(mic_device, mic_frames, "mic", loopback=False))

        for s in streams:
            s.start()

        no_meeting_secs = 0
        try:
            while True:
                if manual_stop_pressed():
                    print("🟡 Manual stop pressed. Stopping...")
                    break

                if is_zoom_meeting_window_open():
                    no_meeting_secs = 0
                else:
                    no_meeting_secs += POLL_SEC

                if no_meeting_secs >= NO_MEETING_GRACE_SEC:
                    print("🛑 Meeting window gone. Stopping recording...")
                    break

                time.sleep(POLL_SEC)
        finally:
            for s in streams:
                s.stop()
                s.close()

        system_audio = concat_frames(sys_frames)
        mic_audio = concat_frames(mic_frames) if MIX_MIC else None
        final_audio = mix_or_pick(system_audio, mic_audio)

        if final_audio.size == 0:
            print("⚠️ No audio captured; skipping transcription.")
        else:
            wavio.write(audio_path, final_audio, FS, sampwidth=2)
            print(f"✅ Audio saved as {audio_path}")

            print("🧠 Loading Whisper model...")
            model = whisper.load_model(MODEL_SIZE)

            print("✍️ Transcribing...")
            result = model.transcribe(
                audio_path,
                language=LANGUAGE,
                task="transcribe",
                temperature=0.0,
                beam_size=5,
                best_of=5,
                verbose=False,
            )

            with open(text_path, "w", encoding="utf-8") as f:
                f.write(result.get("text", ""))

            print(f"📝 Transcription saved as {text_path}")

        print("♻️ Recorder idle. It will auto-start on the next Zoom meeting window.")


if __name__ == "__main__":
    main()
