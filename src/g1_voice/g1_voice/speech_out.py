"""
Szöveg -> magyar beszéd (16 kHz, mono, 16 bites PCM, ahogy a robot hangszórója várja).

Ugyanaz a módszer, mint a common/audio/hu_tts.py-ban: Microsoft Edge felolvasó
(edge-tts, internet kell hozzá, API-kulcs nem), az MP3-at a miniaudio alakítja PCM-mé.
Ide másolva, hogy a mappa önállóan is működjön.

Hangosítás: a felolvasó hangja nem használja ki a teljes hangerőtartományt, ezért
a csúcsot a maximum közelére emeljük (normalizálás), és kérésre még erősítünk is
(gain_db). Az erősítés fölötti részt lágyan levágjuk, hogy ne recsegjen.

Magyar hangok: hu-HU-TamasNeural (férfi), hu-HU-NoemiNeural (nő).
"""

import asyncio

import edge_tts
import miniaudio
import numpy as np

SAMPLE_RATE = 16000


async def _mp3_bytes(text, voice):
    chunks = []
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def louder(pcm, gain_db=0.0):
    """Csúcsra normalizálás (-1 dBFS), majd opcionális erősítés lágy vágással."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-4:
        return pcm
    x = x * (0.89 / peak)                      # -1 dBFS
    if gain_db > 0:
        x = x * (10.0 ** (gain_db / 20.0))
        x = np.tanh(x)                         # lágy vágás: hangosabb, de nem recseg
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def synthesize_pcm(text, voice="hu-HU-TamasNeural", gain_db=0.0):
    mp3 = asyncio.run(_mp3_bytes(text, voice))
    if not mp3:
        raise RuntimeError("Az edge-tts nem adott vissza hangot.")
    decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                               nchannels=1, sample_rate=SAMPLE_RATE)
    return louder(bytes(decoded.samples), gain_db)


def beep_pcm(volume=0.5):
    """Rövid, emelkedő kéthangú jelzés: "figyelek, beszélhetsz"."""
    parts = []
    for freq in (880.0, 1320.0):
        t = np.arange(int(SAMPLE_RATE * 0.11)) / SAMPLE_RATE
        tone = np.sin(2 * np.pi * freq * t)
        fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)   # 10 ms be- és kiúsztatás
        parts.append(tone * fade)
        parts.append(np.zeros(int(SAMPLE_RATE * 0.02)))
    x = np.concatenate(parts) * volume
    return (x * 32767.0).astype(np.int16).tobytes()
