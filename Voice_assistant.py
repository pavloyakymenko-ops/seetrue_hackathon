# voice_assistant.py
"""
Voice assistant for gaze-aware system.
Press 'v' in the video window to talk — the assistant sees what you're
looking at and answers through the laptop speaker.

Requires:
    pip install SpeechRecognition pyttsx3 pyaudio anthropic

Uses:
    • SpeechRecognition + Google free STT  (mic → text)
    • Anthropic Claude API               (text → answer, with gaze context)
    • pyttsx3                            (answer → speaker)
"""

import threading
import time
import queue

try:
    import speech_recognition as sr
    _HAS_SR = True
except ImportError:
    _HAS_SR = False
    print("[Voice] WARNING: SpeechRecognition not installed. pip install SpeechRecognition")

try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False
    print("[Voice] WARNING: pyttsx3 not installed. pip install pyttsx3")

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False
    print("[Voice] WARNING: anthropic not installed. pip install anthropic")


# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = "sk-ant-api03-IiM...xwAA"          # ← paste your key here, or set ANTHROPIC_API_KEY env var
LISTEN_TIMEOUT   = 5           # seconds to wait for speech
PHRASE_LIMIT     = 10          # max seconds of speech per utterance
TTS_RATE         = 175         # words per minute for TTS


SYSTEM_PROMPT = """\
You are a smart assistant embedded in eye-tracking glasses.  The user is
looking at the real world through a scene camera.  You receive a description
of what they are currently gazing at (detected objects, recognised people,
and cognitive load level).

Rules:
• Answer in 1-3 short sentences — the response will be spoken aloud.
• Be conversational and concise, like a helpful friend.
• If gaze context is provided, weave it into your answer naturally.
• If you don't know, say so briefly.
"""


class VoiceAssistant:
    """
    Threaded voice assistant.  Call .trigger() from the main loop when
    the user presses a key.  It records, transcribes, queries Claude,
    and speaks the answer — all off the main thread.
    """

    def __init__(self):
        self._busy = False
        self._status = "idle"          # "idle" | "listening" | "thinking" | "speaking"
        self._status_text = ""         # short string to overlay on the frame
        self._last_answer = ""

        # STT
        if _HAS_SR:
            self._recognizer = sr.Recognizer()
            self._recognizer.dynamic_energy_threshold = True
            self._mic = sr.Microphone()
            # Quick ambient noise calibration
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[Voice] Microphone ready.")
        else:
            self._recognizer = None
            self._mic = None

        # TTS
        if _HAS_TTS:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", TTS_RATE)
            print("[Voice] TTS engine ready.")
        else:
            self._tts_engine = None

        # LLM client
        if _HAS_ANTHROPIC:
            import os
            key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                self._client = anthropic.Anthropic(api_key=key)
                print("[Voice] Anthropic client ready.")
            else:
                self._client = None
                print("[Voice] WARNING: No ANTHROPIC_API_KEY set. LLM disabled.")
        else:
            self._client = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def is_busy(self) -> bool:
        return self._busy

    def trigger(self, gaze_context: dict):
        """
        Start a voice interaction.  Non-blocking — runs in a background thread.

        gaze_context should look like:
            {
                "objects":  ["MacBook laptop 92%", "iPhone 87%"],
                "person":   "Faisal (Student, 22y)" or None,
                "cog_load": "focused (52%)",
            }
        """
        if self._busy:
            print("[Voice] Already processing, ignoring trigger.")
            return
        if not self._recognizer:
            print("[Voice] STT not available.")
            return

        self._busy = True
        t = threading.Thread(target=self._pipeline, args=(gaze_context,), daemon=True)
        t.start()

    # ── Pipeline (runs in background thread) ──────────────────────────────────

    def _pipeline(self, gaze_context: dict):
        try:
            # 1. Listen
            self._status = "listening"
            self._status_text = "Listening..."
            print("[Voice] Listening...")
            user_text = self._listen()
            if not user_text:
                self._status_text = "Didn't catch that"
                print("[Voice] No speech detected.")
                time.sleep(1.5)
                return

            print(f"[Voice] Heard: {user_text}")
            self._status_text = f'"{user_text}"'

            # 2. Think
            self._status = "thinking"
            self._status_text = "Thinking..."
            print("[Voice] Querying LLM...")
            answer = self._ask_llm(user_text, gaze_context)
            print(f"[Voice] Answer: {answer}")
            self._last_answer = answer

            # 3. Speak
            self._status = "speaking"
            self._status_text = answer[:60] + ("..." if len(answer) > 60 else "")
            self._speak(answer)

        except Exception as e:
            print(f"[Voice] Pipeline error: {e}")
            self._status_text = f"Error: {e}"
            time.sleep(2)
        finally:
            self._busy = False
            self._status = "idle"
            self._status_text = ""

    # ── STT ───────────────────────────────────────────────────────────────────

    def _listen(self) -> str:
        try:
            with self._mic as source:
                audio = self._recognizer.listen(
                    source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_LIMIT
                )
            text = self._recognizer.recognize_google(audio)
            return text.strip()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"[Voice] Google STT error: {e}")
            return ""

    # ── LLM ───────────────────────────────────────────────────────────────────

    def _ask_llm(self, user_text: str, gaze_context: dict) -> str:
        # Build context string
        ctx_parts = []
        if gaze_context.get("objects"):
            ctx_parts.append("Objects in view: " + ", ".join(gaze_context["objects"]))
        if gaze_context.get("person"):
            ctx_parts.append("Person being looked at: " + gaze_context["person"])
        if gaze_context.get("cog_load"):
            ctx_parts.append("User's cognitive load: " + gaze_context["cog_load"])

        context_str = "\n".join(ctx_parts) if ctx_parts else "No objects currently detected."

        full_message = f"[GAZE CONTEXT]\n{context_str}\n\n[USER QUESTION]\n{user_text}"

        if self._client:
            try:
                response = self._client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": full_message}],
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"[Voice] Anthropic API error: {e}")
                return f"Sorry, I couldn't process that: {e}"
        else:
            # Fallback: no LLM available, echo context
            return f"I can see: {context_str}. But I need an API key to answer questions."

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        if self._tts_engine:
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        else:
            print(f"[Voice] (would speak): {text}")
            time.sleep(2)


def draw_voice_status(frame, assistant: VoiceAssistant):
    """Draw voice assistant status overlay on the bottom-left of the frame."""
    if assistant.status == "idle":
        return

    status = assistant.status
    text   = assistant.status_text

    # Colours per state
    colors = {
        "listening": (0, 180, 255),   # orange
        "thinking":  (255, 200, 0),   # cyan
        "speaking":  (0, 220, 100),   # green
    }
    color = colors.get(status, (200, 200, 200))

    h, w = frame.shape[:2]
    y = h - 30

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, y - 18), (w - 5, y + 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Status icon
    icon = {"listening": "MIC", "thinking": "...", "speaking": ">>"}
    label = f"[{icon.get(status, '?')}] {text}"

    cv2.putText(frame, label, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)