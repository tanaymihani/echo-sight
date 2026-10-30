"""Speech output (streamed sentence by sentence) and push-to-talk speech input."""

import queue
import re
import shutil
import subprocess
import threading
from collections.abc import Callable

from echosight.rag import strip_citations

_SENTENCE = re.compile(r"(.+?[.!?])(?:\s+|$)", re.S)


class SentenceStream:
    """Collects streamed LLM tokens and hands complete sentences, minus citation markers, to `emit`."""

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.buffer = ""

    def feed(self, delta: str) -> None:
        self.buffer += delta
        while True:
            match = _SENTENCE.match(self.buffer)
            # Wait for the character after the period so "[2]" and "e.g." stay with their sentence.
            if not match or match.end() == len(self.buffer):
                return
            sentence, self.buffer = match.group(1), self.buffer[match.end():]
            if self.buffer.lstrip().startswith("["):
                marker_end = self.buffer.find("]")
                if marker_end < 0:
                    self.buffer = sentence + " " + self.buffer
                    return
                self.buffer = self.buffer[marker_end + 1:]
            self._emit(sentence)

    def flush(self) -> None:
        if self.buffer.strip():
            self._emit(self.buffer)
        self.buffer = ""

    def _emit(self, text: str) -> None:
        text = strip_citations(text)
        if text:
            self.emit(text)


class Speaker:
    """Speaks queued text on a background thread so audio starts while the answer is still streaming."""

    def __init__(self, rate: int = 190, enabled: bool = True):
        self.rate = rate
        self.enabled = enabled
        self.queue: queue.Queue[str] = queue.Queue()
        self._say = shutil.which("say")      # macOS built-in voice
        self._process = None
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str) -> None:
        if self.enabled and text.strip():
            self.queue.put(text)

    def wait(self) -> None:
        if self.enabled:
            self.queue.join()

    def stop(self) -> None:
        """Silence current speech and drop anything queued."""
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def _run(self) -> None:
        engine = None
        if not self._say:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        while True:
            text = self.queue.get()
            try:
                if engine:
                    engine.say(text)
                    engine.runAndWait()
                else:
                    self._process = subprocess.Popen([self._say, "-r", str(self.rate), text])
                    self._process.wait()
            finally:
                self.queue.task_done()


class Listener:
    """Push-to-talk speech recognition: press Enter to start and again to stop, then transcribe locally."""

    SAMPLE_RATE = 16_000

    def __init__(self, model: str = "base.en"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8")

    def listen(self) -> str:
        import numpy as np
        import sounddevice as sd

        frames = []
        input("Press Enter, then speak your question. Press Enter again when you are done.")
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype="float32",
                            callback=lambda data, *_: frames.append(data.copy())):
            input("Listening...")
        if not frames:
            return ""
        audio = np.concatenate(frames)[:, 0]
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
