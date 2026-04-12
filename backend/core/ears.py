from faster_whisper import WhisperModel
import os


class EarsManager:
    def __init__(self, model_size: str = "base.en"):  # UPGRADED from tiny.en
        # base.en is ~140MB - still very safe for 4GB RAM
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root="./models",  # Keep models local for Day 20 Demo
        )

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return ""

        # beam_size=5 tells the model to check the 5 most likely word paths
        # instead of just the 1st one, drastically improving accuracy.
        segments, info = self.model.transcribe(  # type: ignore
            audio_path,
            beam_size=5,
            word_timestamps=False,  # Disable for speed on 4GB RAM
        )

        text = "".join([segment.text for segment in segments])
        return text.strip()


aegis_ears = EarsManager()
