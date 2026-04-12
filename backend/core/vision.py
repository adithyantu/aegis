import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class VisionManager:
    def __init__(self):
        print("Waking up the Vision Engine (Moondream2)...")
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"

        # We initialize these as None so we can check them later
        self.tokenizer = None
        self.model = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)  # type: ignore
            self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                model_id, trust_remote_code=True, revision=revision
            )
            self.model.eval()
            print("Vision Engine Online.")
        except Exception as e:
            # THIS is where the real error was hiding!
            print(f"CRITICAL: Failed to load Vision Engine: {e}")

    def analyze(self, image_path: str, prompt: str) -> str:
        # DEFENSIVE CHECK: Did the model actually load?
        if self.model is None or self.tokenizer is None:  # type: ignore
            return "Error: Vision Engine is offline. Check backend startup logs for missing dependencies."

        if not os.path.exists(image_path):
            return "Error: Image file not found."

        try:
            image = Image.open(image_path).convert("RGB")
            enc_image = self.model.encode_image(image)  # type: ignore
            answer = self.model.answer_question(enc_image, prompt, self.tokenizer)  # type: ignore
            return answer.strip()  # type: ignore
        except Exception as e:
            return f"Vision processing failed: {str(e)}"


aegis_eyes = VisionManager()
