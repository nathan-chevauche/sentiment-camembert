import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "./best_model"

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}


class SentimentPredictor:

    def __init__(self, model_path: str = MODEL_PATH, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading model on {self.device}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        return {
            "label": LABEL_MAP[pred_idx],
            "score": round(probs[pred_idx].item(), 4),
            "probabilities": {
                LABEL_MAP[i]: round(p.item(), 4) for i, p in enumerate(probs)
            },
        }


if __name__ == "__main__":
    predictor = SentimentPredictor()

    samples = [
        "Ce film est absolument magnifique, je recommande vivement !",
        "Vraiment décevant, je me suis ennuyé du début à la fin.",
        "Une belle surprise, des acteurs talentueux et une histoire touchante.",
        "Nul à chier, perte de temps et d'argent.",
    ]

    for text in samples:
        result = predictor.predict(text)
        print(f"[{result['label']} {result['score']:.2%}] {text}")
