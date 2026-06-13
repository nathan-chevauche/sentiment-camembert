import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(text: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best saved model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./best_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("./best_model")

    model.eval()

    # Tokenize input text
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs  = torch.softmax(outputs.logits, dim=1)
    label  = probs.argmax(1).item()
    score  = probs[0][label].item()

    return {
        "label": "POSITIVE" if label == 1 else "NEGATIVE",
        "score": round(score, 4)
    }


if __name__ == "__main__":
    samples = [
        "Ce film est absolument magnifique, je recommande vivement !",
        "Vraiment décevant, je me suis ennuyé du début à la fin.",
        "Une belle surprise, des acteurs talentueux et une histoire touchante.",
        "Nul à chier, perte de temps et d'argent."
    ]

    for text in samples:
        result = predict(text)
        print(f"[{result['label']} {result['score']:.2%}] {text}")