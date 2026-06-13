import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from data import get_dataloaders


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best saved model and tokenizer
    model     = AutoModelForSequenceClassification.from_pretrained("./best_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("./best_model")

    _, _, test_loader = get_dataloaders(batch_size=64)

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs.logits.argmax(1)

            # Move to CPU and store for metric computation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Accuracy : {accuracy:.2%}")
    print(f"Test F1-score : {f1:.4f}")


if __name__ == "__main__":
    evaluate()