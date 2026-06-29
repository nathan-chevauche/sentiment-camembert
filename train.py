import mlflow
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from data import get_dataloaders, MODEL_NAME


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=num_epochs * len(train_loader)   # linear decay over all training steps
    )

    mlflow.set_experiment("sentiment-camembert")

    with mlflow.start_run():

        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("dataset", "tblard/allocine")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("lr_scheduler", "LinearLR")
        mlflow.log_param("max_seq_length", 512)

        best_val_loss = float("inf")

        for epoch in range(num_epochs):

            # Training
            model.train()
            train_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                optimizer.zero_grad()
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += outputs.loss.item()

            # Validation
            model.eval()
            val_loss = 0
            correct  = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                    val_loss += outputs.loss.item()
                    correct  += (outputs.logits.argmax(1) == labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss   = val_loss   / len(val_loader)
            val_accuracy   = correct    / len(val_loader.dataset)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.2%}",
                flush=True
            )

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                model.save_pretrained("./best_model")
                tokenizer.save_pretrained("./best_model")
                print(f"Best model saved (val_loss={best_val_loss:.4f})", flush=True)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_artifacts("./best_model", artifact_path="best_model")


if __name__ == "__main__":
    train()
