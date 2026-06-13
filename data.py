from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

MODEL_NAME = "camembert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    """Tokenize a batch of raw text reviews."""
    return tokenizer(
        batch["review"],
        padding="max_length",
        truncation=True,
        max_length=512
    )


def get_dataloaders(batch_size=32):
    """Return train / val / test DataLoaders for the Allociné dataset."""
    dataset = load_dataset("tblard/allocine")
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print("input_ids shape :", batch["input_ids"].shape)
    print("attention_mask shape :", batch["attention_mask"].shape)
    print("labels shape :", batch["label"].shape)