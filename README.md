# French Sentiment Analysis — CamemBERT fine-tuned on Allociné

Binary sentiment classifier (positive / negative) for French movie reviews,
built by fine-tuning CamemBERT on the Allociné dataset.

## Results

| Metric   | Score |
|----------|-------|
| Accuracy | --    |
| F1-score | --    |

*Evaluated on the Allociné test set (20 000 reviews).*

## Dataset

[Allociné](https://huggingface.co/datasets/tblard/allocine) — 200 000 French movie reviews scraped from Allociné.fr.

| Split      | Size    |
|------------|---------|
| Train      | 160 000 |
| Validation |  20 000 |
| Test       |  20 000 |

Labels: `0` = negative, `1` = positive.

## Model

- **Base model:** [`camembert-base`](https://huggingface.co/camembert/camembert-base)
- **Architecture:** CamemBERT encoder + linear classification head (768 → 2)
- **Fine-tuning:** 3 epochs, AdamW (lr=2e-5), linear lr decay

## Project Structure

```
sentiment-camembert/
├── data.py          # dataset loading and tokenization
├── train.py         # fine-tuning pipeline
├── evaluate.py      # metrics on test set
├── predict.py       # inference on raw text
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/your-username/sentiment-camembert
cd sentiment-camembert
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python train.py
```

**Evaluate on test set:**
```bash
python evaluate.py
```

**Predict:**
```python
from predict import predict

result = predict("Ce film est absolument magnifique !")
print(result)  # {'label': 'POSITIVE', 'score': 0.9987}
```

## License

MIT