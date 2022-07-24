from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
emoModelT = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
emoModel = AutoModelForSequenceClassification.from_pretrained(emoModelT)
emotions = pipeline("text-classification", tokenizer=tokenizer, model=emoModel, return_all_scores=True, device=0)
emoResults = emotions("I am happy")
print(emoResults)
print("\n")
print(type(emoResults))
print("\n")
print()
