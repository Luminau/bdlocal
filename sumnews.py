from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def summarize(text, max_length=150):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)

    generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length, repetition_penalty=2.5,
                                   length_penalty=1.0, early_stopping=True).to(device)

    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds[0]


inputs = summarize(
    'After the sound and the fury, weeks of demonstrations and anguished calls for racial justice, the man whose death gave rise to an international movement, and whose last words — “I can’t breathe” — have been a rallying cry, will be laid to rest on Tuesday at a private funeral in Houston.George Floyd, who was 46, will then be buried in a grave next to his mother’s.The service, scheduled to begin at 11 a.m. at the Fountain of Praise church, comes after five days of public memorials in Minneapolis, North Carolina and Houston and two weeks after a Minneapolis police officer was caught on video pressing his knee into Mr. Floyd’s neck for nearly nine minutes before Mr. Floyd died. That officer, Derek Chauvin, has been charged with second-degree murder and second-degree manslaughter. His bail was set at $1.25 million in a court appearance on Monday. The outpouring of anger and outrage after Mr. Floyd’s death — and the speed at which protests spread from tense, chaotic demonstrations in the city where he died to an international movement from Rome to Rio de Janeiro — has reflected the depth of frustration borne of years of watching black people die at the hands of the police or vigilantes while calls for change went unmet.',
    80)

print(inputs)
