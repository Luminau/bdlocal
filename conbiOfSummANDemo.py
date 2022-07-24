from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from openpyxl import load_workbook
import time

start_time = time.time()

dataN = "performancetest.xlsx"
dataF = pd.read_excel(dataN)
newsBody = pd.read_excel(dataN, usecols=[0])

torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

sumModelT = "t5-base-finetuned-summarize-news"
sumTokenizer = AutoTokenizer.from_pretrained(sumModelT)
sumModel = AutoModelWithLMHead.from_pretrained(sumModelT)
sumModel = sumModel.to(device)

emoModelT = "bert-base-go-emotion"
emoTokenizer = AutoTokenizer.from_pretrained(emoModelT)
emoModel = AutoModelForSequenceClassification.from_pretrained(emoModelT)
emoModel = emoModel.to(device)

emotions = pipeline("text-classification", tokenizer=emoTokenizer, model=emoModel, return_all_scores=True, device=0)


def summarize(text, max_length=150):
    input_ids = sumTokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    generated_ids = sumModel.generate(input_ids=input_ids, num_beams=2, max_length=max_length, repetition_penalty=2.5,
                                      length_penalty=1.0, early_stopping=True).to(device)
    preds = [sumTokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds[0]


def chunkstring(string, length):
    pieces = str(string).split()
    return list(" ".join(pieces[chunki:chunki + length]) for chunki in range(0, len(pieces), length))
    # return list(string[0 + chunki:length + chunki] for chunki in range(0, len(string), length))


maxSingleNum = 320
summarizedInputList = []

for summi in newsBody.index:
    targetWordsNum = 65
    # print(type(newsBody.iloc[summi, 0]))
    print(summi)
    newsWordsNum = round(len(list(str(newsBody.iloc[summi, 0]).split())))
    if newsWordsNum > maxSingleNum:
        ScaleWordsNum = round(targetWordsNum / newsWordsNum * maxSingleNum*1.1)
    else:
        ScaleWordsNum = targetWordsNum
    # print(newsWordsNum, "\n")
    # print(ScaleWordsNum, "\n")

    if newsWordsNum > maxSingleNum:
        bigNews = []
        chunks = chunkstring(newsBody.iloc[summi, 0], maxSingleNum)
        # print(chunks[1])
        # print(len(chunks))
        for chunksNo in range(len(chunks)):
            bigNews.append(summarize(chunks[chunksNo], ScaleWordsNum))
        summarizedInputList.append(''.join(bigNews))

    elif newsWordsNum < targetWordsNum:
        summarizedInputList.append(newsBody.iloc[summi, 0])

    else:
        summarizedInputList.append(summarize(newsBody.iloc[summi, 0], ScaleWordsNum))

summarizedInput = pd.DataFrame(summarizedInputList, columns=["news-Summary"])
# inputs = summarize(it, 80)
# print(summarizedInput)
mid_time = time.time()
print("摘要生成用时")
print(mid_time - start_time)
print("秒")

book = load_workbook(dataN)
writer = pd.ExcelWriter(dataN, engine="openpyxl")
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
# writer.sheets = dict((ws.title))
# summarizedInput.to_excel(writer, sheet_name='Sheet1', index=False, startcol=1)
# writer.save()

emoInputList = []
for emoi in summarizedInput.index:
    emoInputList.append(emotions(str(summarizedInput.iloc[emoi, 0])))

column_name = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
               "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
               "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride",
               "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
coln = pd.DataFrame(columns=column_name)
# print(coln)
for emoOuti in range(len(emoInputList)):
    print(emoOuti)
    emoInput = pd.DataFrame(emoInputList[emoOuti])
    # emoInput = emoInput["score"]
    emoInput = emoInput.transpose()
    emoInput = emoInput.iloc[1:2, :]
    # print(emoInput)
    emoInput.to_excel(writer, sheet_name="Sheet1", header=False, index=False, startcol=2, startrow=emoOuti + 1)

writer.save()
time_end = time.time()
print("运行实际时间为")
print(time_end - start_time)
print("秒")
