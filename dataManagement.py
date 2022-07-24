import pandas as pd
from openpyxl import load_workbook

dataN = "CNNdata.xlsx"
dataF = pd.read_excel(dataN)
newsBody = pd.read_excel(dataN, usecols=[0])
# print(newsBody)
# print(dataF.iloc[1,0])
summarizedInputList = []
for i in newsBody.index:
    print(newsBody.iloc[i, 0])
    summarizedInputList.append(newsBody.iloc[i, 0])

summarizedInput = pd.DataFrame(summarizedInputList, columns=["news-Summary"])
# print(summarizedInput)

book = load_workbook(dataN)
writer = pd.ExcelWriter(dataN, engine="openpyxl")
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
# writer.sheets = dict((ws.title))
# summarizedInput.to_excel(writer, sheet_name='Sheet1', index=False, startcol=1)
column_name = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
               "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
               "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride",
               "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
coln = pd.DataFrame(columns=column_name)
coln.to_excel(writer)
print(coln)
writer.save()

