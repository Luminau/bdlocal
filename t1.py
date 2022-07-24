import os
# import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import xlwt
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

f1=xlrd.open_workbook("data.xls")
sheet=f1.sheet_by_index(0)
rows=sheet.nrows
columns=sheet.ncols

# 训练集：2007年1~12月的所有数据
train_start=17520
train_end=35040
train_data=[[] for i in range(train_start,train_end)]
# 测试集：2008年6~8月的所有数据，测试误差
test_start=42336
test_end=46752
test_data=[[] for i in range(test_start,test_end)]
# 未来的真实数据：2009年1~2月的所有数据，泛化误差
true_start=54096
true_end=56928
true_data=[[] for i in range(true_start,true_end)]

# 获取训练集
for i in range(train_start, train_end):
    train_data[i-train_start] = sheet.row_values(i)[0:columns-1] # 获取电价（包括）之前的特征
    for j in range(i-144,i):
        train_data[i-train_start].append(sheet.row_values(j)[8]) # 获取前72小时的负荷数据
    train_data[i-train_start].append(sheet.row_values(i)[8]) #添加输出值

# # 获取测试集
# for i in range(test_start, test_end):
#     test_data[i-test_start] = sheet.row_values(i)[0:columns-1] # 获取电价（包括）之前的特征
#     for j in range(i-144,i):
#         test_data[i-test_start].append(sheet.row_values(j)[8]) # 获取前72小时的负荷数据
#     test_data[i-test_start].append(sheet.row_values(i)[8]) #添加输出值
#
# # 获取未来的真实数据
# for i in range(true_start, true_end):
#     true_data[i-true_start] = sheet.row_values(i)[0:columns-1] # 获取电价（包括）之前的特征
#     for j in range(i-144,i):
#         true_data[i-true_start].append(sheet.row_values(j)[8]) # 获取前72小时的负荷数据
#     true_data[i-true_start].append(sheet.row_values(i)[8]) #添加输出值


# workbook1 = xlwt.Workbook()
# sheet = workbook1.add_sheet("Sheet")
# for i in range(len(train_data)):
#     for j in range(len(train_data[i])):
#         sheet.write(i, j, train_data[i][j])
# workbook1.save("train.xls")
#
# workbook2 = xlwt.Workbook()
# sheet = workbook2.add_sheet("Sheet")
# for i in range(len(test_data)):
#     for j in range(len(test_data[i])):
#         sheet.write(i, j, test_data[i][j])
# workbook2.save("test.xls")
#
# workbook3 = xlwt.Workbook()
# sheet = workbook3.add_sheet("Sheet")
# for i in range(len(true_data)):
#     for j in range(len(true_data[i])):
#         sheet.write(i, j, true_data[i][j])
# workbook3.save("true.xls")

x=train_data[:][:-1]
y=train_data[:][-1]
model=GradientBoostingRegressor(n_estimators=100,loss='squared_error',max_depth=4,verbose=1,warm_start=True)
model.fit(x,y)
y_pred=model.predict(x)
print(metrics.accuracy_score(y,y_pred))


