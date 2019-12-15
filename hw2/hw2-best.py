import numpy as np
import pandas as pd
import csv
import sys
import tensorflow as tf
from sklearn import preprocessing
import keras
from tensorflow.keras.models import load_model

def normalize_data(x, normalize_column):
    x = x.astype(np.float64)
    tmp = x[:, normalize_column]
    tmp = preprocessing.scale(tmp, axis=0)
    x[:, normalize_column] = tmp

    return x

x_test = pd.read_csv(sys.argv[5])
x_test = x_test.values

normalize_column = [0,1,3,4,5]
x_test = normalize_data(x_test, normalize_column)

# 載入模型
model = load_model('keras-model.h5')

prediction = model.predict(x_test, batch_size=100000)
prediction = np.around(prediction[:, 1])
ans_file = open(sys.argv[6], "w")
writer = csv.writer(ans_file)
title = ['id','label']
writer.writerow(title) 
for i in range(len(x_test)):
    content = [i+1,int(prediction[i])]
    writer.writerow(content)