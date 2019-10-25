import pandas as pd
import numpy as np
import csv
import sys
import math
dim = 106

def normalize_data(x, normalize_column):
    length = len(normalize_column)
    x_mean = np.reshape(np.mean(x[:, normalize_column],0), (1, length))
    x_std = np.reshape(np.std(x[:, normalize_column],0), (1, length))
    
    x[:, normalize_column] = np.divide(np.subtract(x[:, normalize_column], x_mean), x_std)
    return x
def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)
def classify(mu_0, mu_1, shared_cov, cnt_0, cnt_1, x):
    shared_cov_inv = np.linalg.inv(shared_cov)
    
    w = np.dot(np.transpose(mu_1 - mu_0), shared_cov_inv)
    b = (-0.5) * np.dot(np.dot(np.transpose(mu_1), shared_cov_inv), mu_1) + 1/2 * np.dot(np.dot(np.transpose(mu_0), shared_cov_inv), mu_0) + np.log(float(cnt_1) / cnt_0)
    z = np.dot(w, x.T) + b
    return np.around(sigmoid(z))
def accuracy(pred, label):
    return np.sum(pred == label)/len(pred)

def train(x_train, y_train):
    cnt_0 = 0
    cnt_1 = 0
    
    mu_0 = np.zeros((dim,))
    mu_1 = np.zeros((dim,))
    
    for i in range(y_train.shape[0]):
        if y_train[i] == 0:
            cnt_0 += 1
            mu_0 += x_train[i]
        elif y_train[i] == 1:
            cnt_1 += 1
            mu_1 += y_train[i]
    mu_0 /= cnt_0
    mu_1 /= cnt_1
    
    cov_0 = np.zeros((dim, dim))
    cov_1 = np.zeros((dim, dim))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 0:
            cov_0 += np.dot(np.transpose([x_train[i] - mu_0]), [x_train[i] - mu_0]) / cnt_0
        elif y_train[i] == 1:
            cov_1 += np.dot(np.transpose([x_train[i] - mu_1]), [x_train[i] - mu_1]) / cnt_1
    shared_cov = (cnt_0 * cov_0 + cnt_1 * cov_1) / x_train.shape[0]
    return mu_0, mu_1, shared_cov, cnt_0, cnt_1

x_train = pd.read_csv(sys.argv[3])
y_train = pd.read_csv(sys.argv[4], header = None)
x_train = x_train.values
y_train = y_train.values
y_train = y_train.reshape(-1)

x_test = pd.read_csv(sys.argv[5])
x_test = x_test.values

normalize_column = [0,1,3,4,5]
x_train = normalize_data(x_train, normalize_column)
x_test = normalize_data(x_test, normalize_column)

mu_0, mu_1, shared_cov, cnt_0, cnt_1 = train(x_train, y_train)
pred = classify(mu_0, mu_1, shared_cov, cnt_0, cnt_1, x_train)
acc = accuracy(pred, y_train)
print("accuracy: ", acc)

prediction = classify(mu_0, mu_1, shared_cov, cnt_0, cnt_1, x_test)
ans_file = open(sys.argv[6], "w")
writer = csv.writer(ans_file)
title = ['id','label']
writer.writerow(title) 
for i in range(len(x_test)):
    content = [i+1,int(prediction[i])]
    writer.writerow(content)