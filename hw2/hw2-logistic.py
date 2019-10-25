import pandas as pd
import numpy as np
import csv
import sys

def normalize_data(x, normalize_column):
    length = len(normalize_column)
    x_mean = np.reshape(np.mean(x[:, normalize_column],0), (1, length))
    x_std = np.reshape(np.std(x[:, normalize_column],0), (1, length))
    
    x[:, normalize_column] = np.divide(np.subtract(x[:, normalize_column], x_mean), x_std)
    return x
def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)
def cross_entropy(y_train, pred):
    return -y_train * np.log(pred) - (1-y_train) * np.log(1-pred)
def k_fold(x_train, y_train, fold_ratio, it):
    length = int(np.around(len(x_train)*fold_ratio))
    return np.concatenate((x_train[0:it*length],x_train[(it+1)*length:]), axis = 0), \
           np.concatenate((y_train[0:it*length],y_train[(it+1)*length:]), axis = 0), \
           x_train[it*length:(it+1)*length], y_train[it*length:(it+1)*length]
def classify(w, b, x):
    z = np.dot(x, w) + b
    return np.around(sigmoid(z))
def accuracy(pred, label):
    return np.sum(pred == label)/len(pred)

def train(x_train, y_train):
    b = 0.0
    w = np.zeros((x_train.shape[1],))
    lr = 0.2
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    epoch = 2000

    for e in range(epoch):
        z = np.dot(x_train, w) + b
        pred = sigmoid(z)
        loss = y_train - pred
        b_grad = -1*np.sum(loss)
        w_grad = -1*np.dot(loss, x_train)

        b_lr += b_grad**2
        w_lr += w_grad**2


        b = b-lr/np.sqrt(b_lr)*b_grad
        w = w-lr/np.sqrt(w_lr)*w_grad

        if (e+1) % 500 == 0:
            print("epoch: ", e)
            print("loss: ", np.mean(cross_entropy(y_train, pred)))
    return w, b 

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

w,b = train(x_train, y_train)
prediction = classify(w, b, x_test)

ans_file = open(sys.argv[6], "w")
writer = csv.writer(ans_file)
title = ['id','label']
writer.writerow(title) 
for i in range(len(x_test)):
    content = [i+1,int(prediction[i])]
    writer.writerow(content)