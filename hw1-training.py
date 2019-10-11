import sys
import pandas as pd
import numpy as np
import csv
import math

def valid(x, y):
    if y <= 2 or y > 100:
        return False
    for i in range(9):
        if x[9,i] <= 2 or x[9,i] > 100:
            return False
    return True

def minibatch(x, y):
    # 打亂data順序
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 訓練參數以及初始化
    batch_size = 64
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.01
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    for num in range(1000):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            # 計算gradient
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            # 更新weight, bias
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            
    return w, bias

year1_raw_data = pd.read_csv("year1-data.csv")
year2_raw_data = pd.read_csv("year2-data.csv")

raw_data = year1_raw_data.append(year2_raw_data, ignore_index=True)
raw_data = pd.DataFrame(raw_data).fillna(0)
raw_data = raw_data.to_numpy(dtype = "<U5")
data = raw_data[: , 2:]
data = np.char.replace(data, 'NR', '0')
data = np.char.replace(data, '#', '')
data = np.char.replace(data, '*', '')
data = np.char.replace(data, 'x', '')
data = data.astype(np.float)
for i in range(731):
    for j in range(24):
        # if data[15 + 17*i, j] > 180:
        #     data[15 + 17*i, j] = 360 - data[15 + 17*i, j]
        if data[1 + 17*i, j] == 0:
            data[1 + 17*i, j] = 1.9
        if data[8 + 17*i, j] == 1000:
            data[8 + 17*i, j] = 0
        if data[12 + 17*i, j] > 10 or data[12 + 17*i, j] < 0:
            data[12 + 17*i, j] = 0

thirty_days_month = [3,5,8,10,15,17,20,22]
thirtyone_days_month = [0,2,4,6,7,9,11,12,14,16,18,19,21,23]
prev_days = 0
for month in range(24):
    if month in thirty_days_month:
        days = 30
    elif month in thirtyone_days_month:
        days = 31
    elif month == 1:
        days = 29
    elif month == 13:
        days = 28
    tmp = np.empty(shape = (18, days*24))
    
    for day in range(days):
        for hour in range(24):
            tmp[:, day * 24 + hour] = data[18 * (prev_days + day): 18 * (prev_days + day + 1), hour]

    prev_days += days
    if month == 0:
        new_data = tmp
    else:
        new_data = np.hstack((new_data, tmp))

x = []
y = []

# 用前面9筆資料預測下一筆PM2.5 所以需要-9
total_length = new_data.shape[1] - 9
for i in range(total_length):
    x_tmp = new_data[:,i:i+9]
    y_tmp = new_data[9,i+9]
    if valid(x_tmp, y_tmp):
        x.append(x_tmp.reshape(-1,))
        y.append(y_tmp)
# x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
x = np.array(x)
y = np.array(y)

x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1).astype(float)

w, bias = minibatch(x,y)
np.save('weight.npy',w)