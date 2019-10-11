import sys
import pandas as pd
import numpy as np
import csv
import math

w = np.load('weight.npy')                                   ## load weight
test_raw_data = pd.read_csv(sys.argv[1])
test_raw_data = pd.DataFrame(test_raw_data).fillna(0)
test_raw_data = test_raw_data.to_numpy(dtype = "<U5")
test_data = test_raw_data[0:, 2: ]
test_data = np.char.replace(test_data, 'NR', '0')
test_data = np.char.replace(test_data, '#', '')
test_data = np.char.replace(test_data, '*', '')
test_data = np.char.replace(test_data, 'x', '')
test_data = test_data.astype(np.float)

test_x = np.empty(shape = (500, 18 * 9),dtype = float)

for i in range(500):
    test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1)
test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
answer = test_x.dot(w)

f = open(sys.argv[2],"w")
w = csv.writer(f)
title = ['id','value']
w.writerow(title) 
for i in range(500):
    content = ['id_'+str(i),answer[i][0]]
    w.writerow(content) 