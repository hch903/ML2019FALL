import os
import random
import csv
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Linear(512,7)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1*1*512)
        x = self.fc(x)
        
        return x

class hw3_dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        return img


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    torch.cuda.set_device(0)
    
    test_set = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))

    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    test_dataset = hw3_dataset(test_set,transform)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = Resnet18()
    model.load_state_dict(torch.load('./model/model_20.pth'))
    
    if use_gpu:
        print("gpu!")
        model.cuda()
    model.eval()

    prediction = []

    with torch.no_grad():
        for idx, img in enumerate(test_loader):
            if use_gpu:
                img = img.cuda()
            output = model(img)
            predict = torch.max(output, 1)[1]
            prediction += predict.tolist()

    ans_file = open(sys.argv[2], "w")
    writer = csv.writer(ans_file)
    title = ['id','label']
    writer.writerow(title) 
    for i in range(len(prediction)):
        content = [i,int(prediction[i])]
        writer.writerow(content)