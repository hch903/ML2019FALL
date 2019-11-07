import os
import sys
import random
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


def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    train_set = train_data[:28000]
    valid_set = train_data[28000:]

    return train_set, valid_set


class hw3_dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0]).convert('RGB')
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label

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


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    train_set, valid_set = load_data(sys.argv[1], sys.argv[2])

    #transform to tensor, data augmentation
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    train_dataset = hw3_dataset(train_set,transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    valid_dataset = hw3_dataset(valid_set,transform)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    model = Resnet18()
    if use_gpu:
        print("gpu!!")
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    num_epoch = 20
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))


        model.eval()
        with torch.no_grad():
            class_names = np.array(["Angry", "Disgust", "Fear", "Happy", "Sad", "Suprise", "Neutral"])
            valid_loss = []
            valid_acc = []
            y_true = []
            y_pred = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)

                y_true += label.tolist()
                y_pred += predict.tolist()
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))

        if np.mean(train_acc) > 0.9:
            checkpoint_path = 'model/model_{}.pth'.format(epoch+1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)

    