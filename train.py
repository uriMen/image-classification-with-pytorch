import os
import datetime
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor
from torch import nn
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import tarfile
import multiprocessing as mp


# verify gpu
#print("GPU: ", torch.cuda.get_device_name(0))


# unzip
# my_tar = tarfile.open('train.tar')
# my_tar.extractall()
# my_tar.close()

train_data_path = './train' 
#train_imgs = os.listdir(train_data_path)
#print("Number of train imgs: ", len(train_imgs))

test_data_path = './test' 
#test_imgs = os.listdir(test_data_path)
#print("Number of test imgs: ", len(test_imgs))



# Hyper Parameters
batch_size = 32  
learning_rate = 0.001
num_epochs = 120
# default img size = 224 x 224
dflt_size = (224, 224)  #dar

class ImgDataSet(Dataset):

  def __init__(self, img_folder: str = train_data_path):
    
    self.img_folder_path = img_folder
    self.entries = self._create_entries()
    self.data = torch.Tensor()

  def _create_entries(self):
    entries = []
    data = []
    for img in os.listdir(self.img_folder_path):
        img_arr = self._resize(Image.open(self.img_folder_path + '/' + img))
        img_lbl = int(img.split('_')[1].split('.')[0])
        entries.append({'x': img_arr, 'y': img_lbl})
        data.append(img_arr)
    self.data = torch.stack(data)
    return entries
  
  def _resize(self, img):
    """resize img to default size 224 x 224, and standardize"""
    transform = T.Compose([
        T.Resize(dflt_size), #T.RandomResizedCrop(dflt_size) TODO check effect
        T.RandomPerspective(distortion_scale=0.6, p=0.3, interpolation=T.InterpolationMode.BILINEAR, fill=119),  #fill=round(256*mean(0.5227, 0.4495, 0.4206))
        T.RandomHorizontalFlip(), #dar
        T.ToTensor(),
        T.Normalize((0.5227, 0.4495, 0.4206),
                    (0.2389, 0.2276, 0.2239))
    ])
    return transform(img)
  
  def __getitem__(self, index):
    entry = self.entries[index]
    return entry['x'], entry['y']
  
  def __len__(self):
    return len(self.entries)


class ImgTestDataSet(Dataset):

    def __init__(self, img_folder: str = test_data_path):
        self.img_folder_path = img_folder
        self.entries = self._create_entries()
        self.data = torch.Tensor()

    def _create_entries(self):
        entries = []
        data = []
        for img in os.listdir(self.img_folder_path):
            img_arr = self._resize(Image.open(self.img_folder_path + '/' + img))
            img_lbl = int(img.split('_')[1].split('.')[0])
            entries.append({'x': img_arr, 'y': img_lbl})
            data.append(img_arr)
        self.data = torch.stack(data)
        return entries
  
    def _resize(self, img):
        """resize img to default size 300 x 300, and standardize"""
        transform = T.Compose([
        T.Resize(dflt_size),
        T.ToTensor(),
        T.Normalize((0.5227, 0.4495, 0.4206),
                    (0.2389, 0.2276, 0.2239))       
        ])    
        return transform(img)
  
    def __getitem__(self, index):
        entry = self.entries[index]
        return entry['x'], entry['y']
  
    def __len__(self):
        return len(self.entries)



print("Creating train_dataset...")
train_dataset = ImgDataSet()
print(f'train_dataset created. Size: {len(train_dataset)}')
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=3,pin_memory=True)

print("Creating test_dataset...")
test_dataset = ImgTestDataSet(test_data_path)
print(f'train_dataset created. Size: {len(test_dataset)}')

test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=1, pin_memory=True)


# define a model:
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  #reduce spatial to 112
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),   #reduce spatial to 56
            nn.Dropout(p=0.5))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,stride =2))  #reduce spatial to 28
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,stride =2),  #reduce spatial to 14
            nn.Dropout(p=0.5))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   #todo try 256
            nn.BatchNorm2d(128),
            nn.ReLU())
        #self.layer6 = nn.Sequential(                         #todo check when not skipped
        #    nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,stride =2))  #reduce spatial to 7
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=128*7*7, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=1))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)     #todo check when not skipped
        #out = self.layer7(out)
        out = out.view(out.size(0), -1)
        #out = self.dropout(out)
        #out = self.fc1(out)  
        return self.linear_layers(out) #no need for sigmoid because we use binary_cross_entropy_with_logits that has internal sigmoid
    

# Aux functions

def calcAvgLoss (model,loader,criterion):
    loss_array = []
    for i, (images, labels) in enumerate(loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels.float()) #outputs, labels)
        loss_array.append(float(loss.item())) #dar added float
    return np.mean(loss_array)


def calcF1score (model,loader):
    TP = 0
    F = 0  #=FP+FN
    for i, (images, labels) in enumerate(loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        sig = nn.Sigmoid()
        predicted = torch.round(sig(outputs.view(-1)))
        F += (predicted != labels).sum()
        TP += torch.logical_and(predicted,labels).sum()
    return float(2*TP)/float(2*TP+F)


def calcAcc (model,loader):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        sig = nn.Sigmoid()
        predicted = torch.round(sig(outputs.view(-1)))
        correct += (predicted == labels).sum()
        total += labels.shape[0]
    return float(correct)/float(total)


    
    

#creat net
cnn = CNN()
        
cnn.load_state_dict(torch.load('cnn_29.pkl'))  #only after strart of training
print(cnn)  
print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
    


if torch.cuda.is_available():
    cnn = cnn.cuda()

# Loss and Optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,90], gamma=0.2)

print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))


start_time = datetime.datetime.now()
print(f'***** TRAINING STARTED {start_time} ******')

trainLoss = []
testLoss = []
trainError = []
testError = []
trainAcc = []
testAcc = []

# convert all the weights tensors to cuda()
# Loss and Optimizer


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.to('cuda:0', non_blocking=True)
            labels = labels.to('cuda:0', non_blocking=True)
        outputs = cnn(images)
        loss = criterion(outputs.view(outputs.size(0)), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (((i+1) % 112) == 0):
            print(f'{datetime.datetime.now()} Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_dataset)//batch_size}] Loss: {round(loss.data.item(), 7)}')   #dar why round??? 

            # Collect error and Loss at every epoch
    cnn.eval()
    trainLoss.append(calcAvgLoss(cnn,train_loader,criterion))
    testLoss.append(calcAvgLoss(cnn,test_loader,criterion))
    trainError.append(calcF1score(cnn,train_loader))
    testError.append(calcF1score(cnn,test_loader))
    trainAcc.append(calcAcc(cnn,train_loader))
    testAcc.append(calcAcc(cnn,test_loader))
    cnn.train()
    
    # save model and results every 30 epochs
    if ((epoch + 1) % 30 == 0):
        torch.save(cnn.state_dict(), 'cnn_prespctive119_'+str(epoch)+'.pkl')
        data = {"trainLoss": trainLoss, "testLoss": testLoss,
                "trainError": trainError, "testError": testError,
                "trainAcc": trainAcc, "testAcc": testAcc}

        res = pd.DataFrame(data)
        res.to_csv('resultsPrespective119.csv')

        # evaluating the model after training
end_time = datetime.datetime.now()
print(f'Total Training Time: {end_time - start_time}')

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnnPrespective119_dar.pkl')


print(f'{datetime.datetime.now()} - Evaluating the model:')
cnn.eval() 

# correct = 0
# total = 0
# # accuracy = []

# for images, labels in test_loader:
#     if torch.cuda.is_available():
#         images = images.cuda()
#         labels = labels.cuda()
#     outputs = cnn(images)
#     sig = nn.Sigmoid()
#     predicted = torch.round(sig(outputs.data))
#     # accuracy.append()
#      # += (predicted.cpu() != labels).sum()
#     correct += (predicted == labels).sum()
#     total += labels.shape[0]
   

print('Train Accuracy of the model : %.4f' %(trainAcc[-1]))
print('Test Accuracy of the model : %.4f' %(testAcc[-1]))  #%(float(T)/float(T+F)))

data = {"trainLoss": trainLoss, "testLoss": testLoss,
        "trainError": trainError, "testError": testError,
        "trainAcc": trainAcc, "testAcc": testAcc}

res = pd.DataFrame(data)
res.to_csv('results.csv')

print(f'{datetime.datetime.now()} - END OF PROCESS ---------')
    
    #Plots
# Epochs = np.arange(0, num_epochs, 1)
# plt.plot( trainLoss, 'mo-', label = 'train')
# plt.plot(testLoss, 'co-', label = 'test' )
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Vs. Epochs')
# plt.legend()
# plt.show()

# plt.plot(Epochs, trainError, 'mo-',label = 'train')
# plt.plot(Epochs, testError, 'co-', label = 'test' )
# plt.xlabel('Epochs')
# plt.ylabel('Score')
# plt.title('F1 Score Vs. Epochs')
# plt.legend()
# plt.show()

