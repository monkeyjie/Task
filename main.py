import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

class MyDataset(Dataset):
	def __init__(self,raw_datas,labels):
		self.x_data = raw_datas
		self.y_data = labels
		#self.transform = transforms.Compose([transforms.ToTensor()])
	def __getitem__(self,index):
		return self.x_data[index] , self.y_data[index]
	
	def __len__(self):
		return len(self.y_data)


datadir="/Users/jayden/Documents/mindcoord/Task/dataset"
classes=os.listdir(datadir)
i=0
raw_datas=[]
labels=[]
for classname in classes:
	txtfiles=os.listdir(datadir+"/"+classname)
	class_label=i
	i=i+1
	print(txtfiles)
	for txtfile in txtfiles:
		print("I am reading:"+datadir+"/"+classname+"/"+txtfile + "/n")
		f=open(datadir+"/"+classname+"/"+txtfile)
		line=f.readline()
		while line:
			raw_data=line.split(',')[0:-1]
			raw_datas.append(raw_data)
			labels.append(class_label)
			line=f.readline()
		
print(np.array(raw_datas).shape)
print(np.array(labels).shape)
raw_datas=np.array(raw_datas)
labels=np.array(labels)
labels=torch.from_numpy(labels)
#raw_datas=raw_datas[:,np.newaxis,:]
raw_datas=raw_datas.astype(np.float32)
raw_datas=torch.from_numpy(raw_datas)
dataset=MyDataset(raw_datas,labels)
print(raw_datas.dtype)
print(labels.dtype)
print(raw_datas)


batch_size = 16
validation_split = .3
shuffle_dataset = True
random_seed= 42

dataset_size = len(raw_datas)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler,)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)



class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.l1=nn.Linear(3,64)
		self.l2=nn.Linear(64,64)
		self.l3=nn.Linear(64,5)

	def forward(self,x):
		x=x.view(-1,3)
		x=F.sigmoid(self.l1(x))
		x=F.sigmoid(self.l2(x))
		return F.log_softmax(self.l3(x),dim=1)


def train(net, dataloader, criterion, optimizer, epochs=100,device):

	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(dataloader, 0):
        		# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
        		# zero the parameter gradients
			optimizer.zero_grad()

        		# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

        		# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	torch.save(model.state_dict(), "./checkpoint/model_checkpoint.pt")
	print('Finished Training')


model=Net()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
train(model,train_loader,criterion,optimizer,50000,device)

