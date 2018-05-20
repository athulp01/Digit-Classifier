import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, csv):
        self.mnist_frame = pd.read_csv(csv)
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.mnist_frame)
    def __getitem__(self,index):
        label = self.mnist_frame.iloc[index,0]
        image = np.asarray(self.mnist_frame.iloc[index,1:785].values)
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        return image, label


def showImage(arr):
    image = np.resize(arr,(28,28))
    plt.imshow(image)


class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Sigmoid()

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        return output

train_dataset = MNISTDataset('./train.csv')
train_loader =  torch.utils.data.DataLoader(dataset=train_dataset,  batch_size=60, shuffle=True)
model = NeuralNet(784,300, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lossFunction = nn.CrossEntropyLoss()


if __name__ == "__main__":
    for epoch in range(0,5):
        for i,(image,label) in enumerate(train_loader):
            out = model(image)
            loss = lossFunction(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1)%100 == 0:
                print("Epoch[{}/{}], step[{}/{}], loss={:0.4f}".format(epoch+1,5,i+1,len(train_loader),loss.item()))

    torch.save(model.state_dict(), 'model.ckpt')

