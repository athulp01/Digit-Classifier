import mnist

class MNISTtestDataset(torch.utils.data.Dataset):
    def __init__(self, csv):
        self.mnist_frame = pd.read_csv(csv)
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.mnist_frame)
    def __getitem__(self,index):
        image = np.asarray(self.mnist_frame.iloc[index,0:784].values)
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        print(len(image))
        return image

model = mnist.NeuralNet(784,300, 10)
model.load_state_dict(torch.load('./model.ckpt'))

if __name__ == "__main__":
    test_dataset = MNISTtestDataset('./test.csv')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=1, shuffle=False)

    a = []
    b = []
    s = pd.DataFrame(columns=['ImageId','Label'])
    with torch.no_grad():
        for i,image in enumerate(test_loader):
            print(image.size())
            print("test ", i+1)
            out = model(image)
            _,pred = torch.max(out.data,1)
            a.append(int(pred))
            b.append(i+1)

    s['ImageId'] = b
    s['Label'] = a
    s.to_csv("./submit.csv")
