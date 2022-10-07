import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import csv

# checking if GPU is available or not
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# torchvision has CIFAR10 dataset built in
train_ds = datasets.CIFAR10("./data", train=True, download=True)
valid_ds = datasets.CIFAR10("./data", train=False)

class TestData(datasets.VisionDataset):
    
    filepath = "./test.npy"
    
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data = np.load(self.filepath)
    
    def __getitem__(self, index: int):
        
        img = self.data[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img
        
    def __len__(self):
        return len(self.data)

test_ds = TestData("./data")

# # uncomment the below lines for visualizing the data
# # take the first 16 samples and visualize them in a grid
# fig = plt.figure(figsize=(8, 8))
# cls_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', \
#             5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
# for i in range(16):
#     ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
#     ax.imshow(train_ds[i][0])
#     ax.text(1, 4, str(cls_dict[train_ds[i][1]]))
# plt.show()


# first transform the images to tensor format, 
# then normalize the pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_ds.transform = transform
valid_ds.transform = transform
test_ds.transform = transform

train_ds[0][0].shape

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=64, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_ds, batch_size=1000
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=1000
)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

convnet = ConvNet().to(device)
print(convnet)
    
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(convnet.parameters(), lr=0.001, momentum=0.9)

def train(epoch, model, trainloader, criterion, optimizer):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # compare predictions to ground truth
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # gather statistics
        running_loss += loss.item()
        
    running_loss /= len(trainloader)
    
    print('Training | Epoch: {}| Loss: {:.3f} | Accuracy on 50000 train images: {:.1f}'.format \
          (epoch+1, running_loss, 100 * correct / total))

def validate(epoch, model, valloader, criterion):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    for i, data in tqdm(enumerate(valloader, 0)):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # compare predictions to ground truth
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # gather statistics
        running_loss += loss.item()
        
    running_loss /= len(valloader)
    
    print('Validation | Epoch: {}| Loss: {:.3f} | Accuracy on 10000 val images: {:.1f}'.format \
          (epoch+1, running_loss, 100 * correct / total))

# training for 20 epochs
for epoch in range(20):
    train(epoch, convnet, train_loader, criterion, optimizer)

# validating
validate(epoch, convnet, valid_loader, criterion)

def predict(model,testloader):
    
    model.eval()
    preds = []
    with torch.no_grad():
        # labels are not available for the actual test set
        for feature in tqdm(testloader):
            # calculate outputs by running images through the network
            outputs = model(feature.to(device))
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())
    
    return preds

predictions = predict(convnet, test_loader)
with open("submission.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for i, label in enumerate(predictions):
        writer.writerow([i, label])