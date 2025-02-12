import torch
import torch.nn as nn
from torchvision import transforms
import torchvision

# Parameters
num_epochs = 50
batch_size = 1000
learning_rate = 0.001
'''
85.77% Accuracy Rate for Testing Dataset
Key steps which influence the performance of CNN model
Device --> GPU
a. Data Preprocesssing
1. Size of train loader --> Recommodate 100% data from train dateset, which includes 50,000 images
2. Batch Size --> 1000
3. Epoches --> 40 ~ 60
4. Flip and crop images --> Avoid overfitting

b. Model Design 
1. Number of convolution layers --> three layers --> CNN + Relu + Bn + PoolMax
2. Number of fully connected layer --> three layers --> Linear + Relu (The final output layer stay with linear)
3. Stride --> 1, Padding --> 2, Kernal --> 3 for convolution layer and 2 * 2 for pooling max layer

c. Optimizer and loss
1. Optimizer --> AdamW, Loss --> Cross Entrophy
'''

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Preprocessing - dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Check the information for images
for images, labels in train_loader:
    channels, height, width = images.shape[1:]  # input channel, height, weight
    print(f"Channels: {channels}, Height: {height}, Width: {width}")
    break

# Design the CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=2)  
        self.relu1 = nn.LeakyReLU(0.3)
        self.bn1 = nn.BatchNorm2d(128)  # The attribute for bn model is number of current input channels
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=2)  
        self.relu2 = nn.LeakyReLU(0.3)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3=nn.Conv2d(256,512,3, padding=2)
        self.relu3 = nn.LeakyReLU(0.3)
        self.bn3=nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 5 * 5, 260)  # we need to find height and weight after convolution and pooling layers
        self.relu4 = nn.LeakyReLU(0.3)
        self.fc2 = nn.Linear(260, 128)
        self.relu5 = nn.LeakyReLU(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolution layer: conv1 + relu1 + bn + poolmax1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        # Second convolution layer 
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        # Third convolution layer
        x= self.conv3(x)
        x= self.relu3(x)
        x= self.bn3(x)
        x= self.pool3(x)
        # Fully connected layer 1
        x = x.view(-1, 512 * 5 * 5)
        x = self.fc1(x)
        x = self.relu4(x)
        # Fully connected layer 2
        x = self.fc2(x)
        x = self.relu5(x)
        # Fully connected layer 3 (to output, for the classification model, the last layer is softmax)
        x = self.fc3(x)
        return x

model = ConvNet().to(device)  # Move the model to GPU

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-4)

# Training the data
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):    
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
print('Finished Training')

# Check accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network: {acc} %')

for i in range(10):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc} %')