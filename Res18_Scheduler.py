import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,LambdaLR
import torchvision

# Parameters
num_epochs = 465
batch_size = 100
learning_rate = 0.001
'''

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

# Design the convolution block
class ConvBlock(nn.Module):
    def __init__(self,in_channel,output_channel,kernel_size,stride,padding):
        super(ConvBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channel,output_channel,kernel_size,stride,padding)
        self.bn1=nn.BatchNorm2d(output_channel)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        return x
    
# Design the residual block   
class ResBlock(nn.Module):
    def __init__(self,in_channel,output_channel,first=False):
        super(ResBlock, self).__init__() 
        #Set up residual channel and projection
        res_channel= in_channel//4 #set up the res_channel to increase the speed of machine learning
        stride=1
        self.projection=in_channel!=output_channel
        if self.projection:
            self.p=ConvBlock(in_channel,output_channel,1,2,0)
            stride=2
            res_channel=in_channel//2

        if first:
            self.p=ConvBlock(in_channel,output_channel,1,1,0)
            stride=1
            res_channel=in_channel

        #Design the layers
        self.conv1=ConvBlock(in_channel,res_channel,1,1,0)
        self.conv2=ConvBlock(res_channel,res_channel,3,stride,1)
        self.conv3=ConvBlock(res_channel,output_channel,1,1,0)
        self.relu=nn.ReLU()

    def forward(self,x):
        f=self.relu(self.conv1(x))  # The first weight layer convolution block + relu
        f=self.relu(self.conv2(f)) #The residual channel to optimize the calculation complexity
        f=self.conv3(f) #The final layer for residul block
        # Projection - ensure size of x and f(x) are pattern
        if self.projection:
            x=self.p(x)
        # Shortcut the f(x) and x
        h=self.relu(torch.add(f,x)) # h=x+f(x)
        return h

# Design the residual model
class ResNet(nn.Module):
    def __init__(self,no_blocks,in_channel=3,classes=10): #for different projects, change in_channel and classes
        super(ResNet, self).__init__()
        output_feature=[64,128,256,512] #default output channnels, can be optimized based on size of data

        #Set up the first residual block
        self.blocks=nn.ModuleList([ResBlock(64,64,True)])#change the input channel and out channels based on data

        # Iterating the residual block based on number of block
        for i in range(len(output_feature)):
            if i>0:
                self.blocks.append(ResBlock(output_feature[i-1],output_feature[i]))

            for _ in range(no_blocks[i]-1):
                self.blocks.append(ResBlock(output_feature[i],output_feature[i]))
        
        # Design the model
        self.conv1=ConvBlock(in_channel,64,3,1,1)
        self.bn1=nn.BatchNorm2d(64)
        self.pool1=nn.MaxPool2d(2,2,padding=1)
        self.avgpool1=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(output_feature[-1],classes)
        self.relu=nn.ReLU()

    def forward(self, x):
        #set up the first convolution layer
        x=self.relu(self.conv1(x))
        x=self.bn1(x)
        x=self.pool1(x)

        # The residual blocks h = f(x)+x
        for block in self.blocks:
            x=block(x)

        #flatten the x 
        x=self.avgpool1(x)
        x = torch.flatten(x,1)
        # The final fully connected layer, from convolution layer to output layer
        x=self.fc(x)
        return x

no_blocks=[2,2,2,2]
model = ResNet(no_blocks).to(device)  # Move the model to GPU

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-4)

# Warmup function
warm_epoches=10

def lr_lambda(epoch):
    if epoch < warm_epoches:
        return (epoch + 1) / warm_epoches  # Linearly increase LR
    return 1

#learning rate scheduler CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

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
    if epoch < warm_epoches:
        warmup_scheduler.step()  
    else:
        scheduler.step()
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