import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
num_epochs = 100
batch_size = 30
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.CIFAR10(root='./data/',
                            train=True,
                            transform= transforms.Compose([transforms.Scale(112),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])]),
                              download=True)

test_dataset = dsets.CIFAR10(root='./data/',
                           train=False,
                             transform=transforms.Compose([transforms.Scale(112),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ToTensor(),
                                                            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.fc_s = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),               #
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc3 = nn.Sequential(
            nn.Linear(2048, 10))           #101

        # self._features = nn.Sequential(
        #     self.group1,
        #     self.group2,
        #     self.group3,
        #     self.group4,
        #     self.group5
        # )
        #
        # self._classifier = nn.Sequential(
        #     self.fc1,
        #     self.fc2
        # )

    def forward(self, x):
        out = self.layers(x)
        #print(out.size(0))
        out = out.view(out.size(0), -1)
        out = self.fc_s(out)
        return self.fc3(out)


c3d = C3D().cuda(1)
c3d.apply(weights_init)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda(1)
optimizer = torch.optim.Adam(c3d.parameters(), lr=learning_rate)

past_loss_save = 0
past_loss_count = 0
# Train the Model
for epoch in range(num_epochs):
    # if (epoch+1) % 30 == 0:
    #     learning_rate = learning_rate/2
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate
    #         print(param_group['lr'])
    for i, (images, labels) in enumerate(train_loader):
        #print(images[0][0][0][30:60][30:60])
        x = torch.randn(images.size(0), 3, 16, 112, 112)
        images = torch.unsqueeze(images, 2)
        images = images.expand_as(x)
        images = Variable(images).cuda(1)
        labels_ori = labels
        labels = Variable(labels).cuda(1)
        #print(images.size())
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = c3d(images)
        #print(outputs.size())
        #print(labels.size())
        #print(labels.type())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # _, predicted = torch.max(outputs.data, 1)
        # total = labels.size(0)
        # correct = (predicted == labels_ori.cuda()).sum()
        # print('%d %%' % (100 * correct / total))

        if (i + 1) % 10 == 0:

            if past_loss_count == 30:
                past_loss_count = 0
                learning_rate = learning_rate / 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                    print(param_group['lr'])
            if past_loss_save < loss.data[0]:
                past_loss_count += 1
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]*batch_size))
            past_loss_save = loss.data[0]
            torch.save(c3d.state_dict(), 'c3d_pretrain_cifar10.pkl')

# Test the Model
c3d.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda(1)
    outputs = c3d(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda(1)).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(c3d.state_dict(), 'c3d_pretrain_cifar10.pkl')