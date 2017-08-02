from folder_ import ImageFolder, ucf_collate
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


# Hyper Parameters
num_epochs = 100
batch_size = 20
learning_rate = 0.003

data_dir = '/home/vision/annajung/datasets/UCF101_video_and_jpg'
traindir = os.path.join(data_dir, 'train')
valdir = os.path.join(data_dir, 'test')

train_loader = data.DataLoader(ImageFolder(traindir, transforms.Compose([
                                        transforms.RandomCrop(112),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
                                    ])), batch_size=batch_size, shuffle=True, collate_fn=ucf_collate)
val_loader = data.DataLoader(ImageFolder(valdir, transforms.Compose([
                                        transforms.RandomCrop(112),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
                                    ])), batch_size=batch_size, shuffle=True, collate_fn=ucf_collate)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


# C3D Model
class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        #init.xavier_normal(self.group1.state_dict()['weight'])
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group2.state_dict()['weight'])
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group3.state_dict()['weight'])
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group4.state_dict()['weight'])
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group5.state_dict()['weight'])

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),               #
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc1.state_dict()['weight'])
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 32))           #101

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3,
            self.group4,
            self.group5
        )

        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return self.fc3(out)


c3d = C3D().cuda(0)
c3d.apply(weights_init)
#c3d.load_state_dict(torch.load('c3d_4_sgd_cla101.pkl'))

train_mode = 1

if train_mode is 1:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(c3d.parameters(), lr=learning_rate, momentum=0.9)

    past_loss_save = 10
    past_loss_count = 0

    # Train the Model
    for epoch in range(num_epochs):
        if (epoch+1) % 5 == 0:
            learning_rate = learning_rate/2
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        for i, (images, labels) in enumerate(train_loader):
            #x = torch.randn(20, 3, 16, 112, 112)

            #images = images.expand_as(x)
            #images = Variable(torch.randn(20, 3, 16, 112, 112)).cuda(1)
            #labels = torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda(1)
            #labels_variable = Variable(labels)
            #print(images.size())
            #print("images")
            images = Variable(images).cuda(0)
            #print(c3d.state_dict())
            #images = Variable(torch.randn(1, 3, 16, 112, 112)).cuda()
            #print("labels")
            labels_ori = labels
            labels = Variable(labels).cuda(0)
            #print(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = c3d(images)
            #images.register_hook(print)
            #print(outputs.size())
            #print(labels.size())
            loss = criterion(outputs, labels.long())
            #print("before backward")
            loss.backward()
            #print("after backward")
            optimizer.step()

            if (i + 1) % 10 == 0:
                if loss.data[0] < past_loss_save:
                    #print(loss.data[0])
                    past_loss_save = loss.data[0]
                    torch.save(c3d.state_dict(), 'c3d_4_sgd_cla101.pkl')

                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Loss x batch : %.4f'
                      % (epoch + 1, num_epochs, i + 1, 9320 // batch_size, loss.data[0], loss.data[0]*images.size(0)))

                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels_ori.long().cuda(0)).sum()
                print('Training Accuracy : %d %%' % (100 * correct / total))
else:
    c3d.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        #print("validation mode")
        images = Variable(images).cuda(0)
        outputs = c3d(images)
        labels = labels.long().cuda(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if (i + 1) % 10 == 0:
            print('Test Accuracy of the model on the %d test images: %d %%' % (i+1, 100 * correct / total))

# Save the Trained Model
#torch.save(c3d.state_dict(), 'c3d.pkl')