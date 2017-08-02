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
learning_rate = 0.001               # sgd 0.003 adam 0.001
height, width = 112, 112

data_dir = '/home/vision/annajung/datasets/UCF101_video_and_jpg'
traindir = os.path.join(data_dir, 'train_9_1')
valdir = os.path.join(data_dir, 'test_9_1')


train_loader = data.DataLoader(ImageFolder(traindir, transforms.Compose([
                                        transforms.RandomCrop(height),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
                                    ])), batch_size=batch_size, shuffle=True, collate_fn=ucf_collate)
val_loader = data.DataLoader(ImageFolder(valdir, transforms.Compose([
                                        transforms.RandomCrop(height),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
                                    ])), batch_size=batch_size, shuffle=True, collate_fn=ucf_collate)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

# C3D Model

class C3D_cls20(nn.Module):
    def __init__(self):
        super(C3D_cls20, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 20))

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

class C3D_cls46(nn.Module):
    def __init__(self):
        super(C3D_cls46, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 46))

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


if __name__ == '__main__':
    with torch.cuda.device(2):
        c3d = C3D_cls46().cuda().apply(weights_init)
        #c3d.load_state_dict(torch.load('c3d_2.pkl'))
        '''
        c3d_pretrained = C3D_cls20()
        c3d_pretrained.load_state_dict(torch.load('pkl_list/c3d_2_cla20.pkl'))
        c3d_pretrained_dict = c3d_pretrained.state_dict()
        c3d = C3D_cls99()
        c3d_dict = c3d.state_dict()
    
        # reference: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in c3d_pretrained_dict.items() if k in c3d_dict}
        # overwrite entries in the existing state dict
        c3d_dict.update(pretrained_dict)
        # Load the new state dict
        c3d.load_state_dict(c3d_dict)
        # then throw to cuda
        c3d = c3d.cuda(1)
        '''

        pkl_save_path = 'c3d_6_adam_cla46.pkl'
        train_mode = True
        early_stopping_cnt = 0
        early_stopping_flag = False
        best_acc = 0
        #past_loss_save = 10
        correct = 0
        total = 0
        #past_loss_count = 0

        if train_mode:
            criterion = nn.CrossEntropyLoss(size_average=False)
            optimizer = torch.optim.Adam(c3d.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                if not early_stopping_flag:
                    if (epoch+1) % 5 == 0:
                        learning_rate = learning_rate/2
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate

                    for i, (images, labels) in enumerate(train_loader):
                        images = Variable(images).cuda()
                        labels_ori = labels
                        labels = Variable(labels).cuda()

                        # Forward + Backward + Optimize
                        optimizer.zero_grad()
                        outputs = c3d(images)
                        loss = criterion(outputs, labels.long())
                        loss.backward()
                        optimizer.step()

                        if (i + 1) % 10 == 0:
                            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                  % (epoch + 1, num_epochs, i + 1, 650, loss.data[0]))

                            _, predicted = torch.max(outputs.data, 1)
                            total = labels.size(0)
                            correct = (predicted == labels_ori.long().cuda()).sum()
                            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy : %d %%'
                                  % (epoch + 1, num_epochs, i + 1, 650, 100 * correct / total))
                            print('-------------------------------------------------------')

                    c3d.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
                    for i_val, (images_val, labels_val) in enumerate(val_loader):
                        images_val = Variable(images_val).cuda()
                        outputs = c3d(images_val)
                        labels_val = labels_val.long().cuda()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels_val.size(0)
                        correct += (predicted == labels_val).sum()
                        acc = correct / total

                        if acc > best_acc:
                            best_acc = acc
                            torch.save(c3d.state_dict(), pkl_save_path)
                            early_stopping_cnt = 0
                        else:
                            early_stopping_cnt += 1
                            # if early_stopping_cnt > 100:
                            #     early_stopping_flag True

                        if (i_val + 1) % 10 == 0:
                            print('Test Accuracy of the model on the %d test images: %d %%'
                                  % ((i_val + 1) * batch_size, 100 * correct / total))
                            print('--------------------------------------------------------')



        else:
            c3d.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(val_loader):
                images = Variable(images).cuda()
                outputs = c3d(images)
                labels = labels.long().cuda()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                if (i + 1) % 10 == 0:
                    print('Test Accuracy of the model on the %d test images: %d %%'
                          % (i+1, 100 * correct / total))
