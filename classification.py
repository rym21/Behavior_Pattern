import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import shutil
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os

random_state = 42
np.random.seed(random_state)

original_dataset_dir = os.getcwd() + "\CPMN\images"
total_num = int(len(os.listdir(original_dataset_dir)) / 2)
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)

base_dir = os.getcwd() + "\CPMN\images_train_test"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

sub_dirs = ['train', 'test']
samples = ['works', 'lives']
train_idx = random_idx[:int(total_num * 0.67)]
test_idx = random_idx[int(total_num * 0.67):]
numbers = [train_idx, test_idx]
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for sample in samples:
        sample_dir = os.path.join(dir, sample)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        fnames = [sample[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(sample_dir, fname)
            shutil.copyfile(src, dst)

        # 验证训练集、验证集、测试集的划分的照片数目
        print(sample_dir + ' total images : %d' % (len(os.listdir(sample_dir))))

random_state = 1
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)

epochs = 20
batch_size = 8
num_workers = 0
use_gpu = torch.cuda.is_available()
PATH= os.getcwd() + '/model.pt'

data_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(root=os.getcwd() + '\CPMN\images_train_test/train/',
                                     transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

test_dataset = datasets.ImageFolder(os.getcwd() + '\CPMN\images_train_test/test/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
if(os.path.exists(os.getcwd() + '/model.pt')):
    net=torch.load(os.getcwd() + '/model.pt')

if use_gpu:
    net = net.cuda()
print(net)

cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def train():

    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        # model test
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        net.eval()
        test_label = torch.tensor([]).long().cuda()
        test_pred = torch.tensor([]).long().cuda()
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()
            test_label = torch.cat((test_label, labels.data), 0)
            test_pred = torch.cat((test_pred, predicted), 0)
        print('epoch %d train loss: %.3f  train_acc: %.3f ' % (epoch + 1, running_loss / train_total, 100 * train_correct / train_total),
              'test_acc: %.3f ' % (100 * correct / test_total))
        test_label = test_label.cpu().numpy()
        test_pred = test_pred.cpu().numpy()

        acc = accuracy_score(test_label, test_pred)
        f1 = f1_score(test_label, test_pred, average='weighted')
        roc = roc_auc_score(test_label, test_pred, average='weighted')
        print("acc:", acc, " f1:", f1, " roc:", roc)


    torch.save(net, os.getcwd() + '/model.pt')


if __name__ == '__main__':

    train()

