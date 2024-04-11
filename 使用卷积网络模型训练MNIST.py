import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将数据转换为Tensor格式
    transforms.Normalize((0.1307,), (0.3081,))  # 数据归一化处理
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)  # 加载训练集
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)  # 加载测试集

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 创建训练集数据加载器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 创建测试集数据加载器

# 定义自定义模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 第一个卷积层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 最大池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 第二个卷积层
        self.fc1 = nn.Linear(7 * 7 * 32, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 10)  # 第二个全连接层

    def forward(self, x):
        x = self.conv1(x)  # 第一个卷积层
        x = self.relu(x)  # ReLU激活函数
        x = self.maxpool(x)  # 最大池化层
        x = self.conv2(x)  # 第二个卷积层
        x = self.relu(x)  # ReLU激活函数
        x = self.maxpool(x)  # 最大池化层
        x = x.view(x.size(0), -1)  # 展开为一维向量
        x = self.fc1(x)  # 第一个全连接层
        x = self.relu(x)  # ReLU激活函数
        x = self.fc2(x)  # 第二个全连接层
        return x

# 创建模型实例、定义损失函数和优化器
model = CustomModel()  # 创建自定义模型实例
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        accuracy = test(model, test_loader)  # 在测试集上评估模型准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy on test set: {accuracy:.4f}")


def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)  # 前向传播
            _,predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累计样本数量
            correct += (predicted == labels).sum().item()  # 统计正确预测的数量
    accuracy = correct / total  # 计算准确率
    return accuracy

num_epochs = 10
train(model, train_loader, criterion, optimizer, num_epochs)  # 开始训练