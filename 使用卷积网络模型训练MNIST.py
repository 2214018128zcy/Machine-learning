import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



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

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.tolist())  # Collect predicted labels
    accuracy = correct / total
    return accuracy, predicted_labels

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        accuracy, _ = test(model, test_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy on test set: {accuracy:.4f}")

        train_losses.append(running_loss / len(train_loader))
        test_accuracies.append(accuracy)

    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    # Plot accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.show()

    # Calculate confusion matrix and plot
    _, predicted_labels = test(model, test_loader)
    true_labels = test_dataset.targets
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

num_epochs = 10
train(model, train_loader, criterion, optimizer, num_epochs)  # 开始训练
