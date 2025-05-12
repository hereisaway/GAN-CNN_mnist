import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision

# 超参数
latent_dim = 100
batch_size = 64
num_epochs = 50
learning_rate = 0.0002

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST数据集
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 实例化模型
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# 损失和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.cuda()  # 将图像移动到GPU

        # 标签
        real_labels = torch.ones(images.size(0), 1).cuda()
        fake_labels = torch.zeros(images.size(0), 1).cuda()

        # 判别器训练
        optimizer_D.zero_grad()
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(images.size(0), latent_dim, 1, 1).cuda()  # 随机噪声
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_D.step()

        # 生成器训练
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss_real.item() + d_loss_fake.item():.4f}, g_loss: {g_loss.item():.4f}')

    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'./model/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./model/discriminator_{epoch}.pth')
        # 生成样本
        with torch.no_grad():
            z = torch.randn(16, latent_dim, 1, 1).cuda()
            fake_images = generator(z)
            fake_images = fake_images.view(-1, 1, 28, 28)
            grid_img = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')  # 移动到CPU以进行绘图
            plt.axis('off')
            plt.show()