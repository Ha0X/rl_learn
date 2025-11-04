import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ========================
# 1. 判别器
# ========================
class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ========================
# 2. 生成器
# ========================
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# ========================
# 3. 判别器和生成器损失
# ========================
def d_loss_fn(D_real, D_fake):
    return -torch.mean(torch.log(D_real + 1e-8) + torch.log(1 - D_fake + 1e-8))


def g_loss_fn(D_fake):
    return -torch.mean(torch.log(D_fake + 1e-8))


# ========================
# 4. 训练
# ========================
def train_gan(epochs=20, batch_size=128, latent_dim=100, lr=0.0002):
    # 数据加载 (MNIST 归一化到 [-1,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator().to(device)
    G = Generator(latent_dim=latent_dim).to(device)

    # 优化器
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    # 训练循环
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.view(-1, 784).to(device)  # 展平成向量
            batch_size = real_images.size(0)

            # ========================
            # 训练判别器 D
            # ========================
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = G(z).detach()  # 假图 (不更新G)
            D_real = D(real_images)
            D_fake = D(fake_images)
            D_loss = d_loss_fn(D_real, D_fake)

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # ========================
            # 训练生成器 G
            # ========================
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = G(z)
            D_fake = D(fake_images)
            G_loss = g_loss_fn(D_fake)

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {D_loss.item():.4f}  G_loss: {G_loss.item():.4f}")

        # 每隔几个 epoch 可视化生成图片
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                z = torch.randn(16, latent_dim).to(device)
                samples = G(z).view(-1, 1, 28, 28).cpu()
                grid = torch.cat([s for s in samples], dim=2).squeeze().numpy()
                plt.imshow(grid, cmap="gray")
                plt.title(f"Epoch {epoch+1}")
                plt.axis("off")
                plt.show()

    return D, G


if __name__ == "__main__":
    train_gan()
