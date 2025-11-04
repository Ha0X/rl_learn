import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ========================
# 1. 定义 VAE 模型
# ========================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器：x -> h -> (mu, logvar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：z -> h -> x_recon
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log(σ^2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5*logσ²)
        eps = torch.randn_like(std)    # 采样 ε ~ N(0,1)
        return mu + eps * std          # z = μ + σ*ε

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # 输出像素在 [0,1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ========================
# 2. 定义损失函数
# ========================
def vae_loss(x, x_recon, mu, logvar):
    # 重建误差（这里用 BCE）
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL 散度：有闭式解
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ========================
# 3. 训练过程
# ========================
def train_vae():
    # 数据准备（MNIST）
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    model.train()
    for epoch in range(10):
        train_loss = 0
        for x, _ in dataloader:
            x = x.view(-1, 784).to(device)  # 展平 28x28 -> 784
            optimizer.zero_grad()

            x_recon, mu, logvar = model(x)
            loss = vae_loss(x, x_recon, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {train_loss/len(dataset):.4f}")

    return model

import matplotlib.pyplot as plt

def visualize_generation(model, n=20):
    model.eval()
    with torch.no_grad():
        # 随机采样潜在变量 z ~ N(0, I)
        z = torch.randn(n, model.fc_mu.out_features).to(next(model.parameters()).device)
        samples = model.decode(z).cpu()  # 解码成像素

    # 展示生成结果
    fig, axes = plt.subplots(2, n//2, figsize=(n, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].view(28, 28), cmap="gray")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    vae_model = train_vae()
    visualize_generation(vae_model, n=20)


