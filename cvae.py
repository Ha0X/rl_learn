import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================
# CVAE 模型定义
# ========================
class CVAE(nn.Module):
    def __init__(self, state_dim=2, action_dim=6, latent_dim=4, hidden_dim=64):
        super(CVAE, self).__init__()
        # 编码器：输入 (s, a) → 输出 z 的分布参数 (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：输入 (s, z) → 输出动作序列 a_hat
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出动作序列
        )

    def encode(self, s, a):
        h = self.encoder(torch.cat([s, a], dim=-1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, s, z):
        return self.decoder(torch.cat([s, z], dim=-1))

    def forward(self, s, a):
        mu, logvar = self.encode(s, a)
        z = self.reparameterize(mu, logvar)
        a_hat = self.decode(s, z)
        return a_hat, mu, logvar

# ========================
# 损失函数（重建误差 + KL 散度）
# ========================
def cvae_loss(a, a_hat, mu, logvar):
    recon_loss = F.mse_loss(a_hat, a, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ========================
# 示例训练
# ========================
def train_cvae():
    state_dim, action_dim, latent_dim = 2, 6, 4
    model = CVAE(state_dim, action_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 伪造数据：状态=2维，动作序列=6维
    for epoch in range(10):
        s = torch.randn(32, state_dim)   # batch=32
        a = torch.randn(32, action_dim)  # 动作序列
        a_hat, mu, logvar = model(s, a)
        loss = cvae_loss(a, a_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

# ========================
# 采样：给定状态 s，生成动作序列
# ========================
def generate_actions(model, n=5):
    model.eval()
    with torch.no_grad():
        s = torch.randn(n, 2)  # 随机状态
        z = torch.randn(n, 4)  # 从标准正态采样潜变量
        actions = model.decode(s, z)
        print("生成的动作序列:\n", actions)

if __name__ == "__main__":
    cvae_model = train_cvae()
    generate_actions(cvae_model, n=5)
