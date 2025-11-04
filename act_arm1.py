import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class CVAEEncoder(nn.Module):
    def __init__(self,obs_dim,act_dim,latent_dim,chunk_size):
        super().__init__()
        self.fc1=nn.Linear((obs_dim+act_dim)*chunk_size,256)
        self.fc_mu=nn.Linear(256,latent_dim)
        self.fc_logvar=nn.Linear(256,latent_dim)

    def forward(self,obs_chunk,act_chunk):
        x=torch.cat([obs_chunk.reshape(obs_chunk.size(0),-1),act_chunk.reshape(act_chunk.size(0),-1)],dim=-1)
        h=F.relu(self.fc1(x))
        mu=self.fc_mu(h)
        logvar=self.fc_logvar(h)
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        z=mu+eps*std
        return z,mu,logvar

class TransformerDecoder(nn.Module):
    def __init__(self,obs_dim,act_dim,latent_dim,nhead=4,num_layer=3):
        super().__init__()
        self.embedding=nn.Linear(obs_dim+latent_dim,128)
        encoder_layer=nn.TransformerEncoderLayer(d_model=128,nhead=nhead,dim_feedforward=256)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=num_layer)
        self.fc_out=nn.Linear(128,act_dim)

    def forward(self,obs_seq,z):
        seq_len, B= obs_seq.size(0), obs_seq.size(1)
        z_expanded=z.unsqueeze(0).expand(seq_len,B,z.size(-1))
        inp=torch.cat([obs_seq,z_expanded],dim=-1)
        inp=self.embedding(inp)
        mask=nn.Transformer.generate_square_subsequent_mask(seq_len).to(obs_seq.device)
        h=self.transformer(inp,mask=mask)
        actions=self.fc_out(h)
        return actions
    
def train_act(expert_path="expert_data.npz",epochs=10,batch_size=32,chunk_size=10):
    data = np.load(expert_path, allow_pickle=True)["trajectories"]
    obs_dim = data[0]["obs"][0].shape[0]
    act_dim = data[0]["actions"][0].shape[0]

    encoder = CVAEEncoder(obs_dim, act_dim, latent_dim=16, chunk_size=chunk_size)
    decoder = TransformerDecoder(obs_dim, act_dim, latent_dim=16)

    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(epochs):
        for traj in data:
            obs=np.array(traj["obs"])
            acts=np.array(traj["actions"])

            if len(obs) >= chunk_size:
                idx = np.random.randint(0, len(obs) - chunk_size + 1)
                obs_chunk = torch.tensor(obs[idx:idx+chunk_size], dtype=torch.float32).unsqueeze(1)
                act_chunk = torch.tensor(acts[idx:idx+chunk_size], dtype=torch.float32).unsqueeze(1)
                z, mu, logvar = encoder(obs_chunk, act_chunk)
                obs_seq = obs_chunk.permute(1, 0, 2)  
                pred_acts = decoder(obs_seq, z)      

                recon_loss = F.mse_loss(pred_acts.squeeze(1), act_chunk.squeeze(0))
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.01 * kl_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

        print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")





        