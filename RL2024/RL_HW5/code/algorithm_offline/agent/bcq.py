import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm_offline.network.actor import BCQ_Actor
from algorithm_offline.network.critic import Twin_Qnetwork


class BCQ:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed

        self.state_dim = self.args.state_dim
        self.action_dim = self.args.action_dim
        self.hidden_dim = self.args.hidden_dim
        self.action_clip = self.args.action_clip
        self.grad_norm_clip = self.args.grad_norm_clip

        self.gamma = args.gamma
        self.tau = args.tau
        self.lr = args.lr
        self.batch_size = args.batch_size_mf
        self.device = args.device

        if self.device == "gpu":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Networks
        self.actor_eval = BCQ_Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_clip).to(
            self.device
        )
        self.critic_eval = Twin_Qnetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.vae = VAE(self.state_dim, self.action_dim, self.hidden_dim).to(
            self.device
        )  # VAE for generating actions

        # Target networks
        self.actor_target = copy.deepcopy(self.actor_eval).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_eval).to(self.device)

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=self.lr)

        self.total_it = 0

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def select_action(self, state):
        # Generate candidate actions from VAE
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action = self.vae.decode(state)
            # Perturb actions using actor network
            perturbed_action = self.actor_eval(state, action)
            return perturbed_action.cpu().data.numpy().flatten()

    def update_critic(self, state, action, next_state, reward, done) -> float:
        with torch.no_grad():
            # Generate next actions using VAE and perturb
            next_action = self.vae.decode(next_state)
            next_action = self.actor_target(next_state, next_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic_eval(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.item()

    def update_actor(self, state) -> float:
        # Generate actions using VAE
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor_eval(state, sampled_actions)

        # Update actor to maximize Q-value
        q1, q2 = self.critic_eval(state, perturbed_actions)
        actor_loss = -torch.min(q1, q2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return actor_loss.item()

    def update_vae(self, state, action) -> float:
        recon, mean, std = self.vae(state, action)
        vae_loss = self.vae.loss_function(recon, action, mean, std)

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

        return vae_loss.item()

    def train(self, memory) -> tuple[float, float | None, float]:
        self.total_it += 1
        state, action, next_state, reward, done = memory.sample(self.batch_size)

        # VAE training
        vae_loss = self.update_vae(state, action)

        # Critic training
        critic_loss = self.update_critic(state, action, next_state, reward, done)

        # Actor training (delayed)
        actor_loss = None
        if self.total_it % 2 == 0:
            actor_loss = self.update_actor(state)
            # Update target networks
            self.soft_update()

        return critic_loss, actor_loss, vae_loss

    def soft_update(self):
        for param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, path):
        state_dict = {
            "actor_eval": self.actor_eval.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_eval": self.critic_eval.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }

        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.actor_eval.load_state_dict(state_dict["actor_eval"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.critic_eval.load_state_dict(state_dict["critic_eval"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(VAE, self).__init__()
        self.action_dim = action_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def encode(self, state, action):
        h = self.encoder(torch.cat([state, action], 1))
        return self.mean(h), self.log_std(h)

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.size(0), self.action_dim)).to(state.device)
        return self.decoder(torch.cat([state, z], 1))

    def forward(self, state, action):
        mean, log_std = self.encode(state, action)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        return self.decode(state, z), mean, std

    def loss_function(self, recon, action, mean, std):
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        return recon_loss + 0.5 * kl_loss
