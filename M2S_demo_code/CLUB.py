import torch
import torch.nn as nn
import torch.nn.functional as F

class CLUBSample_group(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super(CLUBSample_group, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.p_mu = nn.Sequential(
            nn.Linear(self.x_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.y_dim)
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(self.x_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.y_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        mu = mu.reshape(-1, mu.shape[-1])
        logvar = logvar.reshape(-1, logvar.shape[-1])

        y_samples = y_samples.reshape(-1, y_samples.shape[-1])
        return (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0) / 2

    def mi_est(self, x_samples, y_samples): # x_samples: [batch_size, x_dim], y_samples: [batch_size, T, y_dim]
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]

        random_index = torch.randperm(sample_size).long()
        mu_exp1 = mu
        positive = - ((mu_exp1 - y_samples) ** 2).mean(dim=1) / logvar.mean(dim=1).exp()
        negative = - ((mu_exp1 - y_samples[random_index]) ** 2).mean(dim=1) / logvar.mean(dim=1).exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2