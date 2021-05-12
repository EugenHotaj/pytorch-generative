from torch import distributions
from torch import nn
from torch import optim
from pytorch_generative import trainer

# TODO(eugenhotaj): Add docs, tests, etc.


class GaussianMixtureModel(nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.mixture_coef = nn.Parameter(torch.ones((n_components,)))
        self.mean = nn.Parameter(torch.randn(n_components, n_features) * 0.01)
        self.log_std = nn.Parameter(torch.zeros(n_components, n_features))

    def _gaussian_log_prob(self, x):
        z = -self.log_std - 0.5 * torch.log(torch.tensor(2 * np.pi))
        log_prob = z - 0.5 * ((x.unsqueeze(1) - self.mean) / self.log_std.exp()) ** 2
        return log_prob.sum(-1)

    def sample(self, n_samples):
        with torch.no_grad():
            idxs = distributions.Categorical(logits=self.mixture_coef).sample(
                (n_samples,)
            )
            mean, std = self.mean[idxs], self.log_std[idxs].exp()
            return distributions.Normal(mean, std).sample((1,)).squeeze()

    def forward(self, x):
        mixing_log_prob = torch.log_softmax(self.mixture_coef, dim=-1)
        log_prob = mixing_log_prob + self._gaussian_log_prob(x)
        return torch.logsumexp(log_prob, dim=-1)
