"""Extra optimizers not (yet) implemented in PyTorch.

References (used throughout the code):
  [1]: https://arxiv.org/abs/2010.07468
"""

import torch
from torch import optim


class AdaBelief(optim.Optimizer):
  """Implementation of the AdaBelief algorithm proposed in [1].

  NOTE: This is a minimal implementaiton which does not support weight decay
  and AMSGrad.

  AdaBelief can be viewed as a variation on the Adam optimizer where the running
  second moment of the gradient is replaced by the running variance of the 
  gradient. The running variance can be interpreted as the error of estimating
  the gradient by the running mean  and is used to appropriately scale the step
  size. A small error indicates a strong belief in the observation of the 
  gradient and justififies a large step size (and vice versa).

  More handwavily, the running variance can also be interpreted as taking into
  account the curvature of the loss function. If the loss function has low 
  curvature then successive gradients will have similar values and hence low
  variance (and vice versa).
  """

  def __init__(self, 
               params, 
               lr=1e-3, 
               betas=(0.9, 0.999)):
    """Initializes a new AdaBelief instance.

    Args:
      params: Iterable of parameters to optimize or dicts of parameter groups.
      lr: Learning rate.
      betas: Coeficients for the running mean and variance of the gradient.
    """
    assert 0 <= lr, f"Invalid learning rate: {lr}"
    assert 0 <= betas[0] < 1, f"Invalid beta parameter at index 0: {betas[0]}"
    assert 0 <= betas[1] < 1, f"Invalid beta parameter at index 1: {betas[1]}"
    super().__init__(params,  dict(lr=lr, betas=betas))

  def step(self):
    for group in self.param_groups:
      beta1, beta2, lr = group['betas'], group['lr']
      for param in group['params']:
        if param.grad is None:
          continue
        grad = param.grad.data
        assert not grad.is_sparse, 'AdaBelief does not support sparse gradients.'
        
        state = self.state[param]
        if not state:
          state['step'] = 0
          state['ema_avg'] = torch.zeros_like(param.data)
          state['ema_var'] = torch.zeros_like(param.data)
        ema_avg, ema_var = state['ema_avg'], state['ema_var']
        state['step'] += 1

        # Update running mean and variance.
        ema_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        error = grad - ema_avg
        ema_var.mul_(beta2).addcmul_(error, error, value=1 - beta2).add_(1e-10)
        
        # Bias correct the running mean and variance.
        ema_avg_ = ema_avg / (1 - beta1 ** state['step'])
        ema_var_ = ema_var / (1 - beta2 ** state['step'])

        aaram.data.addcdiv_(-ema_avg_, ema_var_.sqrt() + 1e-10, value=lr)

