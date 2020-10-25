"""Extra optimizers not (yet) implemented in PyTorch.

References (used throughout the code):
  [1]: https://arxiv.org/abs/2010.07468
"""

import torch
from torch import optim


class AdaBelief(optim.Optimizer):
    """Implementation of the AdaBelief algorithm proposed in [1].

    NOTE: This is a minimal implementaiton which does not support more advanced
    features of Adam (e.g. AdamW, RAdam, etc).
    """

    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999)):
      """Initializes a new AdaBelief instance.

      Args:
        params: Iterable of parameters to optimize or dicts defining parameter
          groups.
        lr: Learning rate.
        betas: Coeficients for the exponential moving averages (EMAs) of the
          gradients and the squared gradients.
      """
      assert 0 <= lr, f"Invalid learning rate: {lr}"
      assert 0 <= betas[0] < 1, f"Invalid beta parameter at index 0: {betas[0]}"
      assert 0 <= betas[1] < 1, f"Invalid beta parameter at index 1: {betas[1]}"
      super().__init__(params,  dict(lr=lr, betas=betas))

    def step(self):
      for group in self.param_groups:
        beta1, beta2 = group['betas']
        for param in group['params']:
          if param.grad is None:
            continue
          grad = param.grad.data
          assert not grad.is_sparse, 'AdaBelief does not support sparse gradients.'
          
          # Create the state if this is the first step.
          state = self.state[param]
          if not state:
            state['step'] = 0
            state['ema_avg'] = torch.zeros_like(param.data)
            state['ema_var'] = torch.zeros_like(param.data)
          ema_avg, ema_var = state['ema_avg'], state['ema_var']
          state['step'] += 1
 
          # Update first and second moment running averages.
          ema_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
          ema_var.mul_(beta2).addcmul_(grad - ema_avg, grad - ema_avg, value=1 - beta2).add_(1e-10)
          
          # Bias correct the running averages.
          ema_avg_ = ema_avg / (1 - beta1 ** state['step'])
          ema_var_ = ema_var / (1 - beta2 ** state['step'])

          # Update parameters.
          param.data.addcdiv_(-ema_avg_, ema_var_.sqrt() + 1e-10, value=group['lr'])
