from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                if not state:
                    state = {'t': 0,
                             'betas': [group["betas"][0], group["betas"][1]],
                             'm_t': torch.zeros(grad.size()),
                             'v_t': torch.zeros(grad.size())}
                betas = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                m_t_prev = state['m_t']
                v_t_prev = state['v_t']
                m_t_prev = m_t_prev.to(device)
                v_t_prev = v_t_prev.to(device)
                m_t = betas[0] * m_t_prev + (1 - betas[0]) * grad
                v_t = betas[1] * v_t_prev + (1 - betas[1]) * grad * grad
                alpha_t = alpha * math.sqrt(1 - state['betas'][1]) / (1 - state['betas'][0])
                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)
                p.data -= alpha * weight_decay * p.data

                state['t'] += 1
                state['betas'] = [state['betas'][0] * betas[0], state['betas'][1] * betas[1]]
                state['m_t'] = m_t
                state['v_t'] = v_t
                self.state[p] = state

                # ### TODO
                # raise NotImplementedError
        return loss
