import torch
from torch.optim.optimizer import Optimizer


class BB(Optimizer):
    """Implements BB algorithm.
    It has been proposed in `Barzilai-Borwein-based Adaptive Learning Rate for Deep Learning`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): initial learning rate (default: 1e-1)
        steps (int, optional): how many iterations before calculating bb step size (default: 400)
        beta (float, optional): coefficients used for computing running averages of gradient (default: 0.01)
        min_lr (float, optional): minimal learning rate (default: 1e-1)
        max_lr (float, optional): maximal learning rate (default: 10.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.)
    .. Barzilai-Borwein-based Adaptive Learning Rate for Deep Learning:
        http://www.sciencedirect.com/science/article/pii/S0167865519302429
    """

    def __init__(self,
                 params,
                 lr=1e-1,
                 steps=400,
                 beta=0.01,
                 min_lr=1e-1,
                 max_lr=10.0,
                 weight_decay=0.,
                 ):
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert 0.0 < beta <= 1.0, ValueError("Invalid beta value: {}".format(beta))
        assert min_lr > 0.0, ValueError("Invalid minimal learning rate: {}".format(min_lr))
        assert max_lr > min_lr, ValueError("Invalid maximal learning rate: {}".format(max_lr))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            steps=int(steps),
            beta=beta,
            min_lr=min_lr,
            max_lr=max_lr,
        )

        super(BB, self).__init__(params, defaults)

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the net
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")
        group = self.param_groups[0]
        # register the global state of BB as state for the first param
        state = self.state[self._params[0]]
        state.setdefault('bb_iter', -1)
        state.setdefault('n_iter', -1)

        state['n_iter'] += 1
        if state['n_iter'] % group['steps'] == 0:
            state['bb_iter'] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0
            sum_dg_norm = 0

            for p in self._params:
                if state['n_iter'] == 0:
                    with torch.no_grad():
                        self.state[p]['grad_aver'] = torch.zeros_like(p)
                        self.state[p]['grads_prev'] = torch.zeros_like(p)
                        self.state[p]['params_prev'] = torch.zeros_like(p)

                if state['bb_iter'] > 1:
                    params_diff = p.detach() - self.state[p]['params_prev']
                    grads_diff = self.state[p]['grad_aver'] - self.state[p]['grads_prev']
                    sum_dp_dg += (grads_diff * params_diff).sum().item()
                    sum_dp_norm += params_diff.norm().item() ** 2
                    sum_dg_norm += grads_diff.norm().item() ** 2

                if state['bb_iter'] > 0:
                    self.state[p]['grads_prev'].copy_(self.state[p]['grad_aver'])
                    self.state[p]['params_prev'].copy_(p.detach())
                    self.state[p]['grad_aver'].zero_()

            if state['bb_iter'] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * group['steps'])
                    lr_scaled = abs(lr_hat) * (state['bb_iter'] + 1)
                    if (lr_scaled > group['max_lr']) or (lr_scaled < group['min_lr']):
                        lr = 1.0 / (state['bb_iter'] + 1)
                    else:
                        lr = abs(lr_hat)

                    group['lr'] = lr

        for p in self._params:

            if p.grad is None:
                continue
            d_p = p.grad.data
            if group['weight_decay'] != 0:
                d_p.add_(group['weight_decay'], p.data)

            # update gradients
            p.data.add_(-group['lr'], d_p)

            # average the gradients
            with torch.no_grad():
                self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], d_p)

        return loss
