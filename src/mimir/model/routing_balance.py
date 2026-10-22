"""Step-scoped, distributed auxiliary-loss-free expert load balancing."""
from contextvars import ContextVar
from functools import wraps
import inspect
import torch
import torch.distributed as dist

_ACTIVE = ContextVar('mimir_routing_balance', default=None)


@torch.compiler.disable
@torch.no_grad()
def record_load(router, mask):
    session = _ACTIVE.get()
    # Recomputed activations are not new routing decisions. This also excludes
    # reentrant and non-reentrant checkpoint replay inside backward.
    if session is not None and router.training and torch._C._current_graph_task_id() == -1:
        session.counts[router].add_(mask.detach().reshape(-1, router.n).sum(0))


class BalanceStep:
    """Accumulate real forwards across GAS; commit once after a successful step."""
    def __init__(self, model):
        self.counts = {m: torch.zeros(m.n, dtype=torch.int64, device=m.selection_bias.device)
                       for m in model.modules() if getattr(m, 'balance_enabled', False)}

    def __enter__(self):
        if _ACTIVE.get() is not None:
            raise RuntimeError('Nested routing balance steps are unsupported')
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.reset(self.token)

    @torch.no_grad()
    def commit(self):
        for router, counts in self.counts.items():
            # Every rank participates, including ranks with no selected tokens.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            load = counts.to(torch.float32)
            router.selection_bias.add_(router.balance_update_rate * torch.sign(load.mean() - load))


def balanced_optimizer_step(fn):
    """Keep balance state outside compiled forward/backward and optimizer EMA."""
    signature = inspect.signature(fn)
    @wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        state = bound.arguments['train_state']
        if not any(getattr(m, 'balance_enabled', False) for m in state.model.modules()):
            return fn(*args, **kwargs)
        with BalanceStep(state.model) as step:
            result = fn(*args, **kwargs)
            skipped = result[1] if fn.__name__ == 'train_accumulated_batches' else False
            if bound.arguments.get('optimizer_step_enabled', True) and not skipped:
                step.commit()
            return result
    return wrapped
