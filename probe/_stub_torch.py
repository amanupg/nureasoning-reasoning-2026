"""Minimal torch/PIL stubs so benchmark.py imports without the training stack.

benchmark.py pulls in nuvla.trajectory_provider (model code) at module import,
but the scoring functions themselves are pure numpy/scipy/shapely. Stubbing lets
us exercise the UNMODIFIED devkit scorer instead of a reimplementation.
"""
import sys, types

def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()

for name in ("torch", "torch.nn", "torch.nn.functional", "torch.utils",
             "torch.utils.data", "torch.distributed", "torch.optim",
             "torchvision", "torchvision.transforms", "transformers"):
    m = _mod(name)
    m.__getattr__ = lambda n, _m=m: _Any()
sys.modules["torch"].Tensor = _Any
sys.modules["torch"].device = _Any
sys.modules["torch.utils.data"].Dataset = _Any
sys.modules["torch.utils.data"].DataLoader = _Any
sys.modules["torch.nn"].Module = _Any
