import copy
import tempfile

import pytest
import torch
from packaging.version import Version
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.prototype import low_bit_optim
from torchao.prototype.low_bit_optim.quant_utils import (
    quantize_8bit_with_qmap,
    quantize_4bit_with_qmap,
    _fp32_to_bf16_sr,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_3, TORCH_VERSION_AT_LEAST_2_4, TORCH_VERSION_AT_LEAST_2_6

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    import lpmm
except ImportError:
    lpmm = None

if torch.cuda.is_available():
    _DEVICES = ["cpu", "cuda"]
elif torch.xpu.is_available():
    _DEVICES = ["cpu", "xpu"]
else:
    _DEVICES = ["cpu"]


@pytest.mark.skipif(not torch.cuda.is_available() and not torch.xpu.is_available(), reason="optim CPU offload requires CUDA or XPU")
@pytest.mark.parametrize("offload_grad,grad_accum", [(False, 1), (False, 2), (True, 1)])
def test_optim_cpu_offload_correctness(offload_grad, grad_accum):
    device = _DEVICES[-1]
    model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
    model2 = copy.deepcopy(model1)

    optim1 = torch.optim.AdamW(model1.parameters())
    optim2 = low_bit_optim.CPUOffloadOptimizer(
        model2.parameters(), torch.optim.AdamW, offload_gradients=offload_grad, device=device
    )

    for _ in range(2):
        for _ in range(grad_accum):
            x = torch.randn(4, 32, device=device)
            model1(x).sum().backward()
            model2(x).sum().backward()

        optim1.step()
        optim1.zero_grad()

        optim2.step()
        optim2.zero_grad()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p2, p1)

@pytest.mark.skipif(not torch.cuda.is_available() and not torch.xpu.is_available(), reason="optim CPU offload requires CUDA or XPU")
def test_optim_cpu_offload_save_load():
    device = _DEVICES[-1]
    model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
    optim1 = low_bit_optim.CPUOffloadOptimizer(model1.parameters(), torch.optim.AdamW, device=device)

    for _ in range(2):
        x = torch.randn(4, 32, device=device)
        model1(x).sum().backward()
        optim1.step()
        optim1.zero_grad()

    # save checkpoint. make sure it can be serialized by torch.save()
    with tempfile.NamedTemporaryFile() as file:
        torch.save(optim1.state_dict(), file.name)
        state_dict = torch.load(file.name, map_location="cpu")

    # resume training
    model2 = copy.deepcopy(model1)
    optim2 = low_bit_optim.CPUOffloadOptimizer(model2.parameters(), torch.optim.AdamW, device=device)
    optim2.load_state_dict(state_dict)

    for _ in range(2):
        x = torch.randn(4, 32, device=device)

        model1(x).sum().backward()
        optim1.step()
        optim1.zero_grad()

        model2(x).sum().backward()
        optim2.step()
        optim2.zero_grad()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p2, p1)
