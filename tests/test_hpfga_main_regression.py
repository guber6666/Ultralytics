import torch

from ultralytics.nn.extra_modules.module.HPFGA import HPFGA


def test_hpfga_forward_handles_batched_inputs_used_by_main_demo():
    module = HPFGA(16, 32)
    inputs = torch.randn(2, 16, 32, 32)

    outputs = module(inputs)

    assert outputs.shape == (2, 32, 32, 32)
