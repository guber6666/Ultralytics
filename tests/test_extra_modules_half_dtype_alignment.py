import pytest
import torch

from ultralytics.nn.extra_modules import CirculantAttention, DSEBlock


CUDA_UNAVAILABLE = not torch.cuda.is_available()


@pytest.mark.skipif(CUDA_UNAVAILABLE, reason="CUDA is required")
def test_circulant_attention_supports_half_on_non_power_of_two_spatial_size():
    x = torch.randn(1, 64, 20, 20, device="cuda", dtype=torch.float16)
    module = CirculantAttention(64).cuda().half().eval()

    with torch.inference_mode():
        y = module(x)

    assert y.shape == x.shape
    assert y.dtype == torch.float16


@pytest.mark.skipif(CUDA_UNAVAILABLE, reason="CUDA is required")
def test_dseblock_keeps_half_dtype_through_selective_scan_path():
    x = torch.randn(1, 16, 32, 32, device="cuda", dtype=torch.float16)
    module = DSEBlock(16, 32).cuda().half().eval()

    with torch.inference_mode():
        y = module(x)

    assert y.shape == (1, 32, 32, 32)
    assert y.dtype == torch.float16
