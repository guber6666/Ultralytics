import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "tests/tmp/extra_modules_half_main_report.json"
PYTHON = "/root/code/conda/env/pytorch_2_8_0_py311/bin/python"


RUNNER = r"""
import json
import runpy
import sys
import time
import traceback

import torch

path = sys.argv[1]
orig_default = torch.get_default_dtype()
orig_tensor_to = torch.Tensor.to
orig_module_to = torch.nn.Module.to


def wants_cuda(args, kwargs):
    vals = list(args)
    if "device" in kwargs:
        vals.append(kwargs["device"])
    for v in vals:
        if isinstance(v, torch.device) and v.type == "cuda":
            return True
        if isinstance(v, str) and v.startswith("cuda"):
            return True
    return False


def patched_tensor_to(self, *args, **kwargs):
    out = orig_tensor_to(self, *args, **kwargs)
    if out.is_floating_point() and wants_cuda(args, kwargs):
        if kwargs.get("dtype") is None and out.dtype != torch.float16:
            out = out.half()
    return out


def patched_module_to(self, *args, **kwargs):
    out = orig_module_to(self, *args, **kwargs)
    if wants_cuda(args, kwargs) and kwargs.get("dtype") is None:
        out = out.half()
    return out


start = time.time()
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for half test")
    torch.set_default_dtype(torch.float16)
    torch.Tensor.to = patched_tensor_to
    torch.nn.Module.to = patched_module_to
    runpy.run_path(path, run_name="__main__")
    payload = {
        "status": "PASS",
        "error_type": None,
        "error_message": None,
        "elapsed": round(time.time() - start, 3),
    }
except Exception as e:
    payload = {
        "status": "FAIL",
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc(),
        "elapsed": round(time.time() - start, 3),
    }
finally:
    torch.set_default_dtype(orig_default)
    torch.Tensor.to = orig_tensor_to
    torch.nn.Module.to = orig_module_to

print(json.dumps(payload, ensure_ascii=False))
"""


def load_failed_records():
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return [r for r in payload["records"] if r["status"] != "PASS"]


def classify_failure(record):
    message = record.get("error_message") or ""
    category = record.get("category") or ""
    path = record["module_path"]

    if category == "FAIL_DTYPE_MISMATCH":
        return "dtype_mismatch"
    if category == "FAIL_HALF_OP":
        return "half_op"
    if category == "FAIL_IMPORT" or category == "FAIL_OOM":
        return "script_misc"
    if "cuFFT only supports dimensions whose sizes are powers of two" in message:
        return "fft_half_limit"
    if "mat1 and mat2" in message:
        return "matmul"
    if "expected scalar type Float but found Half" in message:
        return "hard_float"
    if "ComplexHalf" in message or "ComplexFloat" in message:
        return "script_misc"
    if path.endswith("DSA.py") or path.endswith("TransMixer.py") or path.endswith("csm_tritonk2.py"):
        return "script_misc"
    return "script_misc"


def make_param(record):
    mark_name = classify_failure(record)
    mark = getattr(pytest.mark, mark_name)
    path = record["module_path"]
    return pytest.param(path, marks=mark, id=f"{mark_name}:{Path(path).stem}")


FAILED_MODULE_PARAMS = [make_param(record) for record in load_failed_records()]


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="half regression report not found")
@pytest.mark.skipif(not subprocess.run([PYTHON, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True, check=False).stdout.strip().endswith("True"), reason="CUDA is required")
@pytest.mark.parametrize("module_path", FAILED_MODULE_PARAMS)
def test_failed_extra_module_main_half_regressions(module_path: str):
    completed = subprocess.run(
        [PYTHON, "-c", RUNNER, str(ROOT / module_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
        timeout=180,
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert output_lines, f"no runner output for {module_path}\nstderr:\n{completed.stderr}"
    payload = json.loads(output_lines[-1])
    assert payload["status"] == "PASS", (
        f"{module_path} still fails in half mode\n"
        f"error_type={payload.get('error_type')}\n"
        f"error_message={payload.get('error_message')}\n"
        f"stderr_tail={completed.stderr[-2000:]}"
    )
