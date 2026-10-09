import subprocess
import sys


def test_ultralytics_cfg_package_can_run_as_module():
    result = subprocess.run(
        [sys.executable, "-m", "ultralytics.cfg", "cfg"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "baseline_monitor" in result.stdout
