import os
import subprocess
import sys
from pathlib import Path


def test_dense_label_script_help_works_outside_repository(tmp_path):
    project_root = Path(__file__).parents[1].resolve()
    script = project_root / "datagen" / "generate_dense_label.py"
    python_paths = []
    for entry in sys.path:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if resolved != project_root and project_root not in resolved.parents:
            python_paths.append(entry)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--vol-id" in result.stdout
    assert "--output-root" in result.stdout
