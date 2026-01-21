#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


# -------------------------
# CONFIG (set these)
# -------------------------
GITHUB_OWNER = "marksocaciu"
GITHUB_REPO = "Spectra-Processing-Tool"

# If your default branch is not set correctly on GitHub, you can hardcode it here,
# otherwise it will be discovered via the repo API.
HARDCODE_DEFAULT_BRANCH: str | None = None

# Entry point to run after install (recommended: package + main()).
# Example below assumes: from spectra_processing.app import main; main()
RUN_COMMAND = ["-c", "from spectra_processing.app import main; main()"]


# -------------------------
# PATHS (user-accessible)
# -------------------------
DOCS_ROOT = Path.home() / "Documents" / "Spectra Processing"
DATA_DIR = DOCS_ROOT / "Executable"  # keep as-is
APP_ROOT = DOCS_ROOT / "App"
VERSIONS_DIR = APP_ROOT / "versions"
CURRENT_FILE = APP_ROOT / "current.json"


def is_windows() -> bool:
    return os.name == "nt"


def http_json(url: str) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SpectraProcessingBootstrap/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_bytes(url: str) -> bytes:
    headers = {"User-Agent": "SpectraProcessingBootstrap/1.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    APP_ROOT.mkdir(parents=True, exist_ok=True)


def read_current() -> dict | None:
    if not CURRENT_FILE.exists():
        return None
    return json.loads(CURRENT_FILE.read_text(encoding="utf-8"))


def write_current(commit_sha: str, install_path: Path) -> None:
    CURRENT_FILE.write_text(
        json.dumps({"sha": commit_sha, "path": str(install_path)}, indent=2),
        encoding="utf-8",
    )


def extract_zip(zip_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)


def repo_api_url(path: str) -> str:
    return f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}{path}"


def get_default_branch() -> str:
    if HARDCODE_DEFAULT_BRANCH:
        return HARDCODE_DEFAULT_BRANCH
    repo = http_json(repo_api_url(""))
    return repo["default_branch"]


def get_latest_commit_sha(branch: str) -> str:
    # Most recent commit on branch
    commits = http_json(repo_api_url(f"/commits?sha={branch}&per_page=1"))
    if not commits:
        raise RuntimeError(f"No commits found for branch '{branch}'.")
    return commits[0]["sha"]


def install_from_commit_zip(sha: str) -> Path:
    """
    Downloads the zipball for a specific commit SHA and installs it into:
      ~/Documents/Spectra Processing/App/versions/<sha>/
    """
    target = VERSIONS_DIR / sha
    if target.exists():
        return target

    zip_url = repo_api_url(f"/zipball/{sha}")
    zip_bytes = download_bytes(zip_url)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "src"
        extract_zip(zip_bytes, tmp)

        # GitHub zipball contains one top-level directory with a generated name
        roots = [p for p in tmp.iterdir() if p.is_dir()]
        if not roots:
            raise RuntimeError("Zipball did not contain an extracted directory.")
        extracted_root = roots[0]

        shutil.copytree(extracted_root, target)

    return target


def ensure_venv(project_dir: Path) -> Path:
    """
    Creates a venv inside the installed version folder.
    """
    venv_dir = project_dir / ".venv"
    py = venv_dir / ("Scripts/python.exe" if is_windows() else "bin/python")

    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

    return py


def pip_install_project(py: Path, project_dir: Path) -> None:
    """
    Installs the downloaded project according to pyproject.toml.
    Non-editable install is correct for an installed copy.
    """
    subprocess.run([str(py), "-m", "pip", "install", "."], cwd=str(project_dir), check=True)


def seed_data_if_missing(installed_project_dir: Path) -> None:
    """
    Optional: seed fluorophor_data.txt into the user-accessible DATA_DIR if missing.
    Never overwrite user data.
    """
    dest = DATA_DIR / "fluorophor_data.txt"
    if dest.exists():
        return

    # Prefer packaged resource path if you place it there
    candidate1 = installed_project_dir / "src" / "spectra_processing" / "resources" / "fluorophor_data.txt"
    candidate2 = installed_project_dir / "fluorophor_data.txt"

    src = candidate1 if candidate1.exists() else candidate2
    if src.exists():
        shutil.copy2(src, dest)
        print(f"Seeded data file: {dest}")
    else:
        print("WARNING: Could not find fluorophor_data.txt to seed user data directory.")


def run_app(py: Path) -> int:
    """
    Runs the app using the installed environment.
    Adjust RUN_COMMAND to match your actual entry point.
    """
    cmd = [str(py)] + RUN_COMMAND
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install/update Spectra Processing from the latest commit zipball.")
    parser.add_argument("--no-update", action="store_true", help="Skip update check; run currently installed version.")
    parser.add_argument("--force", action="store_true", help="Reinstall even if the latest SHA matches current.")
    args = parser.parse_args()

    ensure_dirs()

    current = read_current()
    current_sha = current["sha"] if current else None
    current_path = Path(current["path"]) if current else None

    if not args.no_update:
        branch = get_default_branch()
        latest_sha = get_latest_commit_sha(branch)

        needs_install = args.force or (current_sha != latest_sha) or (not current_path) or (not current_path.exists())
        if needs_install:
            print(f"Installing latest commit {latest_sha} from branch '{branch}'...")
            installed_dir = install_from_commit_zip(latest_sha)
            py = ensure_venv(installed_dir)
            pip_install_project(py, installed_dir)
            seed_data_if_missing(installed_dir)
            write_current(latest_sha, installed_dir)
            return run_app(py)

        print(f"Already up-to-date at commit {current_sha}.")
        # Run current
        installed_dir = current_path
        py = ensure_venv(installed_dir)
        return run_app(py)

    # no-update path
    if not current_path or not current_path.exists():
        raise SystemExit("No installed version found. Run without --no-update to install the latest commit.")

    py = ensure_venv(current_path)
    return run_app(py)


if __name__ == "__main__":
    raise SystemExit(main())
