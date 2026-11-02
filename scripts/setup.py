#!/usr/bin/env python3
"""Instala o ambiente local, incluindo um npm temporário se só houver Node.js."""

import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
NPM_VERSION = "10.9.2"


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def install_frontend(npm: list[str]) -> None:
    run([*npm, "ci"], FRONTEND)


def main() -> None:
    if sys.version_info < (3, 12):  # noqa: UP036 - diagnóstico do instalador em Python antigo
        raise SystemExit("Use Python 3.12 ou superior para a instalação do ANN Lab.")
    node = shutil.which("node")
    if not node:
        raise SystemExit("Instale Node.js 18.18+ (recomendado: Node.js 22 LTS) e tente novamente.")
    version = subprocess.check_output([node, "--version"], text=True).strip().lstrip("v")
    if tuple(int(part) for part in version.split(".")[:2]) < (18, 18):
        raise SystemExit("É necessário Node.js 18.18 ou superior.")

    environment = BACKEND / ".venv"
    if not (environment / "bin" / "python").exists():
        venv.create(environment, with_pip=True)
    run([
        str(environment / "bin" / "python"), "-m", "pip", "install", "-r", "requirements.txt",
    ], BACKEND)

    npm = shutil.which("npm")
    if npm:
        install_frontend([npm])
    else:
        print(
            "npm não encontrado. Usando npm temporário, sem alterar a instalação global.",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="ann-lab-npm-") as directory:
            archive = Path(directory) / "npm.tgz"
            url = f"https://registry.npmjs.org/npm/-/npm-{NPM_VERSION}.tgz"
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as package:
                package.extractall(directory, filter="data")
            install_frontend([node, str(Path(directory) / "package/bin/npm-cli.js")])
    print("\nInstalação concluída. Execute: python3 scripts/dev.py", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError) as error:
        raise SystemExit(f"A instalação não foi concluída: {error}") from error
