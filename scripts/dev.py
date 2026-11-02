#!/usr/bin/env python3
"""Inicia API e interface locais; Ctrl+C encerra os dois serviços."""

import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    python = ROOT / "backend/.venv/bin/python"
    vite = ROOT / "frontend/node_modules/vite/bin/vite.js"
    node = shutil.which("node")
    if not python.exists() or not vite.exists() or node is None:
        raise SystemExit("Instale as dependências primeiro: python3 scripts/setup.py")

    processes: list[subprocess.Popen] = []

    def stop(_signum=None, _frame=None):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        processes.append(subprocess.Popen([
            str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000",
        ], cwd=ROOT / "backend"))
        processes.append(subprocess.Popen([
            node, str(vite), "--host", "127.0.0.1", "--port", "5173", "--strictPort",
        ], cwd=ROOT / "frontend"))
        print(
            "\nANN Lab: http://127.0.0.1:5173\n"
            "API: http://127.0.0.1:8000/docs\nCtrl+C para encerrar.\n",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.3)
        raise SystemExit(
            "Um serviço parou. Confira os logs e se as portas 5173/8000 estão livres."
        )
    except KeyboardInterrupt:
        print("\nEncerrando o ANN Lab…", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
