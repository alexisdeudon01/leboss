# start_env.py
# Lance avec le python du venv:
#   .\Scripts\python.exe .\start_env.py

from __future__ import annotations
import os
import sys
from pathlib import Path

def print_tree(root: Path, max_depth: int = 3, max_items_per_dir: int = 200) -> None:
    root = root.resolve()
    print(f"\n=== Arborescence: {root} (max_depth={max_depth}) ===")
    if not root.exists():
        print("[!] Dossier introuvable")
        return

    def iter_dir(p: Path):
        try:
            return sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return []

    def walk(p: Path, depth: int):
        if depth > max_depth:
            return
        items = iter_dir(p)
        if len(items) > max_items_per_dir:
            items = items[:max_items_per_dir]
            truncated = True
        else:
            truncated = False

        prefix = "  " * depth
        for it in items:
            marker = "/" if it.is_dir() else ""
            print(f"{prefix}- {it.name}{marker}")
            if it.is_dir():
                walk(it, depth + 1)

        if truncated:
            print(f"{prefix}... (truncated, too many items)")

    walk(root, 0)

def main() -> int:
    here = Path(__file__).resolve().parent  # envMaster
    # ROOT projet = 2 niveaux au-dessus: Request\Python\envMaster -> Request
    project_root = (here / ".." / "..").resolve()

    # Set PYTHONPATH pour ce process (et children) + ajoute au sys.path
    os.environ["PYTHONPATH"] = str(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("sys.executable :", sys.executable)
    print("VIRTUAL_ENV    :", os.environ.get("VIRTUAL_ENV"))
    print("PYTHONPATH     :", os.environ.get("PYTHONPATH"))
    print("ROOT (project) :", project_root)
    print("HERE (script)  :", here)

    # Vérif que le ROOT est bien dans sys.path
    print("ROOT in sys.path ?", str(project_root) in sys.path)

    # Affiche l'arborescence du dossier où se trouve le script
    print_tree(here, max_depth=2)
    # Affiche l'arborescence du ROOT projet
    print_tree(project_root, max_depth=2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
