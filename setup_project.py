"""
setup_project.py
Creates the full folder structure for the SamriddhiAI project.
Run once from the SamriddhiAI/ directory: python setup_project.py
"""

from pathlib import Path

# All folders we want to exist under the project root
FOLDERS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/chatbot",
    "src/guardrails",
    "src/api",
    "app",
    "models",
    "docs",
    "tests",
    "outputs",
    "outputs/figures",
    "outputs/reports",
]

# Empty __init__.py files so Python treats src/ as a package
INIT_FILES = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/chatbot/__init__.py",
    "src/guardrails/__init__.py",
    "src/api/__init__.py",
    "tests/__init__.py",
]


def create_folders():
    root = Path.cwd()
    print(f"Project root: {root}\n")
    for folder in FOLDERS:
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {folder}/")


def create_init_files():
    root = Path.cwd()
    print()
    for init in INIT_FILES:
        path = root / init
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("")
            print(f"  [OK] {init}")
        else:
            print(f"  [SKIP] {init} (already exists)")


def create_gitignore():
    root = Path.cwd()
    gitignore = root / ".gitignore"
    if gitignore.exists():
        print(f"\n  [SKIP] .gitignore (already exists)")
        return
    content = """# Data
data/raw/*
data/interim/*
data/processed/*
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep

# Models
models/*
!models/.gitkeep

# Outputs
outputs/*
!outputs/.gitkeep

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.ipynb_checkpoints/

# Env
.env

# IDE
.vscode/*
!.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db
"""
    gitignore.write_text(content)
    print(f"\n  [OK] .gitignore")


def create_gitkeeps():
    """Add .gitkeep so empty folders are tracked by git."""
    root = Path.cwd()
    keep_dirs = [
        "data/raw", "data/interim", "data/processed",
        "models", "outputs", "outputs/figures", "outputs/reports",
    ]
    print()
    for d in keep_dirs:
        keep = root / d / ".gitkeep"
        if not keep.exists():
            keep.write_text("")
            print(f"  [OK] {d}/.gitkeep")


if __name__ == "__main__":
    print("=" * 60)
    print("Setting up SamriddhiAI project structure")
    print("=" * 60)
    create_folders()
    create_init_files()
    create_gitignore()
    create_gitkeeps()
    print("\n" + "=" * 60)
    print("Done. Structure created.")
    print("=" * 60)