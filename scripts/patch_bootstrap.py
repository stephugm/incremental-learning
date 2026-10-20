"""
Patch bootstrap cells in NB02/03/04 and regenerate NB00.
Run: python scripts/patch_bootstrap.py
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"

# ── New bootstrap source lines ─────────────────────────────────────────────────
NEW_BOOTSTRAP = [
    "import os, sys, subprocess\n",
    "from pathlib import Path\n",
    "\n",
    'GITHUB_URL = "https://github.com/stephugm/incremental-learning.git"\n',
    "FORCE_DOWNLOAD = False\n",
    "\n",
    "def detect_env():\n",
    '    if "google.colab" in sys.modules: return "colab"\n',
    '    if os.path.exists("/kaggle/input"): return "kaggle"\n',
    '    return "local"\n',
    "\n",
    "ENV = detect_env()\n",
    'print(f"Environment: {ENV}")\n',
    "\n",
    'if ENV in ("colab", "kaggle"):\n',
    '    REPO = (Path("/content/incremental-learning") if ENV == "colab"\n',
    '            else Path("/kaggle/working/incremental-learning"))\n',
    "    if not REPO.exists():\n",
    '        subprocess.run(["git", "clone", GITHUB_URL, str(REPO)], check=True)\n',
    "\n",
    "    # ── Step 1: install requirements (tvkit hanya untuk lokal / NB05) ────\n",
    '    subprocess.run(["pip", "install", "-q", "-r",\n',
    '                    str(REPO / "requirements.txt")], check=False)\n',
    "\n",
    "    # ── Step 2: salin parquet hasil NB05 (tvkit) dari Kaggle Dataset ──────\n",
    "    import shutil\n",
    '    KAGGLE_DS = Path("/kaggle/input/hblstm-index-15m-raw")\n',
    '    RAW_DIR = REPO / "data" / "raw"\n',
    "    RAW_DIR.mkdir(parents=True, exist_ok=True)\n",
    "    if KAGGLE_DS.exists():\n",
    '        for _f in KAGGLE_DS.glob("*.parquet"):\n',
    "            if not (RAW_DIR / _f.name).exists():\n",
    "                shutil.copy(_f, RAW_DIR / _f.name)\n",
    '            print(f"Dataset : {_f.name}")\n',
    '        print("tvkit parquet : OK  -> target 6000 bar dapat dicapai")\n',
    "    else:\n",
    '        print(f"tvkit parquet : {KAGGLE_DS} TIDAK ADA (Add Data di sidebar Kaggle)")\n',
    '        print("  -> fallback yfinance, hanya ~1500 bar (limit 60 hari)")\n',
    '        print("  -> n_bars=6000 di config TIDAK akan tercapai via yfinance")\n',
    "\n",
    "else:\n",
    "    REPO = Path().resolve()\n",
    '    while not (REPO / "configs" / "config.yaml").exists() and REPO != REPO.parent:\n',
    "        REPO = REPO.parent\n",
    "\n",
    "if str(REPO) not in sys.path:\n",
    "    sys.path.insert(0, str(REPO))\n",
    "\n",
    'print(f"Repo root: {REPO}")\n',
    "# autoreload: best-effort, skip gracefully on Python 3.12+ if imp removed\n",
    "try:\n",
    "    _ip = get_ipython()  # noqa: F821\n",
    "    _ip.run_line_magic('load_ext', 'autoreload')\n",
    "    _ip.run_line_magic('autoreload', '2')\n",
    "except Exception as _ar_e:\n",
    "    print(f'autoreload skipped: {_ar_e}')",
]


def _is_bootstrap_cell(cell: dict) -> bool:
    """True jika cell adalah bootstrap cell yang perlu di-patch."""
    if cell.get("cell_type") != "code":
        return False
    src = "".join(cell.get("source", []))
    return "requirements.txt" in src and "subprocess.run" in src


def patch_notebook(path: Path) -> int:
    """Patch bootstrap cells in one notebook. Returns number of cells patched."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb["cells"]:
        if _is_bootstrap_cell(cell):
            cell["source"] = NEW_BOOTSTRAP
            # clear stale outputs
            cell["outputs"] = []
            cell["execution_count"] = None
            n += 1
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


if __name__ == "__main__":
    targets = [
        "01_data_exploration.ipynb",
        "02_data_pipeline.ipynb",
        "03_hblstm_reproduction.ipynb",
        "04_results_vs_paper.ipynb",
    ]
    total = 0
    for name in targets:
        p = NB_DIR / name
        n = patch_notebook(p)
        print(f"  {name}: {n} cell(s) patched")
        total += n
    print(f"\nTotal cells patched: {total}")
