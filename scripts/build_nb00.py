import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "notebooks" / "00_run_all.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

def md(i, src):
    return {"cell_type": "markdown", "id": i, "metadata": {}, "source": src}

def code(i, src):
    return {"cell_type": "code", "execution_count": None, "id": i,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# Title
cells.append(md("md-title", [
    "# 00 - Run All (NB02 + NB03 + NB04)\n",
    "\n",
    "**Satu notebook untuk dijalankan di Kaggle / Colab.**  \n",
    "Klik **Run All** - tidak perlu buka notebook lain.\n",
    "\n",
    "Sections:\n",
    "1. Bootstrap\n",
    "2. Imports & config\n",
    "3. [NB02] Data pipeline\n",
    "4. [NB03] H.BLSTM demo + full loop\n",
    "5. [NB04] Results vs paper\n",
]))

# 1. Bootstrap
cells.append(md("md-s1", ["## 1. Bootstrap"]))
cells.append(code("c-bootstrap", [
    "import os, sys, subprocess, json\n",
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
    "    # Step 1: install requirements (tvDatafeed sudah dipisah)\n",
    '    subprocess.run(["pip", "install", "-q", "-r",\n',
    '                    str(REPO / "requirements.txt")], check=False)\n',
    "\n",
    "    # Step 2: tvDatafeed best-effort dari GitHub\n",
    '    print("Installing tvDatafeed from GitHub ...")\n',
    "    try:\n",
    "        subprocess.run(\n",
    '            ["pip", "install", "-q",\n',
    '             "git+https://github.com/rongardF/tvdatafeed.git"],\n',
    "            capture_output=True, timeout=120,\n",
    "        )\n",
    "        from tvDatafeed import TvDatafeed  # noqa: F401\n",
    '        print("tvDatafeed : OK  -> target 6000 bar dapat dicapai")\n',
    "    except Exception as _e:\n",
    '        print(f"tvDatafeed : TIDAK TERSEDIA ({_e})")\n',
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
]))

# 2. Imports & config
cells.append(md("md-s2", ["## 2. Imports & config"]))
cells.append(code("c-imports", [
    "import yaml\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import torch\n",
    "\n",
    "from src.paths import Paths\n",
    "from src.data import build_dataset, IncrementalEMA\n",
    "from src.model import BiLSTMModel, get_device\n",
    "from src.hblstm import HBLSTMRunner\n",
    "from src.metrics import (\n",
    "    evaluate, compare_to_paper, summarise_runs,\n",
    "    mae, rmse, mape, mase, directional_accuracy, naive_persistence\n",
    ")\n",
    "\n",
    'sns.set_theme(style="whitegrid", palette="muted")\n',
    'plt.rcParams["figure.dpi"] = 110\n',
    "\n",
    "paths = Paths()\n",
    'with open(paths.configs / "config.yaml") as f:\n',
    "    cfg = yaml.safe_load(f)\n",
    "\n",
    'DEVICE       = get_device(cfg["runtime"]["device"])\n',
    'INDICES      = cfg["data"]["indices"]\n',
    'FEATURE_SETS = list(cfg["data"]["feature_sets"].keys())\n',
    'LAG          = cfg["data"]["lag"]\n',
    'TRAIN_RATIO  = cfg["data"]["split"]["train"]\n',
    'N_BARS       = cfg["data"]["n_bars"]\n',
    'EMA5_PERIOD  = cfg["indicators"]["ema5_period"]\n',
    'SEEDS        = cfg["training"]["seeds"]\n',
    "\n",
    "MODEL_KWARGS = {\n",
    '    "hidden_size": cfg["model"]["hidden_size"],\n',
    '    "num_layers":  cfg["model"]["num_layers"],\n',
    '    "dropout":     cfg["model"]["dropout"],\n',
    "}\n",
    "TRAINER_KWARGS = {\n",
    '    "lr":         cfg["training"]["learning_rate"],\n',
    '    "max_epochs": cfg["training"]["max_epochs"],\n',
    '    "batch_size": cfg["training"]["batch_size"],\n',
    '    "verbose":    False,\n',
    "}\n",
    "\n",
    'print(f"Device      : {DEVICE}")\n',
    'if DEVICE.type == "cuda":\n',
    '    print(f"GPU         : {torch.cuda.get_device_name(0)}")\n',
    "syms = [i['symbol'] for i in INDICES]\n",
    'print(f"Indices     : {syms}")\n',
    'print(f"Feature sets: {FEATURE_SETS}")\n',
    'print(f"Seeds       : {SEEDS}")',
]))

# 3. Data Pipeline (NB02)
cells.append(md("md-s3", [
    "---\n",
    "## 3. Data Pipeline (NB02)\n",
    "Download -> clean -> build datasets -> anti-leakage check -> scaler verification\n",
]))
cells.append(code("c-build-datasets", [
    "datasets = {}  # (symbol, feature_set) -> ds\n",
    "for idx_cfg in INDICES:\n",
    "    sym = idx_cfg['symbol']\n",
    "    for fset in FEATURE_SETS:\n",
    '        print(f"Building: {sym} | {fset} ...")\n',
    "        ds = build_dataset(\n",
    "            symbol=sym, exchange=idx_cfg['exchange'], tz=idx_cfg['tz'],\n",
    "            feature_set=fset, paths=paths,\n",
    "            n_bars=N_BARS, lag=LAG, train_ratio=TRAIN_RATIO,\n",
    "            force_download=FORCE_DOWNLOAD, ema5_period=EMA5_PERIOD,\n",
    "        )\n",
    "        datasets[(sym, fset)] = ds\n",
    "        print(f\"  X_train={ds['X_train'].shape}  X_test={ds['X_test'].shape}\")\n",
    'print(f"\\nTotal datasets: {len(datasets)}")',
]))
cells.append(code("c-antileak", [
    "print('Anti-leakage assertions:')\n",
    "all_pass = True\n",
    "for (sym, fset), ds in datasets.items():\n",
    "    overlap = set(ds['ts_train']) & set(ds['ts_test'])\n",
    "    ok = len(overlap) == 0\n",
    "    if not ok: all_pass = False\n",
    "    print(f\"  {'OK' if ok else 'FAIL'} {sym}/{fset}: overlap={len(overlap)}\")\n",
    "assert all_pass, 'LEAKAGE DETECTED!'\n",
    "print('All assertions passed.')",
]))
cells.append(code("c-dataset-summary", [
    "rows = []\n",
    "for (sym, fset), ds in datasets.items():\n",
    "    rows.append({\n",
    "        'Symbol': sym, 'Feature set': fset,\n",
    "        'Train bars': len(ds['y_train']),\n",
    "        'Test bars':  len(ds['y_test']),\n",
    "        'Features':   ds['X_train'].shape[2],\n",
    "        'Lag':        ds['X_train'].shape[1],\n",
    "    })\n",
    "df_summary = pd.DataFrame(rows)\n",
    "print(df_summary.to_string(index=False))\n",
    "df_summary.to_csv(paths.results / 'dataset_summary.csv', index=False)\n",
    "print(f\"Saved -> {paths.results / 'dataset_summary.csv'}\")",
]))
cells.append(code("c-scaler-check", [
    "print('Scaler verification (fit on train only):')\n",
    "for (sym, fset), ds in datasets.items():\n",
    "    sy = ds['scaler_y']\n",
    "    tmin, tmax = ds['y_train_raw'].min(), ds['y_train_raw'].max()\n",
    "    smin, smax = float(sy.data_min_[0]), float(sy.data_max_[0])\n",
    "    match = abs(tmin - smin) < 0.01 and abs(tmax - smax) < 0.01\n",
    "    print(f\"  {'OK' if match else 'MISMATCH'} {sym}/{fset}: \"\n",
    "          f\"train=[{tmin:.2f},{tmax:.2f}] scaler=[{smin:.2f},{smax:.2f}]\")",
]))
cells.append(code("c-ema5-check", [
    "ds_spx = datasets[('SPX', 'univariate')]\n",
    "close  = ds_spx['df_clean']['close'].values\n",
    "ema5i  = ds_spx['df_clean']['ema5'].values\n",
    "beta   = 2 / (5 + 1)\n",
    "manual = [close[0]]\n",
    "for i in range(1, 20):\n",
    "    manual.append(beta * close[i] + (1 - beta) * manual[-1])\n",
    "ok = all(abs(ema5i[i] - manual[i]) < 1e-4 for i in range(20))\n",
    "print(f\"EMA5 causal check: {'PASS' if ok else 'FAIL'}\")",
]))
cells.append(code("c-files-on-disk", [
    "print(f'Processed dir: {paths.processed}')\n",
    "for f in sorted(paths.processed.iterdir()):\n",
    "    print(f\"  {f.name:<40} {f.stat().st_size/1024:8.1f} KB\")",
]))


# 4. H.BLSTM Demo (NB03)
cells.append(md("md-s4", [
    "---\n",
    "## 4. H.BLSTM Reproduction (NB03)\n",
    "### 4a. Demo - SPX multivariate, seed[0]\n",
]))
cells.append(code("c-demo-run", [
    "DEMO_SYM  = 'SPX'\n",
    "DEMO_FSET = 'multivariate'\n",
    "DEMO_SEED = SEEDS[0]\n",
    "\n",
    "ds_demo = datasets[(DEMO_SYM, DEMO_FSET)]\n",
    "session_ids_demo = np.array(\n",
    "    ds_demo['df_clean'].loc[ds_demo['ts_test'], 'session_id'].values\n",
    ")\n",
    "runner_demo = HBLSTMRunner(\n",
    "    model_kwargs=MODEL_KWARGS,\n",
    "    trainer_kwargs={**TRAINER_KWARGS, 'verbose': True},\n",
    "    incremental_lr=cfg['training']['incremental_lr'],\n",
    "    update_policy='fixed',\n",
    "    checkpoint_every=cfg['runtime']['checkpoint_every_n_sessions'],\n",
    "    device=cfg['runtime']['device'],\n",
    "    seed=DEMO_SEED, verbose=True,\n",
    ")\n",
    "history = runner_demo.fit_initial(ds_demo['X_train'], ds_demo['y_train'])\n",
    "print(f\"Training done. Final loss: {history['train_loss'][-1]:.6f}\")",
]))
cells.append(code("c-demo-loss-plot", [
    "fig, ax = plt.subplots(figsize=(8, 3))\n",
    "ax.plot(history['train_loss'], lw=1)\n",
    "ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')\n",
    "ax.set_title(f'{DEMO_SYM} {DEMO_FSET} seed={DEMO_SEED} - training loss')\n",
    "ax.set_yscale('log'); plt.tight_layout()\n",
    "fig.savefig(paths.figures / f'{DEMO_SYM}_{DEMO_FSET}_s{DEMO_SEED}_train_loss.png', dpi=150)\n",
    "plt.show()",
]))
cells.append(code("c-demo-test", [
    "ckpt_demo = paths.checkpoint_path(DEMO_SYM, DEMO_SEED)\n",
    "df_preds_demo = runner_demo.run_test(\n",
    "    X_test=ds_demo['X_test'], y_test=ds_demo['y_test'],\n",
    "    y_test_raw=ds_demo['y_test_raw'],\n",
    "    timestamps=ds_demo['ts_test'],\n",
    "    session_ids=session_ids_demo,\n",
    "    scaler_y=ds_demo['scaler_y'],\n",
    "    checkpoint_path=ckpt_demo,\n",
    ")\n",
    "print(df_preds_demo.head())",
]))
cells.append(code("c-demo-plot", [
    "fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)\n",
    "axes[0].plot(df_preds_demo.index, df_preds_demo['y_true_raw'],\n",
    "             lw=0.5, label='Actual', color='steelblue')\n",
    "axes[0].plot(df_preds_demo.index, df_preds_demo['y_pred_raw'],\n",
    "             lw=0.5, label='Predicted', color='tomato', alpha=0.8)\n",
    "axes[0].set_title(f'{DEMO_SYM} - Actual vs Predicted')\n",
    "axes[0].set_ylabel('Price'); axes[0].legend()\n",
    "res = df_preds_demo['y_true_raw'] - df_preds_demo['y_pred_raw']\n",
    "axes[1].plot(df_preds_demo.index, res, lw=0.4, color='gray')\n",
    "axes[1].axhline(0, color='red', ls='--', lw=0.8)\n",
    "axes[1].set_ylabel('Residual')\n",
    "plt.tight_layout()\n",
    "fig.savefig(paths.figures / f'{DEMO_SYM}_{DEMO_FSET}_s{DEMO_SEED}_predictions.png', dpi=150)\n",
    "plt.show()",
]))
cells.append(code("c-demo-metrics", [
    "result_demo = evaluate(\n",
    "    df_preds_demo['y_true_raw'].values,\n",
    "    df_preds_demo['y_pred_raw'].values,\n",
    "    symbol=DEMO_SYM, feature_set=DEMO_FSET, seed=DEMO_SEED\n",
    ")\n",
    "print(f'Demo ({DEMO_SYM} {DEMO_FSET} seed={DEMO_SEED}):')\n",
    "for k in ['mae', 'rmse', 'mape', 'mase', 'da']:\n",
    "    print(f'  {k.upper():5s}: {result_demo[k]:.4f}')\n",
    "print()\n",
    "print('Paper: MAE=5.617  RMSE=8.574  MAPE=0.001  (SPX multivariate)')\n",
    "preds_path_demo = paths.preds_path(DEMO_SYM, DEMO_FSET, DEMO_SEED)\n",
    "df_preds_demo.to_parquet(preds_path_demo)\n",
    "print(f'Saved -> {preds_path_demo}')",
]))


# 4b. Full Loop
cells.append(md("md-s4b", [
    "### 4b. Full loop - 3 indeks x 2 feature sets x 3 seeds\n",
    "\n",
    "> **Estimasi (GPU T4):** ~30-60 menit.  \n",
    "> Prediksi di-cache ke parquet - aman jika session Kaggle terputus.\n",
]))
cells.append(code("c-full-loop", [
    "all_results = []\n",
    "\n",
    "for idx_cfg in INDICES:\n",
    "    sym = idx_cfg['symbol']\n",
    "    for fset in FEATURE_SETS:\n",
    "        print(f\"\\n{'='*55}\\n{sym} | {fset}\\n{'='*55}\")\n",
    "        ds = datasets[(sym, fset)]\n",
    "        sid_test = np.array(ds['df_clean'].loc[ds['ts_test'], 'session_id'].values)\n",
    "        for seed in SEEDS:\n",
    "            print(f'  seed={seed} ...', end=' ')\n",
    "            pp = paths.preds_path(sym, fset, seed)\n",
    "            if pp.exists():\n",
    "                print('(cached)'); df_preds = pd.read_parquet(pp)\n",
    "            else:\n",
    "                runner = HBLSTMRunner(\n",
    "                    model_kwargs=MODEL_KWARGS, trainer_kwargs=TRAINER_KWARGS,\n",
    "                    incremental_lr=cfg['training']['incremental_lr'],\n",
    "                    update_policy='fixed',\n",
    "                    checkpoint_every=cfg['runtime']['checkpoint_every_n_sessions'],\n",
    "                    device=cfg['runtime']['device'], seed=seed, verbose=False,\n",
    "                )\n",
    "                runner.fit_initial(ds['X_train'], ds['y_train'])\n",
    "                df_preds = runner.run_test(\n",
    "                    X_test=ds['X_test'], y_test=ds['y_test'],\n",
    "                    y_test_raw=ds['y_test_raw'], timestamps=ds['ts_test'],\n",
    "                    session_ids=sid_test, scaler_y=ds['scaler_y'],\n",
    "                    checkpoint_path=paths.checkpoint_path(sym, seed),\n",
    "                )\n",
    "                df_preds.to_parquet(pp); print('done')\n",
    "            res = evaluate(\n",
    "                df_preds['y_true_raw'].values, df_preds['y_pred_raw'].values,\n",
    "                symbol=sym, feature_set=fset, seed=seed\n",
    "            )\n",
    "            all_results.append(res)\n",
    "            print(f\"  MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}\"\n",
    "                  f\"  MAPE={res['mape']:.4f}  MASE={res['mase']:.3f}  DA={res['da']:.3f}\")\n",
    "\n",
    "print(f'\\nFull loop complete: {len(all_results)} runs')\n",
    "rj = paths.results / 'hblstm_all_results.json'\n",
    "with open(rj, 'w') as f: json.dump(all_results, f, indent=2)\n",
    "print(f'Saved -> {rj}')",
]))


# 5. Results vs Paper (NB04)
cells.append(md("md-s5", [
    "---\n",
    "## 5. Results vs Paper (NB04)\n",
    "### 5a. Reproduction comparison table\n",
]))
cells.append(code("c-cmp-multi", [
    "print('=' * 65)\n",
    "print('MULTIVARIATE - H.BLSTM vs Kalra et al. (2024)')\n",
    "print('=' * 65)\n",
    "multi_results = [r for r in all_results if r['feature_set'] == 'multivariate']\n",
    "cmp_df = compare_to_paper(multi_results, feature_set='multivariate')",
]))
cells.append(code("c-cmp-uni", [
    "print('=' * 65)\n",
    "print('UNIVARIATE - U_H.BLSTM reproduction')\n",
    "print('=' * 65)\n",
    "uni_results = [r for r in all_results if r['feature_set'] == 'univariate']\n",
    "cmp_uni_df = compare_to_paper(uni_results, feature_set='univariate')",
]))
cells.append(md("md-s5b", ["### 5b. MASE analysis (RQ1)"]))
cells.append(code("c-mase", [
    "symbols  = [i['symbol'] for i in INDICES]\n",
    "mase_rows = []\n",
    "for idx_cfg in INDICES:\n",
    "    sym = idx_cfg['symbol']\n",
    "    for fset in FEATURE_SETS:\n",
    "        for seed in SEEDS:\n",
    "            pp = paths.preds_path(sym, fset, seed)\n",
    "            if not pp.exists(): continue\n",
    "            df_p = pd.read_parquet(pp)\n",
    "            y_t = df_p['y_true_raw'].values\n",
    "            y_p = df_p['y_pred_raw'].values\n",
    "            y_naive = naive_persistence(y_t)\n",
    "            m = mase(y_t[1:], y_p[1:], y_naive=y_naive[1:])\n",
    "            mase_rows.append({'symbol': sym, 'feature_set': fset, 'seed': seed, 'mase': m})\n",
    "mase_df = pd.DataFrame(mase_rows)\n",
    "mase_summary = mase_df.groupby(['symbol', 'feature_set'])['mase'].agg(['mean', 'std'])\n",
    "print('MASE (mean +/- std across seeds):')\n",
    "print(mase_summary.to_string())",
]))
cells.append(code("c-mase-plot", [
    "x = np.arange(len(symbols)); w = 0.35\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "for j, fset in enumerate(FEATURE_SETS):\n",
    "    ms = [mase_df[(mase_df.symbol==s)&(mase_df.feature_set==fset)]['mase'].mean() for s in symbols]\n",
    "    ss = [mase_df[(mase_df.symbol==s)&(mase_df.feature_set==fset)]['mase'].std()  for s in symbols]\n",
    "    ax.bar(x + j*w, ms, w, label=fset, yerr=ss, capsize=4, alpha=0.8)\n",
    "ax.axhline(1.0,  color='red',   ls='--', lw=1.2, label='MASE=1.0 (no skill)')\n",
    "ax.axhline(0.85, color='green', ls=':',  lw=1.0, label='MASE=0.85 (H1 threshold)')\n",
    "ax.set_xticks(x + w/2); ax.set_xticklabels(symbols)\n",
    "ax.set_ylabel('MASE'); ax.set_title('MASE by index and feature set')\n",
    "ax.legend(); plt.tight_layout()\n",
    "fig.savefig(paths.figures / 'mase_comparison.png', dpi=150); plt.show()",
]))
cells.append(md("md-s5c", ["### 5c. Directional Accuracy (DA)"]))
cells.append(code("c-da", [
    "da_rows = []\n",
    "for idx_cfg in INDICES:\n",
    "    sym = idx_cfg['symbol']\n",
    "    for fset in FEATURE_SETS:\n",
    "        for seed in SEEDS:\n",
    "            pp = paths.preds_path(sym, fset, seed)\n",
    "            if not pp.exists(): continue\n",
    "            df_p = pd.read_parquet(pp)\n",
    "            dv = directional_accuracy(df_p['y_true_raw'].values, df_p['y_pred_raw'].values)\n",
    "            da_rows.append({'symbol': sym, 'feature_set': fset, 'seed': seed, 'da': dv})\n",
    "da_df = pd.DataFrame(da_rows)\n",
    "da_summary = da_df.groupby(['symbol', 'feature_set'])['da'].agg(['mean', 'std'])\n",
    "print('Directional Accuracy (mean +/- std):')\n",
    "print(da_summary.to_string())",
]))
cells.append(code("c-da-plot", [
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "for j, fset in enumerate(FEATURE_SETS):\n",
    "    ms = [da_df[(da_df.symbol==s)&(da_df.feature_set==fset)]['da'].mean() for s in symbols]\n",
    "    ss = [da_df[(da_df.symbol==s)&(da_df.feature_set==fset)]['da'].std()  for s in symbols]\n",
    "    ax.bar(x + j*w, ms, w, label=fset, yerr=ss, capsize=4, alpha=0.8)\n",
    "ax.axhline(0.5, color='red', ls='--', lw=1.2, label='DA=0.50 (random)')\n",
    "ax.set_xticks(x + w/2); ax.set_xticklabels(symbols)\n",
    "ax.set_ylabel('Directional Accuracy'); ax.set_ylim(0.4, 0.7)\n",
    "ax.set_title('DA by index and feature set')\n",
    "ax.legend(); plt.tight_layout()\n",
    "fig.savefig(paths.figures / 'da_comparison.png', dpi=150); plt.show()",
]))


# Latency + export + footer
cells.append(md("md-s5d", ["### 5d. Inference latency"]))
cells.append(code("c-latency", [
    "pp_lat = paths.preds_path('SPX', 'multivariate', SEEDS[0])\n",
    "if pp_lat.exists():\n",
    "    lat = pd.read_parquet(pp_lat)['infer_ms']\n",
    "    print('Inference latency (ms):')\n",
    "    print(lat.describe().to_string())\n",
    "    print(f'\\nPaper ~2000 ms  |  Our mean: {lat.mean():.1f} ms')\n",
    "    fig, ax = plt.subplots(figsize=(8, 3))\n",
    "    lat.clip(upper=lat.quantile(0.99)).plot.hist(ax=ax, bins=60, density=True)\n",
    "    ax.axvline(lat.mean(), color='red',    ls='--', lw=1.2, label=f'Mean={lat.mean():.1f}ms')\n",
    "    ax.axvline(2000,       color='orange', ls=':',  lw=1.2, label='Paper: 2000ms')\n",
    "    ax.set_xlabel('Inference latency (ms)')\n",
    "    ax.legend(); plt.tight_layout()\n",
    "    fig.savefig(paths.figures / 'SPX_latency.png', dpi=150); plt.show()",
]))
cells.append(md("md-s5e", ["### 5e. Export thesis-ready tables"]))
cells.append(code("c-export", [
    "cmp_df.to_csv(paths.results / 'table_reproduction.csv')\n",
    "print(f\"Saved -> {paths.results / 'table_reproduction.csv'}\")\n",
    "\n",
    "rq1_df = mase_summary.join(da_summary, rsuffix='_da').rename(columns={\n",
    "    'mean': 'mase_mean', 'std': 'mase_std',\n",
    "    'mean_da': 'da_mean', 'std_da': 'da_std',\n",
    "})\n",
    "rq1_df.to_csv(paths.results / 'table_rq1_mase_da.csv')\n",
    "print(f\"Saved -> {paths.results / 'table_rq1_mase_da.csv'}\")\n",
    "\n",
    "summary = summarise_runs(all_results)\n",
    "summary.to_csv(paths.results / 'full_metrics_summary.csv', index=False)\n",
    "print(f\"Saved -> {paths.results / 'full_metrics_summary.csv'}\")\n",
    "\n",
    "print('\\nAll done! Output files:')\n",
    "for f in sorted(paths.results.rglob('*.csv')):\n",
    "    print(f'  {f}')",
]))
cells.append(md("md-s5f", [
    "### 5f. Interpretation checklist\n",
    "\n",
    "| Metric | Threshold | Interpretation |\n",
    "|--------|-----------|----------------|\n",
    "| MAPE   | ~0.001    | Paper level; gap -> discrepancy |\n",
    "| MASE   | >= 0.90   | H1 supported (no skill over naive) |\n",
    "| MASE   | < 0.85    | H1 falsified (genuine skill) |\n",
    "| DA     | ~= 0.50   | Coin-flip, no directional skill |\n",
    "| DA     | > 0.55    | Some directional signal |\n",
]))

# Write notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print(f"Written : {OUT}")
print(f"Cells   : {len(cells)}")
print(f"Size    : {OUT.stat().st_size / 1024:.1f} KB")


