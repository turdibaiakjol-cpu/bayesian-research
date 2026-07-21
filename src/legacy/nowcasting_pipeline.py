# %% [markdown]
# # Nowcasting GDP in a Data-Constrained Commodity Economy:
# # A Bayesian vs Classical Comparative Framework for Kazakhstan
#
# **Author:** [Author Name]
#
# **Abstract:** [Placeholder]
#
# ---

# %% [markdown]
# ## Cell 1: Dependencies & Version Logging

# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats, optimize, linalg
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
from io import BytesIO
import os, sys, traceback, json
from datetime import datetime

# Set paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
FIG_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.figsize': (12, 6), 'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': 0.3, 'lines.linewidth': 1.5
})

print("=" * 60)
print("PACKAGE VERSIONS")
print("=" * 60)
for pkg_name, pkg in [("pandas", pd), ("numpy", np), ("scipy", stats),
                       ("statsmodels", sm), ("matplotlib", matplotlib)]:
    print(f"  {pkg_name}: {pkg.__version__ if hasattr(pkg, '__version__') else 'N/A'}")
print(f"  Python: {sys.version.split()[0]}")
print("=" * 60)

# Logging helper
def log_step(step_name, status, detail=""):
    ts = datetime.now().strftime("%H:%M:%S")
    symbol = "✓" if status == "SUCCESS" else "✗"
    print(f"[{ts}] [{symbol} {status}] {step_name}")
    if detail:
        print(f"         {detail}")

