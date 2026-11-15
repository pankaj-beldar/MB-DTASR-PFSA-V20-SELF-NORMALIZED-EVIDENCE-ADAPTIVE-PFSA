# OUTPUT POLICY (memory-safe best-seed reporting)
# All seeds: best x/f, error, FE count, runtime, nonfinite count, SHA256,
# per-TRUE-FE convergence CSV + PNG/PDF; aggregate statistics and convergence.
# After all seeds of EACH function: replay lowest-f seed (earliest run on ties)
# into BEST_RUN_DETAILED. Verify exact X/y trajectory digest against original.
# Replay is NOT an independent trial; its FE/runtime are excluded from statistics.
# Full candidate/selection CSVs stream in 512-row buffers. Scatter plots use
# a bounded sample; candidate-count summaries use ALL records in CSV chunks.
# Detailed replay preserves the original script's diagnostics, not new internal
# solver instrumentation. Existing experiment folders are preserved on rerun.
# RUN_MODE='SUITE': every function; RUN_MODE='SINGLE': FUNCTION_ID only.
# Both modes use this output policy. Full CEC-suite validation is not bundled.

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:53:33 2026

@author: mech
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:25:37 2026

@author: cadcam
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 21:09:01 2026

@author: Win 11
"""

"""
MB-DTASR-PFSA V20 FOR CEC-2017: SELF-NORMALIZED EVIDENCE-ADAPTIVE PFSA
=====================================================================
Multi-Basin Trust-Region Parameter-Free Surrogate-Assisted Optimization

V20 adaptive-acquisition addition
-------------------------------
The fixed 0.40/0.25/0.20/0.15 acquisition mixture is replaced by online,
self-normalized weights computed from verified productivity, surrogate reliability,
global basin confidence, and meaningful stagnation. The weights are logged per cycle.

V19 additions
-------------
1. TRUE evaluation slots are evidence-adaptive instead of fixed state recipes.
2. One global and one local slot are preserved; remaining slots compete by verified productivity.
3. Global basin confidence is separated from local surrogate reliability.
4. Repeated failed recovery episodes trigger a generic hard basin-distribution reset.
5. No function ID, optimum value/vector, shift, or rotation is used by search decisions.

V10 features
--------------
1. BMR trust regions no longer shrink after every failed TRUE evaluation.
   They shrink only after FAILURE_PATIENCE consecutive local failures.

2. BWR and BMR TRUE-evaluation slots are structurally guaranteed.
   If a source is absent from the Pareto-admissible set, selection falls
   back to that source's full candidate pool.

3. Stale-basin replacement:
   a basin that remains at minimum radius without TRUE improvement for
   STALE_BASIN_TRUE_FES is reinitialized around a promising, spatially
   separated BWR candidate.

4. Recovery remains bounded and has a TRUE-FE cooldown.

5. Basin success is judged against the BASIN incumbent, not only against
   the global incumbent.

6. Known CEC optimum is NEVER used by the optimizer.

Dependencies
------------

"""
# Install the GitHub CEC-2017 backend in the SAME Python environment used by Spyder:
# python -m pip install "git+https://github.com/tilleyd/cec2017-py.git"
# Other dependencies:
# python -m pip install numpy scipy scikit-learn matplotlib pandas numba
import os
import csv
import hashlib
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr, shapiro, wilcoxon, mannwhitneyu, friedmanchisquare, rankdata
from sklearn.cluster import KMeans
from numba import njit
warnings.filterwarnings('ignore')
from cycler import cycler

# Publication-ready vibrant graph theme (reporting only; optimizer logic unchanged)
VIBRANT_PUBLICATION_COLORS = [
    '#FF006E', '#00B4D8', '#FF9F1C', '#2EC4B6', '#8338EC', '#FFBE0B',
    '#3A86FF', '#FB5607', '#06D6A0', '#D90429', '#9B5DE5', '#00F5D4'
]
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 15,
    'figure.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.8,
    'lines.markersize': 8,
    'grid.alpha': 0.20,
    'grid.linewidth': 1.0,
    'figure.dpi': 130,
    'savefig.dpi': 450,
    'savefig.bbox': 'tight',
    'axes.prop_cycle': cycler(color=VIBRANT_PUBLICATION_COLORS),
})


RUN_MODE = 'SUITE'
FUNCTION_ID = 9
DIM = 10
SUITE_FUNCTION_IDS = [1] + list(range(3, 31))  # corrected CEC-2017: F2 excluded
RUNS_PER_FUNCTION = 30
DISPLAY_EACH_RUN_LIVE = True  # Print each parallel run's cycle-by-cycle execution log to console
PARALLEL_WORKERS = 3
RUN_SEED_START = 42
MAX_TRUE_FES = 10000
USE_CEC2017_OFFICIAL_BUDGET = False
SKIP_UNAVAILABLE_FUNCTIONS = True
INITIAL_DOE = max(80, 8 * DIM)
GLOBAL_POP_SIZE = max(50, 5 * DIM)
LOCAL_POP_SIZE = max(30, 3 * DIM)
MAX_BASINS = 4
ELITE_FRACTION = 0.2
BATCH_SIZE = 5
GLOBAL_BWR_OFFSPRING = max(250, 25 * DIM)
LOCAL_BMR_OFFSPRING = max(120, 12 * DIM)
COVERAGE_CANDIDATES = max(100, 10 * DIM)
GLOBAL_RBF_MAX_POINTS = 600
LOCAL_RBF_MAX_POINTS = max(80, 8 * DIM)
RBF_RIDGE = 1e-08
NC_NEIGHBORS = max(4, min(10, DIM))
TR_RADIUS_INITIAL = 0.2
TR_RADIUS_MIN = 0.04
TR_RADIUS_MAX = 0.35
TR_EXPAND = 1.2
TR_SHRINK = 0.8
FAILURE_PATIENCE = 5
STALL_TRUE_FES = max(120, 12 * DIM)
RECOVERY_CYCLES = 4
RECOVERY_COOLDOWN_TRUE_FES = max(150, 15 * DIM)
STALE_BASIN_TRUE_FES = max(150, 15 * DIM)
BASIN_RESTART_MIN_DISTANCE = 0.15
RESTART_DISTANCE_REWARD = 0.35
RELIABILITY_WINDOW = 40
TRUE_ANCHOR_FRACTION = 0.5
TRUE_ANCHOR_ELITE_FRACTION = 0.7
RECENT_IMPROVEMENT_WINDOW_TRUE_FES = max(50, 5 * DIM)
BASIN_SUCCESS_WINDOW = 20
LOCAL_SUCCESS_RATE_MIN = 0.1
INCUMBENT_REFINE_CANDIDATES = max(150, 15 * DIM)
INCUMBENT_REFINE_RADIUS = 0.06
INCUMBENT_REFINE_MIN_RADIUS = 0.01
MAX_BMR_TRUE_SLOTS = 2
BASIN_CREDIT_WINDOW = 40
LOCAL_BASIN_CREDIT_MIN = 0.05
INTENSIFY_RELIABILITY_MIN = 0.75
INTENSIFY_STALL_WINDOW_TRUE_FES = max(30, 3 * DIM)
INTENSIFY_RADII = (0.04, 0.015, 0.005)
INTENSIFY_CANDIDATES_PER_RADIUS = max(80, 8 * DIM)
RECOVERY_BWR_TRUE_SLOTS = 3
RESTART_ELITE_FRACTION = 0.2
RESTART_MIN_ELITE_DISTANCE = 0.12
INTENSIFY_START_STALL_FES = max(20, 2 * DIM)
INTENSIFY_MAX_STALL_FES = max(100, 10 * DIM)
LOCAL_RELIABILITY_MIN = 0.65
POLL_RADII = (0.05, 0.02, 0.008, 0.003)
SURR_LOCAL_RESTARTS = max(6, DIM // 2)
SURR_LOCAL_MAXITER = 80
SURR_LOCAL_RADIUS = 0.08
NOVELTY_POOL_MULTIPLIER = 20
NOVELTY_KEEP = max(100, 10 * DIM)
DEEP_RECOVERY_CYCLES = 6
DEEP_RECOVERY_COOLDOWN_TRUE_FES = max(250, 25 * DIM)
DEEP_RECOVERY_BWR_OFFSPRING_MULTIPLIER = 2.5
DEEP_RECOVERY_RESEED_FRACTION = 0.5
DEEP_NOVELTY_KEEP = max(200, 20 * DIM)
DEEP_NOVELTY_POOL_MULTIPLIER = 40
ADAPTIVE_POLL_MIN = 0.0005
ADAPTIVE_POLL_MAX = 0.05
PERSISTENT_SURR_LOCAL_RESTARTS = max(10, DIM)
LOCAL_SUCCESS_MEMORY_FES = max(50, 5 * DIM)
QUAD_LOCAL_MAX_POINTS = max(120, 12 * DIM)
QUAD_LOCAL_MIN_POINTS = max(2 * DIM + 5, 25)
QUAD_RIDGE = 1e-06
QUAD_TR_RADIUS_INITIAL = 0.08
QUAD_TR_RADIUS_MIN = 0.002
QUAD_TR_RADIUS_MAX = 0.15
SOURCE_CREDIT_WINDOW = 60
PRODUCTIVITY_WINDOW = 80
PRODUCTIVITY_PRIOR_SUCCESS = 1.0
PRODUCTIVITY_PRIOR_TOTAL = 4.0
PRODUCTIVITY_GAIN_WEIGHT = 2.0
PRODUCTIVITY_MIN_OBS = 5
FORCE_NOVELTY_AFTER_FES = max(120, 12 * DIM)
ANISO_MIN_POINTS = max(3 * DIM + 8, 40)
ANISO_MAX_POINTS = max(180, 18 * DIM)
ANISO_RADIUS_INITIAL = 0.05
ANISO_RADIUS_MIN = 0.0002
ANISO_RADIUS_MAX = 0.12
ANISO_AXIS_RATIO_MAX = 50.0
ANISO_CURVATURE_WEIGHT = 0.7
ANISO_SPREAD_WEIGHT = 0.3
ANISO_RANDOM_CANDIDATES = max(120, 12 * DIM)
ANISO_LBFGSB_RESTARTS = max(8, DIM)
ANISO_LBFGSB_MAXITER = 100
ANISO_TRIGGER_STALL_FES = max(25, 3 * DIM)
ANISO_DEEP_TRIGGER_FES = max(80, 8 * DIM)
ANISO_SUCCESS_MEMORY_FES = max(60, 6 * DIM)
PRECISION_RADIUS_INITIAL = 0.005
PRECISION_RADIUS_MIN = 1e-07
PRECISION_RADIUS_MAX = 0.02
PRECISION_TRIGGER_STALL_FES = max(20, 2 * DIM)
PRECISION_SUCCESS_MEMORY_FES = max(40, 4 * DIM)
PRECISION_AXIS_LEVELS = (1.0, 0.25, 0.0625, 0.015625)
PRECISION_RANDOM_CANDIDATES = max(60, 6 * DIM)
PRECISION_RELIABILITY_MIN = 0.6
ANISO_FAILURE_CYCLE_LIMIT = 10
PRECISION_RETRY_FAILURE_LIMIT = 4
LOCAL_MODE_SUCCESS_WINDOW = 24
ANISO_MIN_SUCCESS_RATIO = 0.05
PRECISION_MIN_SUCCESS_RATIO = 0.15
PRECISION_MEMORY_INITIAL_RADIUS = 0.005
PRECISION_SUCCESS_SHRINK = 0.8
PRECISION_FAILURE_EXPAND = 1.5
PRECISION_MEMORY_MIN_RADIUS = 1e-06
PRECISION_MEMORY_MAX_RADIUS = 0.02
PRECISION_RETRY_SCALE = 0.2
MAG_SUCCESS_REL_GAIN = 0.0001
MAG_SUCCESS_ABS_GAIN = 1e-08
SLOW_PROGRESS_WINDOW_FES = max(100, 10 * DIM)
SLOW_PROGRESS_REL_TOL = 0.0001
MEANINGFUL_STALL_RECOVERY_FES = max(60, 6 * DIM)
MEANINGFUL_STALL_DEEP_FES = max(180, 18 * DIM)
MEANINGFUL_RECOVERY_COOLDOWN_FES = max(60, 6 * DIM)
SLOW_PROGRESS_RECOVERY_CYCLES = 8
ROT_QUAD_MIN_POINTS = max(3 * DIM + 10, 40)
ROT_QUAD_MAX_POINTS = max(250, 25 * DIM)
ROT_QUAD_RIDGE = 0.0001
ROT_TR_RADIUS_INITIAL = 0.06
ROT_TR_RADIUS_MIN = 1e-05
ROT_TR_RADIUS_MAX = 0.15
ROT_TR_AXIS_RATIO_MAX = 100.0
ROT_TR_RANDOM_CANDIDATES = max(160, 16 * DIM)
ROT_TR_RESTARTS = max(10, DIM)
ROT_TR_MAXITER = 120
ROT_TR_ETA_BAD = 0.1
ROT_TR_ETA_GOOD = 0.5
ROT_TR_SHRINK = 0.6
ROT_TR_EXPAND = 1.25
NEWTON_TR_TRIGGER_FES = max(60, 6 * DIM)
NEWTON_TR_COOLDOWN_FES = max(80, 8 * DIM)
NEWTON_TR_MAX_ITERS = 4
NEWTON_TR_RADIUS_INITIAL = 0.02
NEWTON_TR_RADIUS_MIN = 2e-06
NEWTON_TR_RADIUS_MAX = 0.08
NEWTON_TR_SHRINK = 0.5
NEWTON_TR_EXPAND = 1.6
NEWTON_TR_ETA_ACCEPT = 0.05
NEWTON_TR_ETA_EXPAND = 0.75
NEWTON_TR_CURV_FLOOR = 1e-08
NEWTON_TR_MIN_PRED_REDUCTION = 1e-12
NEWTON_TR_MAX_STEP_NORM_FACTOR = 1.0
V13_RECOVERY_CYCLES = 6
STRICT_BACKEND_REFERENCE = True
LOCAL_SOURCES = ('BMR', 'INCUMBENT', 'POLL', 'SURR_LOCAL', 'QUAD_TR', 'ANISO', 'PRECISION', 'NEWTON_TR')
GLOBAL_SOURCES = ('BWR', 'NOVELTY', 'COVERAGE')
MICRO_LOCAL_TRIGGER_FES = max(40, 4 * DIM)
MICRO_LOCAL_MAX_FES = max(500, 50 * DIM)
DEEP_RECOVERY_TRIGGER_FES = max(600, 60 * DIM)
DUP_TOL = 1e-10
MASTER_SEED = 42
COMMON_SEED_ACROSS_FUNCTIONS = True  # Same seed schedule for every CEC function
if ('COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ):
    RESULT_DIR = '/content/MB_DTASR_PFSA_CEC2017_RESULTS'
else:
    try:
        _SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        _SCRIPT_DIR = Path.cwd().resolve()
    RESULT_DIR = str((_SCRIPT_DIR / 'MB_DTASR_PFSA_CEC2017_RESULTS').resolve())
os.makedirs(RESULT_DIR, exist_ok=True)
try:
    with open(os.path.join(RESULT_DIR, 'OPEN_THIS_FILE_TO_FIND_RESULTS.txt'), 'w', encoding='utf-8') as _fh:
        _fh.write('CEC-2017 V20 results root\n' + os.path.abspath(RESULT_DIR) + '\n')
except Exception:
    pass
print(f'[RESULTS ROOT] {os.path.abspath(RESULT_DIR)}', flush=True)
PUBLICATION_RESULTS_ROOT = '/content/MB_DTASR_PFSA_CEC2017_PUBLICATION_RESULTS' if ('COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ) else 'MB_DTASR_PFSA_CEC2017_PUBLICATION_RESULTS'
PUBLICATION_SUCCESS_TOL = 1e-08
PUBLICATION_BOOTSTRAP_SAMPLES = 5000
PUBLICATION_CONFIDENCE = 0.95
PUBLICATION_SHOW_PLOTS = True
PUBLICATION_COMPARATOR_CSV = None
PUBLICATION_ABLATION_CSV = None
# V17 generic robustness additions (NO function-ID / optimum information)
V17_ESCAPE_TRIGGER_FES = max(100, 10 * DIM)
V17_ESCAPE_RELEASE_FES = max(30, 3 * DIM)
V17_ESCAPE_COOLDOWN_FES = max(120, 12 * DIM)
V17_ESCAPE_CYCLES = 6
V17_ESCAPE_BWR_MULTIPLIER = 2.0
V17_ESCAPE_NOVELTY_KEEP = max(200, 20 * DIM)
V17_BASIN_SHOCK_TRIGGER_FES = max(160, 16 * DIM)
V17_BASIN_SHOCK_COOLDOWN_FES = max(250, 25 * DIM)
V17_BASIN_SHOCK_FRACTION = 0.75

# ---------------------------------------------------------------------
# V18: saturated-basin quarantine + maximin basin regeneration
# ---------------------------------------------------------------------
# A basin is locally saturated when it has accumulated enough verified
# local evaluations but has not produced meaningful progress for a long
# TRUE-FE interval.  No function ID, f_global, shift, or rotation is used.
V18_SATURATION_TRIGGER_FES = max(120, 12 * DIM)
V18_SATURATION_MIN_BMR_EVALS = max(8, DIM)
V18_SATURATION_MAX_SUCCESS_RATE = 0.05
V18_QUARANTINE_FES = max(300, 30 * DIM)
V18_QUARANTINE_DISTANCE = 0.12
V18_REGENERATE_FRACTION = 0.75
V18_MAXIMIN_POOL = max(2000, 200 * DIM)
V18_MAXIMIN_MIN_ARCHIVE_DISTANCE = 0.10
V18_SATURATION_ESCAPE_CYCLES = 8
V18_SATURATION_COOLDOWN_FES = max(180, 18 * DIM)

# ---------------------------------------------------------------------
# V19: evidence-adaptive TRUE-FE allocation + fitness-aware recovery
# ---------------------------------------------------------------------
V19_MIN_GLOBAL_SLOTS = 1
V19_MIN_LOCAL_SLOTS = 1
V19_RECOVERY_FAIL_LIMIT = 3
V19_RECOVERY_SUCCESS_REL = 1e-4
# Legacy V19 acquisition coefficients retained for reproducibility metadata only.
# V20 candidate ranking DOES NOT use these fixed coefficients.
V19_SOURCE_PRODUCTIVITY_WEIGHT = 0.40
V19_PREDICTION_WEIGHT = 0.25
V19_NOVELTY_WEIGHT = 0.20
V19_DIVERSITY_WEIGHT = 0.15
V19_LOW_RELIABILITY_THRESHOLD = 0.50
V19_LOW_RELIABILITY_EXPLORATION_BONUS = 0.20
V19_LONG_STALL_FES = max(400, 40 * DIM)
V19_GLOBAL_RESET_FRACTION = 0.75
V19_FITNESS_REGEN_POOL = max(4096, 400 * DIM)
V19_REGEN_KEEP = max(128, 12 * DIM)
V19_REGEN_PRED_WEIGHT = 0.40
V19_REGEN_ARCHIVE_DISTANCE_WEIGHT = 0.30
V19_REGEN_BASIN_DISTANCE_WEIGHT = 0.20
V19_REGEN_UNCERTAINTY_WEIGHT = 0.10
V19_GLOBAL_CONFIDENCE_DIVERSITY_RADIUS = 0.12
V17_NEWTON_PCA_POINTS = max(40, 4 * DIM)
V17_NEWTON_MIN_VALID_DIRS = max(2, DIM // 3)
V17_NEWTON_FD_RADIUS_MIN = 5.0e-5
V17_NONFINITE_PENALTY = 1.0e300


# =====================================================================
# PURE-PYTHON CEC-2017 BACKEND
# =====================================================================
"""
CEC-2017 Bound-Constrained Test Suite -- pure Python/NumPy port.

Ported from the user-supplied official-style `cec17_func.cpp` (Noor Awad,
last update 26-05-2017). It preserves the C formulas, shift/rotation,
hybrid grouping/shuffling, composition weights, and CEC biases.

IMPORTANT
---------
This module still requires the ORIGINAL CEC-2017 `input_data` directory:
  M_<func>_D<dim>.txt
  shift_data_<func>.txt
  shuffle_data_<func>_D<dim>.txt where required

No MATLAB, MEX, OPFUNU, or MATLAB Engine is required.

Corrected suite: F1 and F3-F30. F2 is deleted in the supplied source.
"""

import math
from pathlib import Path
import numpy as np

PI = math.pi
E = math.e
INF = 1.0e99
VALID_FUNCTIONS = (1,) + tuple(range(3, 31))
VALID_DIMS = (2, 10, 20, 30, 50, 100)


class PurePythonCEC2017:
    def __init__(self, func_num: int, dim: int, data_dir="input_data"):
        self.func_num = int(func_num)
        self.dim = int(dim)
        self.data_dir = Path(data_dir)

        if self.func_num == 2:
            raise ValueError("CEC-2017 F2 was deleted in the supplied official source.")
        if self.func_num not in VALID_FUNCTIONS:
            raise ValueError("func_num must be F1 or F3-F30.")
        if self.dim not in VALID_DIMS:
            raise ValueError(f"Unsupported dimension D={dim}; valid: {VALID_DIMS}.")
        if self.dim == 2 and (17 <= self.func_num <= 22 or self.func_num in (29, 30)):
            raise ValueError(f"F{self.func_num} is not defined for D=2.")

        self.M = self._load_matrix()
        self.OShift = self._load_shift()
        self.SS = self._load_shuffle()

    @property
    def optimum(self):
        return 100.0 * self.func_num

    def _load_matrix(self):
        p = self.data_dir / f"M_{self.func_num}_D{self.dim}.txt"
        a = np.loadtxt(p, dtype=float).reshape(-1)

        # CEC-2017 data layout:
        #   F1-F20  : one D x D rotation matrix
        #   F21-F30 : ten D x D matrices (composition-function components)
        #
        # F20 is a HYBRID function, not a composition function.  Therefore
        # M_20_D10.txt correctly contains 100 values, not 1000.
        nmat = 1 if self.func_num <= 20 else 10
        need = nmat * self.dim * self.dim

        if a.size != need:
            raise ValueError(
                f"{p}: CEC-2017 matrix-layout mismatch. "
                f"F{self.func_num}, D={self.dim} requires exactly {need} "
                f"values ({nmat} matrix/matrices), but file contains {a.size}."
            )
        return a.reshape(nmat, self.dim, self.dim)

    def _load_shift(self):
        p = self.data_dir / f"shift_data_{self.func_num}.txt"
        # Composition files may have long lines; load all numeric values.
        rows = []
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                vals = np.fromstring(line, sep=" ")
                if vals.size:
                    rows.append(vals)
        # F1-F20 use one shift vector. F21-F30 are composition functions
        # and use ten component shift vectors.
        if self.func_num <= 20:
            flat = np.concatenate(rows)
            if flat.size < self.dim:
                raise ValueError(f"{p}: insufficient shift values")
            return flat[:self.dim].reshape(1, self.dim)

        # Official composition data are stored as ten rows; the original C
        # loader consumes the first D values of each row.
        if len(rows) >= 10 and all(r.size >= self.dim for r in rows[:10]):
            return np.vstack([r[:self.dim] for r in rows[:10]])

        # Fallback for whitespace-flattened copies of the official files.
        flat = np.concatenate(rows)
        if flat.size < 10*self.dim:
            raise ValueError(
                f"{p}: expected at least {10*self.dim} composition shift "
                f"values for F{self.func_num}, got {flat.size}"
            )
        return flat[:10*self.dim].reshape(10, self.dim)

    def _load_shuffle(self):
        f = self.func_num
        if not (11 <= f <= 20 or f in (29, 30)):
            return None
        p = self.data_dir / f"shuffle_data_{f}_D{self.dim}.txt"
        s = np.loadtxt(p, dtype=int).reshape(-1)
        need = self.dim * (10 if f in (29, 30) else 1)
        if s.size < need:
            raise ValueError(f"{p}: expected {need} shuffle indices, got {s.size}")
        # Convert CEC's 1-based permutation to Python 0-based.
        return s[:need] - 1

    @staticmethod
    def _rot(x, M):
        return M @ x

    def _sr(self, x, Os, M, rate=1.0, shift=True, rotate=True):
        y = np.asarray(x, float).copy()
        if shift:
            y = y - Os
        y *= rate
        return M @ y if rotate else y

    # -------------------------- base functions --------------------------
    def sphere(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r); return float(z@z)

    def ellips(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r); n=len(z)
        return float(np.sum((10.0**(6*np.arange(n)/(n-1))) * z*z))

    def bent_cigar(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r)
        return float(z[0]**2 + 1e6*np.sum(z[1:]**2))

    def discus(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r)
        return float(1e6*z[0]**2 + np.sum(z[1:]**2))

    def rosenbrock(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,2.048/100.0,s,r)+1.0
        return float(np.sum(100*(z[:-1]**2-z[1:])**2+(z[:-1]-1)**2))

    def schaffer_f7(self,x,Os,M,s=1,r=1):
        # C source uses y after sr_func; y is shifted/scaled pre-rotation.
        y=np.asarray(x,float).copy()
        if s: y=y-Os
        y*=1.0
        if r: z=M@y
        else: z=y.copy()
        q=np.sqrt(y[:-1]**2+y[1:]**2)
        v=np.sqrt(q)*(1+np.sin(50*q**0.2)**2)
        return float((np.sum(v)/(len(x)-1))**2)

    def ackley(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r); n=len(z)
        return float(E-20*np.exp(-.2*np.sqrt(np.sum(z*z)/n))
                     -np.exp(np.sum(np.cos(2*PI*z))/n)+20)

    def weierstrass(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,.5/100.0,s,r)
        k=np.arange(21); a=.5**k; b=3.0**k
        c=np.sum(a*np.cos(2*PI*b*.5))
        return float(sum(np.sum(a*np.cos(2*PI*b*(zi+.5))) for zi in z)-len(z)*c)

    def griewank(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,600/100.0,s,r)
        return float(1+np.sum(z*z)/4000-np.prod(np.cos(z/np.sqrt(np.arange(1,len(z)+1)))))

    def rastrigin(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,5.12/100.0,s,r)
        return float(np.sum(z*z-10*np.cos(2*PI*z)+10))

    def step_rastrigin(self,x,Os,M,s=1,r=1):
        # Preserve source behavior: round shifted values when |x-Os| > .5.
        xx=np.asarray(x,float).copy()
        mask=np.abs(xx-Os)>.5
        xx[mask]=Os[mask]+np.floor(2*(xx[mask]-Os[mask])+.5)/2
        z=self._sr(xx,Os,M,5.12/100.0,s,r)
        return float(np.sum(z*z-10*np.cos(2*PI*z)+10))

    def schwefel(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1000/100.0,s,r)+420.9687462275036
        n=len(z); f=0.0
        for v in z:
            if v>500:
                q=500-np.fmod(v,500)
                f-=q*np.sin(np.sqrt(q)); f+=((v-500)/100)**2/n
            elif v<-500:
                q=-500+np.fmod(abs(v),500)
                f-=q*np.sin(np.sqrt(500-np.fmod(abs(v),500)))
                f+=((v+500)/100)**2/n
            else:
                f-=v*np.sin(np.sqrt(abs(v)))
        return float(f+418.9828872724338*n)

    def katsuura(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,5/100.0,s,r); n=len(z)
        prod=1.0; tmp3=n**1.2
        for i,v in enumerate(z):
            temp=0.0
            for j in range(1,33):
                p=2.0**j
                temp+=abs(p*v-np.floor(p*v+.5))/p
            prod*= (1+(i+1)*temp)**(10/tmp3)
        q=10/n/n
        return float(prod*q-q)

    def bi_rastrigin(self,x,Os,M,s=1,r=1):
        n=len(x); mu0=2.5; d=1.0
        ss=1-1/(2*np.sqrt(n+20)-8.2); mu1=-np.sqrt((mu0*mu0-d)/ss)
        y=np.asarray(x,float).copy()
        if s: y-=Os
        y*=.1
        tmpx=2*y
        tmpx=np.where(Os<0,-tmpx,tmpx)
        z=tmpx.copy(); tmpx=tmpx+mu0
        t1=np.sum((tmpx-mu0)**2)
        t2=ss*np.sum((tmpx-mu1)**2)+d*n
        rz=M@z if r else z
        return float(min(t1,t2)+10*(n-np.sum(np.cos(2*PI*rz))))

    def grie_rosen(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,5/100.0,s,r)+1
        a=z*z-np.roll(z,-1); b=z-1
        t=100*a*a+b*b
        return float(np.sum(t*t/4000-np.cos(t)+1))

    def escaffer6(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r); zz=np.roll(z,-1)
        q=z*z+zz*zz
        return float(np.sum(.5+(np.sin(np.sqrt(q))**2-.5)/(1+.001*q)**2))

    def happycat(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,5/100.0,s,r)-1
        r2=np.sum(z*z); sm=np.sum(z); n=len(z)
        return float(abs(r2-n)**.25 + (.5*r2+sm)/n + .5)

    def hgbat(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,5/100.0,s,r)-1
        r2=np.sum(z*z); sm=np.sum(z); n=len(z)
        return float(abs(r2*r2-sm*sm)**.5 + (.5*r2+sm)/n + .5)

    def zakharov(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r)
        q=np.sum(.5*np.arange(1,len(z)+1)*z)
        return float(np.sum(z*z)+q*q+q**4)

    def levy(self,x,Os,M,s=1,r=1):
        z=self._sr(x,Os,M,1.0,s,r); w=1+(z-1)/4
        t1=np.sin(PI*w[0])**2
        t3=(w[-1]-1)**2*(1+np.sin(2*PI*w[-1])**2)
        # Match uploaded C exactly: sin(PI*wi + 1), not sin(PI*(wi+1)).
        sm=np.sum((w[:-1]-1)**2*(1+10*np.sin(PI*w[:-1]+1)**2))
        return float(t1+sm+t3)

    # ---------------------------- hybrids -------------------------------
    def _hybrid(self,x,Os,M,S,props,funcs):
        z=self._sr(x,Os,M,1.0,1,1)
        y=z[np.asarray(S,dtype=int)]
        sizes=[]; used=0
        for p in props[:-1]:
            q=int(math.ceil(p*len(x))); sizes.append(q); used+=q
        sizes.append(len(x)-used)
        ans=0.0; st=0
        dummyO=np.zeros(len(x)); dummyM=np.eye(len(x))
        for size,fn in zip(sizes,funcs):
            seg=y[st:st+size]
            ans+=fn(seg,dummyO[:size],dummyM[:size,:size],0,0)
            st+=size
        return float(ans)

    def hf01(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.2,.4,.4],[self.zakharov,self.rosenbrock,self.rastrigin])
    def hf02(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.3,.3,.4],[self.ellips,self.schwefel,self.bent_cigar])
    def hf03(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.3,.3,.4],[self.bent_cigar,self.rosenbrock,self.bi_rastrigin])
    def hf04(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.2,.2,.2,.4],[self.ellips,self.ackley,self.schaffer_f7,self.rastrigin])
    def hf05(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.2,.2,.3,.3],[self.bent_cigar,self.hgbat,self.rastrigin,self.rosenbrock])
    def hf06(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.2,.2,.3,.3],[self.escaffer6,self.hgbat,self.rosenbrock,self.schwefel])
    def hf07(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.1,.2,.2,.2,.3],[self.katsuura,self.ackley,self.grie_rosen,self.schwefel,self.rastrigin])
    def hf08(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.2,.2,.2,.2,.2],[self.ellips,self.ackley,self.rastrigin,self.hgbat,self.discus])
    def hf09(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.2,.2,.2,.2,.2],[self.bent_cigar,self.rastrigin,self.grie_rosen,self.weierstrass,self.escaffer6])
    def hf10(self,x,Os,M,S): return self._hybrid(x,Os,M,S,[.1,.1,.2,.2,.2,.2],[self.hgbat,self.katsuura,self.ackley,self.rastrigin,self.schwefel,self.schaffer_f7])

    # -------------------------- compositions ----------------------------
    @staticmethod
    def _cf_cal(x, Os, delta, bias, fit):
        x=np.asarray(x,float); Os=np.asarray(Os,float)
        delta=np.asarray(delta,float); bias=np.asarray(bias,float); fit=np.asarray(fit,float)+bias
        d2=np.sum((Os-x)**2,axis=1)
        w=np.empty(len(fit))
        nz=d2!=0
        w[nz]=np.sqrt(1/d2[nz])*np.exp(-d2[nz]/(2*len(x)*delta[nz]**2))
        w[~nz]=INF
        if np.max(w)==0:
            w[:]=1
        return float(np.sum(w/np.sum(w)*fit))

    def _comp(self,x,defs,delta,bias):
        fit=[]
        for i,(fn,scale) in enumerate(defs):
            fit.append(scale*fn(x,self.OShift[i],self.M[i],1,1))
        return self._cf_cal(x,self.OShift[:len(defs)],delta,bias,fit)

    def cf01(self,x): return self._comp(x,[(self.rosenbrock,1),(self.ellips,1e4/1e10),(self.rastrigin,1)],[10,20,30],[0,100,200])
    def cf02(self,x): return self._comp(x,[(self.rastrigin,1),(self.griewank,1000/100),(self.schwefel,1)],[10,20,30],[0,100,200])
    def cf03(self,x): return self._comp(x,[(self.rosenbrock,1),(self.ackley,1000/100),(self.schwefel,1),(self.rastrigin,1)],[10,20,30,40],[0,100,200,300])
    def cf04(self,x): return self._comp(x,[(self.ackley,1000/100),(self.ellips,1e4/1e10),(self.griewank,1000/100),(self.rastrigin,1)],[10,20,30,40],[0,100,200,300])
    def cf05(self,x): return self._comp(x,[(self.rastrigin,1e4/1e3),(self.happycat,1000/1e3),(self.ackley,1000/100),(self.discus,1e4/1e10),(self.rosenbrock,1)],[10,20,30,40,50],[0,100,200,300,400])
    def cf06(self,x): return self._comp(x,[(self.escaffer6,1e4/2e7),(self.schwefel,1),(self.griewank,1000/100),(self.rosenbrock,1),(self.rastrigin,1e4/1e3)],[10,20,20,30,40],[0,100,200,300,400])
    def cf07(self,x): return self._comp(x,[(self.hgbat,1e4/1000),(self.rastrigin,1e4/1e3),(self.schwefel,1e4/4e3),(self.bent_cigar,1e4/1e30),(self.ellips,1e4/1e10),(self.escaffer6,1e4/2e7)],[10,20,30,40,50,60],[0,100,200,300,400,500])
    def cf08(self,x): return self._comp(x,[(self.ackley,1000/100),(self.griewank,1000/100),(self.discus,1e4/1e10),(self.rosenbrock,1),(self.happycat,1000/1e3),(self.escaffer6,1e4/2e7)],[10,20,30,40,50,60],[0,100,200,300,400,500])

    def cf09(self,x):
        fit=[
            self.hf05(x,self.OShift[0],self.M[0],self.SS[0*self.dim:1*self.dim]),
            self.hf06(x,self.OShift[1],self.M[1],self.SS[1*self.dim:2*self.dim]),
            self.hf07(x,self.OShift[2],self.M[2],self.SS[2*self.dim:3*self.dim]),
        ]
        return self._cf_cal(x,self.OShift[:3],[10,30,50],[0,100,200],fit)

    def cf10(self,x):
        fit=[
            self.hf05(x,self.OShift[0],self.M[0],self.SS[0*self.dim:1*self.dim]),
            self.hf08(x,self.OShift[1],self.M[1],self.SS[1*self.dim:2*self.dim]),
            self.hf09(x,self.OShift[2],self.M[2],self.SS[2*self.dim:3*self.dim]),
        ]
        return self._cf_cal(x,self.OShift[:3],[10,30,50],[0,100,200],fit)

    # ----------------------------- public -------------------------------
    def evaluate_one(self, x):
        x=np.asarray(x,float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Expected {self.dim} variables, got {x.size}")
        O=self.OShift[0]; M=self.M[0]; S=self.SS
        f=self.func_num
        if f==1: v=self.bent_cigar(x,O,M)
        elif f==3: v=self.zakharov(x,O,M)
        elif f==4: v=self.rosenbrock(x,O,M)
        elif f==5: v=self.rastrigin(x,O,M)
        elif f==6: v=self.schaffer_f7(x,O,M)
        elif f==7: v=self.bi_rastrigin(x,O,M)
        elif f==8: v=self.step_rastrigin(x,O,M)
        elif f==9: v=self.levy(x,O,M)
        elif f==10: v=self.schwefel(x,O,M)
        elif 11<=f<=20:
            v=getattr(self,f"hf{f-10:02d}")(x,O,M,S)
        elif 21<=f<=28:
            v=getattr(self,f"cf{f-20:02d}")(x)
        elif f==29: v=self.cf09(x)
        elif f==30: v=self.cf10(x)
        else: raise ValueError(f"Unsupported F{f}")
        return float(v + 100.0*f)

    def __call__(self, X):
        X=np.asarray(X,float)
        if X.ndim==1:
            return self.evaluate_one(X)
        if X.ndim!=2:
            raise ValueError("X must be shape (D,), (N,D), or (D,N).")
        # Python convention: N x D. Also accept official MATLAB convention D x N.
        if X.shape[1]==self.dim:
            return np.array([self.evaluate_one(row) for row in X])
        if X.shape[0]==self.dim:
            return np.array([self.evaluate_one(X[:,j]) for j in range(X.shape[1])])
        raise ValueError(f"Cannot infer candidate axis from X shape {X.shape} for D={self.dim}")


def cec17_func(X, func_num, data_dir="input_data"):
    """Convenience interface analogous to the supplied C/MATLAB function."""
    X=np.asarray(X,float)
    if X.ndim==1:
        dim=X.size
    elif X.ndim==2:
        # Prefer conventional N x D unless first dimension is a known CEC D.
        if X.shape[0] in VALID_DIMS and X.shape[1] not in VALID_DIMS:
            dim=X.shape[0]
        else:
            dim=X.shape[1]
    else:
        raise ValueError("X must be 1-D or 2-D.")
    return PurePythonCEC2017(func_num,dim,data_dir)(X)



def _cec2017_required_files(data_dir, dimension):
    """Return missing files required by corrected CEC-2017 F1 and F3-F30."""
    p = Path(data_dir)
    funcs = [1] + list(range(3, 31))
    missing = []

    for fid in funcs:
        if not (p / f"M_{fid}_D{dimension}.txt").is_file():
            missing.append(f"M_{fid}_D{dimension}.txt")
        if not (p / f"shift_data_{fid}.txt").is_file():
            missing.append(f"shift_data_{fid}.txt")

        # Hybrid functions F11-F20 require one shuffle vector.
        # Composition-hybrid F29/F30 require ten shuffle vectors.
        if 11 <= fid <= 20 or fid in (29, 30):
            if not (p / f"shuffle_data_{fid}_D{dimension}.txt").is_file():
                missing.append(f"shuffle_data_{fid}_D{dimension}.txt")

    return missing


def _is_complete_cec2017_data(data_dir, dimension):
    """Strictly accept only a complete corrected CEC-2017 data directory."""
    p = Path(data_dir)
    if not p.is_dir():
        return False

    # Explicitly reject known CEC-2020 paths.
    path_upper = str(p.resolve()).upper()
    if "CEC2020" in path_upper or "CEC-2020" in path_upper:
        return False

    return len(_cec2017_required_files(p, dimension)) == 0


def _running_in_colab():
    try:
        import google.colab
        return True
    except Exception:
        return False

def _download_and_prepare_cec2017_colab(dimension):
    import subprocess, shutil
    root=Path("/content/_CEC2017_OFFICIAL_AUTO")
    repo=root/"cec17-python"
    data=repo/"input_data"
    root.mkdir(parents=True,exist_ok=True)
    if _is_complete_cec2017_data(data,dimension):
        print(f"[CEC2017 COLAB] Reusing validated data: {data}")
        return str(data.resolve())
    if repo.exists():
        shutil.rmtree(repo,ignore_errors=True)
    print("="*112)
    print("CEC-2017 GOOGLE COLAB AUTO-SETUP")
    print("="*112)
    print("Downloading benchmark support data...")
    try:
        subprocess.run(["git","clone","--depth","1",
                        "https://github.com/oakboat/cec17-python.git",
                        str(repo)],check=True)
    except Exception as exc:
        raise RuntimeError("CEC-2017 Colab download failed. Check Internet access.") from exc
    missing=_cec2017_required_files(data,dimension)
    if missing:
        raise FileNotFoundError(
            f"Downloaded CEC-2017 data incomplete for D={dimension}; "
            f"missing {len(missing)} files; examples: {', '.join(missing[:10])}")
    print(f"[CEC2017 COLAB] Validated: {data}")
    return str(data.resolve())

def _find_cec2017_input_data():
    if _running_in_colab():
        env=os.environ.get("CEC2017_INPUT_DATA","").strip()
        candidates=[]
        if env:
            candidates.append(Path(env))
        candidates += [
            Path("/content/input_data"),
            Path("/content/CEC2017/input_data"),
            Path("/content/CEC-2017/input_data"),
            Path("/content/_CEC2017_OFFICIAL_AUTO/cec17-python/input_data")]
        for p in candidates:
            if _is_complete_cec2017_data(p,DIM):
                print(f"[CEC2017 DATA] Colab strict match found: {p.resolve()}")
                return str(p.resolve())
        return _download_and_prepare_cec2017_colab(DIM)

    roots=[]
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    cwd=Path.cwd()
    if cwd not in roots:
        roots.append(cwd)
    for base in list(roots):
        for parent in list(base.parents)[:3]:
            if parent not in roots:
                roots.append(parent)

    candidates=[]
    env=os.environ.get("CEC2017_INPUT_DATA","").strip()
    if env:
        candidates.append(Path(env))
    for root in roots:
        candidates += [
            root/"_CEC2017_OFFICIAL_AUTO"/"codes_extracted"/"codes"/"Matlab"/"input_data",
            root/"_CEC2017_OFFICIAL_AUTO"/"codes_extracted"/"codes"/"C"/"input_data",
            root/"_CEC2017_OFFICIAL_AUTO"/"cec17-python"/"input_data",
            root/"CEC2017"/"input_data", root/"CEC-2017"/"input_data",
            root/"input_data"]
    checked=set()
    for p in candidates:
        key=str(p).lower()
        if key in checked:
            continue
        checked.add(key)
        if _is_complete_cec2017_data(p,DIM):
            print(f"[CEC2017 DATA] Strict match found: {p.resolve()}")
            return str(p.resolve())

    for root in roots:
        try:
            if root.parent==root:
                continue
            for p in root.rglob("input_data"):
                up=str(p).upper()
                if "CEC2020" in up or "CEC-2020" in up:
                    continue
                if _is_complete_cec2017_data(p,DIM):
                    print(f"[CEC2017 DATA] Strict recursive match found: {p.resolve()}")
                    return str(p.resolve())
        except (PermissionError,OSError):
            continue

    # Windows / other local Python: if no valid local copy was found,
    # automatically obtain the CEC-2017 support data. Optimizer logic is untouched.
    import subprocess, shutil, urllib.request, zipfile

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd().resolve()

    auto_root = script_dir / "_CEC2017_OFFICIAL_AUTO"
    repo = auto_root / "cec17-python"
    data = repo / "input_data"
    auto_root.mkdir(parents=True, exist_ok=True)

    print("="*112)
    print("CEC-2017 WINDOWS/LOCAL AUTO-SETUP")
    print("="*112)
    print(f"Target directory : {auto_root}")
    print("Repository       : https://github.com/oakboat/cec17-python")
    print("No complete local CEC-2017 input_data was found.")
    print("Downloading benchmark support data automatically...")

    if _is_complete_cec2017_data(data, DIM):
        print(f"[CEC2017 DATA] Reusing validated data: {data.resolve()}")
        return str(data.resolve())

    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)

    git_error = None
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/oakboat/cec17-python.git", str(repo)],
            check=True
        )
    except Exception as exc:
        git_error = exc

    if not repo.exists():
        zip_path = auto_root / "cec17-python.zip"
        extract_dir = auto_root / "_zip_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        try:
            print("[CEC2017 SETUP] Git clone unavailable; trying GitHub ZIP fallback...")
            urllib.request.urlretrieve(
                "https://github.com/oakboat/cec17-python/archive/refs/heads/master.zip",
                zip_path
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            extracted = next(
                (p for p in extract_dir.iterdir()
                 if p.is_dir() and p.name.lower().startswith("cec17-python")),
                None
            )
            if extracted is None:
                raise FileNotFoundError("Could not locate extracted cec17-python repository.")
            shutil.move(str(extracted), str(repo))
        except Exception as zip_exc:
            raise RuntimeError(
                "CEC-2017 automatic Windows setup failed. Check Internet access. "
                f"Git clone error: {git_error!r}; ZIP fallback error: {zip_exc!r}"
            ) from zip_exc
        finally:
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except Exception:
                pass
            shutil.rmtree(extract_dir, ignore_errors=True)

    if not _is_complete_cec2017_data(data, DIM):
        found = None
        try:
            for p in repo.rglob("input_data"):
                if _is_complete_cec2017_data(p, DIM):
                    found = p
                    break
        except (PermissionError, OSError):
            found = None
        if found is not None:
            data = found

    missing = _cec2017_required_files(data, DIM)
    if missing:
        raise FileNotFoundError(
            f"Downloaded CEC-2017 data are incomplete for D={DIM}; "
            f"missing {len(missing)} required files. "
            f"Examples: {', '.join(missing[:10])}. Checked: {data}"
        )

    print(f"[CEC2017 SETUP] Validated input_data: {data.resolve()}")
    print("[CEC2017 SETUP] Pure-Python NumPy backend will be used; no DLL/MEX is required.")
    print("="*112)
    return str(data.resolve())


CEC2017_INPUT_DATA = None


class CEC2017Problem:
    """
    Counted TRUE-evaluation wrapper around the local pure-Python/NumPy
    translation of the supplied CEC-2017 source.

    No OPFUNU, MATLAB, MATLAB Engine, MEX, or cec2017-py package is used.
    F2 is excluded because it is deleted in the corrected CEC-2017 source.
    """

    def __init__(self, function_id=1, dimension=10):
        global CEC2017_INPUT_DATA

        self.function_id = int(function_id)
        self.dimension = int(dimension)

        if self.function_id == 2:
            raise ValueError("CEC-2017 F2 is deleted; use F1 and F3-F30.")
        if self.function_id not in ([1] + list(range(3, 31))):
            raise ValueError("Corrected CEC-2017 supports F1 and F3-F30.")
        if self.dimension not in (10, 30, 50, 100):
            raise ValueError("CEC-2017 dimensions for this study: 10, 30, 50, 100.")

        if CEC2017_INPUT_DATA is None:
            CEC2017_INPUT_DATA = _find_cec2017_input_data()

        self.problem = PurePythonCEC2017(
            func_num=self.function_id,
            dim=self.dimension,
            data_dir=CEC2017_INPUT_DATA,
        )

        self.backend_module = "local pure-Python NumPy port of supplied cec17_func.cpp"
        self.class_name = f"PurePythonCEC2017_F{self.function_id}"
        self.problem_name = f"CEC-2017 F{self.function_id}"

        self.lb = np.full(self.dimension, -100.0, dtype=float)
        self.ub = np.full(self.dimension, 100.0, dtype=float)

        # Reporting only; never supplied to the search.
        self.reference_optimum = float(100.0 * self.function_id)
        self.reference_source = "CEC-2017 prescribed additive bias (100 * function_id)"
        self.x_global = None

        self.true_fe = 0
        self.nonfinite_count = 0
        self._validate_metadata()
        self._validate_backend_shape()

    def _validate_metadata(self):
        if len(self.lb) != self.dimension or len(self.ub) != self.dimension:
            raise RuntimeError("Dimension/bounds metadata mismatch.")
        if np.any(self.ub <= self.lb):
            raise RuntimeError("Invalid CEC-2017 bounds.")

    def _validate_backend_shape(self):
        # Uncounted preflight only.
        x0 = np.zeros(self.dimension, dtype=float)
        y0 = float(self.problem(x0))
        if not np.isfinite(y0):
            raise RuntimeError(
                f"Pure-Python backend returned non-finite value for "
                f"F{self.function_id}, D={self.dimension}."
            )

    def metadata(self):
        return {
            "suite": "CEC-2017",
            "backend": self.backend_module,
            "backend_class": self.class_name,
            "problem_name": self.problem_name,
            "function_id": self.function_id,
            "dimension": self.dimension,
            "input_data": CEC2017_INPUT_DATA,
            "lb_min": float(np.min(self.lb)),
            "lb_max": float(np.max(self.lb)),
            "ub_min": float(np.min(self.ub)),
            "ub_max": float(np.max(self.ub)),
            "reference_optimum": float(self.reference_optimum),
            "reference_source": self.reference_source,
            "has_x_global": False,
        }

    def print_metadata(self):
        m = self.metadata()
        print("-" * 112)
        print("CEC-2017 PURE-PYTHON BACKEND SELF-VALIDATION")
        print("-" * 112)
        print(f"Backend                     : {m['backend']}")
        print(f"Function                    : F{m['function_id']}")
        print(f"Dimension                   : {m['dimension']}")
        print(f"Input data                  : {m['input_data']}")
        print(f"Bounds                      : [{m['lb_min']:.6g}, {m['ub_max']:.6g}]")
        print(f"Post-run reference optimum  : {m['reference_optimum']:.15e}")
        print("Known optimum vector used   : NO")
        print("Reference supplied to search: NO")
        print("-" * 112)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dimension:
            raise ValueError(f"Expected D={self.dimension}, received {x.size}.")
        x = np.clip(x, self.lb, self.ub)

        try:
            y = float(self.problem(x))
        finally:
            self.true_fe += 1

        if not np.isfinite(y):
            self.nonfinite_count += 1
            return float(V17_NONFINITE_PENALTY)
        return y


def resolve_true_fe_budget(dimension, requested_budget=MAX_TRUE_FES):
    """
    If USE_CEC2017_OFFICIAL_BUDGET=False:
        use requested MAX_TRUE_FES.

    If True:
        use 10000 * D TRUE evaluations.
    """
    if USE_CEC2017_OFFICIAL_BUDGET:
        return int(10000 * int(dimension))
    return int(requested_budget)

def normalize_x(X, lb, ub):
    X = np.asarray(X, dtype=float)
    return (X - lb) / np.maximum(ub - lb, 1e-15)

def denormalize_x(Z, lb, ub):
    Z = np.asarray(Z, dtype=float)
    return lb + Z * (ub - lb)

def clip01(Z):
    return np.clip(Z, 0.0, 1.0)

def robust_scale(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v.copy()
    lo = np.nanpercentile(v, 5.0)
    hi = np.nanpercentile(v, 95.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-15:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

def unique_rows(C):
    C = np.asarray(C, dtype=float)
    if len(C) == 0:
        return C
    rounded = np.round(C, 12)
    _, ids = np.unique(rounded, axis=0, return_index=True)
    return C[np.sort(ids)]

# Memory-only batching: preserve Euclidean distances and RNG draws.
DISTANCE_BLOCK_ROWS = 256
DISTANCE_BLOCK_COLS = 1024

def min_distances_blocked(C, X_true):
    """Exact nearest-reference distances without a full pairwise matrix."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    X_true = np.atleast_2d(np.asarray(X_true, dtype=float))
    result = np.full(len(C), np.inf, dtype=float)
    for i in range(0, len(C), DISTANCE_BLOCK_ROWS):
        stop = min(i + DISTANCE_BLOCK_ROWS, len(C))
        best = result[i:stop]
        for j in range(0, len(X_true), DISTANCE_BLOCK_COLS):
            distances = cdist(C[i:stop], X_true[j:j + DISTANCE_BLOCK_COLS])
            np.minimum(best, np.min(distances, axis=1), out=best)
            del distances
    return result

def remove_true_duplicates(C, X_true, tol=DUP_TOL):
    C = unique_rows(C)
    if len(C) == 0 or len(X_true) == 0:
        return C
    d = min_distances_blocked(C, X_true)
    return C[d > tol]

class TrueArchive:

    def __init__(self, dim):
        self.dim = int(dim)
        self.X = []
        self.y = []

    def add(self, x, y):
        self.X.append(np.asarray(x, dtype=float).copy())
        self.y.append(float(y))

    @property
    def X_array(self):
        if len(self.X) == 0:
            return np.empty((0, self.dim))
        return np.asarray(self.X, dtype=float)

    @property
    def y_array(self):
        return np.asarray(self.y, dtype=float)

    def best_index(self):
        return int(np.argmin(self.y_array))

    def best_x(self):
        return self.X_array[self.best_index()].copy()

    def best_y(self):
        return float(self.y_array[self.best_index()])

def lhs(n, dim, lb, ub, seed):
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    Z = sampler.random(n=n)
    return qmc.scale(Z, lb, ub)

class RegularizedRBF:
    """
    Gaussian regularized RBF in normalized decision space.
    """

    def __init__(self, ridge=1e-08):
        self.ridge = float(ridge)
        self.X = None
        self.w = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.sigma = None
        self.fitted = False

    @staticmethod
    def _sigma(X):
        if len(X) < 2:
            return 0.2
        if len(X) > 300:
            ids = np.linspace(0, len(X) - 1, 300).astype(int)
            Xs = X[ids]
        else:
            Xs = X
        D = cdist(Xs, Xs)
        d = D[np.triu_indices_from(D, k=1)]
        d = d[d > 1e-12]
        if len(d) == 0:
            return 0.2
        return max(float(np.median(d)), 0.001)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) < 2:
            raise ValueError('RBF requires at least two points.')
        self.X = X.copy()
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y))
        if self.y_std < 1e-12:
            self.y_std = 1.0
        yn = (y - self.y_mean) / self.y_std
        self.sigma = self._sigma(X)
        D2 = cdist(X, X, metric='sqeuclidean')
        Phi = np.exp(-D2 / self.sigma ** 2)
        A = Phi + self.ridge * np.eye(len(X))
        try:
            self.w = np.linalg.solve(A, yn)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(A, yn, rcond=None)[0]
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError('RBF has not been fitted.')
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # Bound prediction distance/kernel storage to at most 2 MiB per block.
        rows = max(1, min(DISTANCE_BLOCK_ROWS, 262144 // max(1, len(self.X))))
        prediction = np.empty(len(X), dtype=float)
        for start in range(0, len(X), rows):
            stop = min(start + rows, len(X))
            kernel = cdist(X[start:stop], self.X, metric='sqeuclidean')
            np.negative(kernel, out=kernel)
            kernel /= self.sigma ** 2
            np.exp(kernel, out=kernel)
            prediction[start:stop] = self.y_mean + self.y_std * (kernel @ self.w)
            del kernel
        return prediction

def select_global_training(Xz, y, max_points, rng):
    if len(y) <= max_points:
        return (Xz, y)
    order = np.argsort(y)
    n_best = int(0.6 * max_points)
    best_ids = order[:n_best]
    remainder = np.setdiff1d(np.arange(len(y)), best_ids)
    n_other = max_points - n_best
    other_ids = rng.choice(remainder, size=n_other, replace=False)
    ids = np.concatenate([best_ids, other_ids])
    return (Xz[ids], y[ids])

@dataclass
class Basin:
    basin_id: int
    center: np.ndarray
    best_z: np.ndarray
    best_y: float
    radius: float = TR_RADIUS_INITIAL
    population: np.ndarray = field(default_factory=lambda: np.empty((0, DIM)))
    failure_count: int = 0
    last_improvement_fe: int = 0
    restarted_this_cycle: bool = False
    success_history: list = field(default_factory=list)
    true_bmr_evals: int = 0
    true_bmr_successes: int = 0
    credit_history: list = field(default_factory=list)
    # V18 saturation memory
    last_significant_improvement_fe: int = 0
    quarantine_until_fe: int = 0
    saturation_count: int = 0

def discover_basin_centers(Xz, y, max_basins, elite_fraction, seed):
    n_elite = max(max_basins * 4, int(np.ceil(elite_fraction * len(y))))
    n_elite = min(n_elite, len(y))
    ids = np.argsort(y)[:n_elite]
    Xe = Xz[ids]
    ye = y[ids]
    k = min(max_basins, max(1, len(Xe) // 5))
    if k == 1:
        labels = np.zeros(len(Xe), dtype=int)
        centers = np.mean(Xe, axis=0, keepdims=True)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xe)
        centers = km.cluster_centers_
    out = []
    for b in range(k):
        loc = np.where(labels == b)[0]
        if len(loc) == 0:
            continue
        local_best = loc[np.argmin(ye[loc])]
        out.append((centers[b].copy(), Xe[local_best].copy(), float(ye[local_best])))
    out.sort(key=lambda q: q[2])
    return out

class BasinManager:
    """
    Persistent basin state.

    Key V3 behavior
    ---------------
    * Radius shrinks only after FAILURE_PATIENCE failures.
    * Radius expands immediately after a genuine basin improvement.
    * Stale minimum-radius basins can be restarted from BWR candidates.
    """

    def __init__(self, max_basins, local_pop_size, rng):
        self.max_basins = int(max_basins)
        self.local_pop_size = int(local_pop_size)
        self.rng = rng
        self.basins = []

    def _init_local_population(self, center, radius):
        center = np.asarray(center, dtype=float)
        P = center + self.rng.uniform(-radius, radius, size=(self.local_pop_size, center.size))
        return clip01(P)

    def update_from_archive(self, Xz, y, seed, current_fe):
        info = discover_basin_centers(Xz, y, self.max_basins, ELITE_FRACTION, seed)
        old_basins = self.basins
        used_old = set()
        new_basins = []
        for b_id, (center, best_z, best_y) in enumerate(info):
            old = None
            old_idx = None
            if len(old_basins) > 0:
                old_centers = np.vstack([q.center for q in old_basins])
                distances = np.linalg.norm(old_centers - center, axis=1)
                order = np.argsort(distances)
                for j in order:
                    if int(j) not in used_old and distances[j] < 0.25:
                        old = old_basins[int(j)]
                        old_idx = int(j)
                        used_old.add(int(j))
                        break
            if old is None:
                radius = TR_RADIUS_INITIAL
                failure_count = 0
                last_improvement_fe = current_fe
                success_history = []
                true_bmr_evals = 0
                true_bmr_successes = 0
                credit_history = []
                last_significant_improvement_fe = current_fe
                quarantine_until_fe = 0
                saturation_count = 0
                population = self._init_local_population(best_z, radius)
            else:
                radius = old.radius
                failure_count = old.failure_count
                last_improvement_fe = old.last_improvement_fe
                success_history = list(old.success_history)
                true_bmr_evals = int(old.true_bmr_evals)
                true_bmr_successes = int(old.true_bmr_successes)
                credit_history = list(old.credit_history)
                last_significant_improvement_fe = int(getattr(old, 'last_significant_improvement_fe', old.last_improvement_fe))
                quarantine_until_fe = int(getattr(old, 'quarantine_until_fe', 0))
                saturation_count = int(getattr(old, 'saturation_count', 0))
                population = old.population.copy()
                if best_y < old.best_y:
                    old_best_y = float(old.best_y)
                    last_improvement_fe = current_fe
                    failure_count = 0
                    radius = min(TR_RADIUS_MAX, radius * TR_EXPAND)
                    gain = max(0.0, old_best_y - float(best_y))
                    threshold = max(MAG_SUCCESS_ABS_GAIN, MAG_SUCCESS_REL_GAIN * max(abs(old_best_y), 1.0))
                    if gain >= threshold:
                        last_significant_improvement_fe = current_fe
                n_refresh = max(2, int(0.2 * self.local_pop_size))
                refresh = best_z + self.rng.uniform(-radius, radius, size=(n_refresh, DIM))
                refresh = clip01(refresh)
                if len(population) != self.local_pop_size:
                    population = self._init_local_population(best_z, radius)
                else:
                    population[:n_refresh] = refresh
            new_basins.append(Basin(basin_id=b_id, center=center.copy(), best_z=best_z.copy(), best_y=float(best_y), radius=float(radius), population=population.copy(), failure_count=int(failure_count), last_improvement_fe=int(last_improvement_fe), restarted_this_cycle=False, success_history=success_history[-BASIN_SUCCESS_WINDOW:], true_bmr_evals=true_bmr_evals, true_bmr_successes=true_bmr_successes, credit_history=credit_history[-BASIN_CREDIT_WINDOW:], last_significant_improvement_fe=int(last_significant_improvement_fe), quarantine_until_fe=int(quarantine_until_fe), saturation_count=int(saturation_count)))
        self.basins = new_basins

    def record_true_bmr_result(self, basin_idx, z, true_y, current_fe):
        """
        Update trust region using basin-level success.

        IMPORTANT:
        Radius shrinks only after FAILURE_PATIENCE consecutive failures.
        """
        if basin_idx < 0 or basin_idx >= len(self.basins):
            return False
        b = self.basins[basin_idx]
        previous_best = float(b.best_y)
        improved = true_y < previous_best
        b.success_history.append(1 if improved else 0)
        b.success_history = b.success_history[-BASIN_SUCCESS_WINDOW:]
        b.true_bmr_evals += 1
        if improved:
            b.true_bmr_successes += 1
        b.credit_history.append(1 if improved else 0)
        b.credit_history = b.credit_history[-BASIN_CREDIT_WINDOW:]
        if improved:
            b.best_y = float(true_y)
            b.best_z = np.asarray(z, dtype=float).copy()
            b.center = 0.5 * b.center + 0.5 * b.best_z
            b.failure_count = 0
            b.last_improvement_fe = int(current_fe)
            gain = max(0.0, previous_best - float(true_y))
            threshold = max(MAG_SUCCESS_ABS_GAIN, MAG_SUCCESS_REL_GAIN * max(abs(previous_best), 1.0))
            if gain >= threshold:
                b.last_significant_improvement_fe = int(current_fe)
                b.quarantine_until_fe = 0
            b.radius = min(TR_RADIUS_MAX, b.radius * TR_EXPAND)
        else:
            b.failure_count += 1
            if b.failure_count >= FAILURE_PATIENCE:
                b.radius = max(TR_RADIUS_MIN, b.radius * TR_SHRINK)
                b.failure_count = 0
        return improved

    def stale_indices(self, current_fe):
        stale = []
        for i, b in enumerate(self.basins):
            at_min_radius = b.radius <= TR_RADIUS_MIN + 1e-12
            long_no_improvement = current_fe - b.last_improvement_fe >= STALE_BASIN_TRUE_FES
            if at_min_radius and long_no_improvement:
                stale.append(i)
        return stale

    def restart_stale_from_bwr(self, C_bwr, pred_bwr, current_fe, Xz_true=None, y_true=None):
        """
        Replace stale minimum-radius basins with promising BWR seeds.

        Returns
        -------
        restart_seeds : list of (basin_idx, z, predicted_f)
            These seeds are later given priority for TRUE evaluation.
        """
        stale = self.stale_indices(current_fe)
        if len(stale) == 0 or len(C_bwr) == 0:
            return []
        C_bwr = np.asarray(C_bwr, dtype=float)
        pred_bwr = np.asarray(pred_bwr, dtype=float)
        restart_seeds = []
        used_candidate_ids = set()
        for stale_idx in stale:
            surviving_centers = []
            for j, b in enumerate(self.basins):
                if j != stale_idx:
                    surviving_centers.append(b.center)
            if len(surviving_centers) > 0:
                centers = np.vstack(surviving_centers)
                min_d = min_distances_blocked(C_bwr, centers)
            else:
                min_d = np.ones(len(C_bwr))
            if Xz_true is not None and y_true is not None and (len(Xz_true) > 0):
                elite = elite_archive_points(Xz_true, y_true, RESTART_ELITE_FRACTION)
                elite_d = min_distances_blocked(C_bwr, elite)
            else:
                elite_d = np.ones(len(C_bwr))
            valid = np.where((min_d >= BASIN_RESTART_MIN_DISTANCE) & (elite_d >= RESTART_MIN_ELITE_DISTANCE))[0]
            if len(valid) == 0:
                combined_distance = np.minimum(min_d, elite_d)
                order_far = np.argsort(-combined_distance)
                valid = order_far[:max(10, min(100, len(order_far)))]
            valid = np.asarray([i for i in valid if int(i) not in used_candidate_ids], dtype=int)
            if len(valid) == 0:
                break
            p = robust_scale(pred_bwr[valid])
            d = robust_scale(min_d[valid])
            score = p - RESTART_DISTANCE_REWARD * d
            candidate_id = int(valid[np.argmin(score)])
            used_candidate_ids.add(candidate_id)
            z0 = C_bwr[candidate_id].copy()
            pred0 = float(pred_bwr[candidate_id])
            b = self.basins[stale_idx]
            b.center = z0.copy()
            b.best_z = z0.copy()
            b.best_y = np.inf
            b.radius = TR_RADIUS_INITIAL
            b.failure_count = 0
            b.last_improvement_fe = int(current_fe)
            b.restarted_this_cycle = True
            b.success_history = []
            b.true_bmr_evals = 0
            b.true_bmr_successes = 0
            b.credit_history = []
            b.last_significant_improvement_fe = int(current_fe)
            b.quarantine_until_fe = 0
            b.population = self._init_local_population(z0, b.radius)
            restart_seeds.append((stale_idx, z0.copy(), pred0))
        return restart_seeds

@njit(cache=True)
def _bwr_kernel(X, XR, T, R1, R2, R3, R4, best, worst):
    n_offspring, d = X.shape
    C = np.empty((n_offspring, d), dtype=np.float64)
    for k in range(n_offspring):
        for j in range(d):
            guided = X[k, j] + R1[k, j] * (best[j] - T[k] * XR[k, j]) - R2[k, j] * (worst[j] - XR[k, j])
            v = guided if R4[k, j] > 0.5 else R3[k, j]
            if v < 0.0: v = 0.0
            elif v > 1.0: v = 1.0
            C[k, j] = v
    return C

@njit(cache=True)
def _bmr_kernel(X, XR, T, R1, R2, R3, R4, best, mean, lo, hi):
    n_offspring, d = X.shape
    C = np.empty((n_offspring, d), dtype=np.float64)
    for k in range(n_offspring):
        for j in range(d):
            guided = X[k, j] + R1[k, j] * (best[j] - T[k] * mean[j]) + R2[k, j] * (best[j] - XR[k, j])
            local_random = lo[j] + R3[k, j] * (hi[j] - lo[j])
            v = guided if R4[k, j] > 0.5 else local_random
            if v < lo[j]: v = lo[j]
            elif v > hi[j]: v = hi[j]
            if v < 0.0: v = 0.0
            elif v > 1.0: v = 1.0
            C[k, j] = v
    return C

def _operator_random_draws(P, n_offspring, rng):
    n, d = P.shape
    X = np.empty((n_offspring, d)); XR = np.empty((n_offspring, d)); T = np.empty(n_offspring)
    R1 = np.empty((n_offspring, d)); R2 = np.empty((n_offspring, d)); R3 = np.empty((n_offspring, d)); R4 = np.empty((n_offspring, d))
    for k in range(n_offspring):
        X[k] = P[rng.integers(n)]; XR[k] = P[rng.integers(n)]; T[k] = rng.choice([1.0, 2.0])
        R1[k] = rng.random(d); R2[k] = rng.random(d); R3[k] = rng.random(d); R4[k] = rng.random(d)
    return X, XR, T, R1, R2, R3, R4

def bwr_offspring(population, fitness, n_offspring, rng):
    """BWR global operator; RNG sequence is unchanged, arithmetic is Numba-JIT compiled."""
    P = np.asarray(population, dtype=float); f = np.asarray(fitness, dtype=float)
    best = P[np.argmin(f)]; worst = P[np.argmax(f)]
    X, XR, T, R1, R2, R3, R4 = _operator_random_draws(P, n_offspring, rng)
    return _bwr_kernel(X, XR, T, R1, R2, R3, R4, best, worst)

def bmr_trust_region_offspring(population, pred_fitness, center, radius, n_offspring, rng):
    """BMR local trust-region operator; RNG sequence is unchanged, arithmetic is Numba-JIT compiled."""
    P = np.asarray(population, dtype=float); f = np.asarray(pred_fitness, dtype=float)
    best = P[np.argmin(f)]; mean = np.mean(P, axis=0); lo = clip01(center - radius); hi = clip01(center + radius)
    X, XR, T, R1, R2, R3, R4 = _operator_random_draws(P, n_offspring, rng)
    return _bmr_kernel(X, XR, T, R1, R2, R3, R4, best, mean, lo, hi)

def initialize_global_population(Xz, y, pop_size, rng):
    order = np.argsort(y)
    n_elite = min(int(0.7 * pop_size), len(y))
    elite_ids = order[:n_elite]
    remainder = np.setdiff1d(np.arange(len(y)), elite_ids)
    n_other = pop_size - n_elite
    if n_other > 0:
        if len(remainder) >= n_other:
            other_ids = rng.choice(remainder, size=n_other, replace=False)
        else:
            other_ids = rng.choice(np.arange(len(y)), size=n_other, replace=True)
        ids = np.concatenate([elite_ids, other_ids])
    else:
        ids = elite_ids
    return Xz[ids].copy()

def update_global_population(current_pop, current_pred, candidates, candidate_pred, pop_size):
    P = np.vstack([current_pop, candidates])
    f = np.concatenate([current_pred, candidate_pred])
    order = np.argsort(f)
    return P[order[:pop_size]].copy()

def truth_anchor_global_population(current_pop, global_model, Xz_true, y_true, pop_size, rng):
    """
    Rebuild a fixed fraction of the BWR parent population from TRUE
    evaluated archive points every cycle.

    This prevents the persistent global population from becoming a
    surrogate-only evolutionary population.
    """
    n_anchor = int(round(TRUE_ANCHOR_FRACTION * pop_size))
    n_anchor = min(max(1, n_anchor), pop_size, len(y_true))
    n_anchor_elite = min(n_anchor, max(1, int(round(TRUE_ANCHOR_ELITE_FRACTION * n_anchor))))
    order = np.argsort(y_true)
    elite_ids = order[:n_anchor_elite]
    anchors = [Xz_true[elite_ids]]
    n_diverse = n_anchor - n_anchor_elite
    if n_diverse > 0:
        remaining = np.setdiff1d(np.arange(len(y_true)), elite_ids)
        if len(remaining) > 0:
            chosen = list(elite_ids)
            for _ in range(min(n_diverse, len(remaining))):
                base = Xz_true[np.asarray(chosen, dtype=int)]
                d = min_distances_blocked(Xz_true[remaining], base)
                j = int(np.argmax(d))
                chosen_id = int(remaining[j])
                chosen.append(chosen_id)
                remaining = remaining[remaining != chosen_id]
                if len(remaining) == 0:
                    break
            diverse_ids = np.asarray(chosen[n_anchor_elite:], dtype=int)
            if len(diverse_ids) > 0:
                anchors.append(Xz_true[diverse_ids])
    true_anchor_pop = np.vstack(anchors)
    n_surrogate = pop_size - len(true_anchor_pop)
    if n_surrogate > 0:
        pred = global_model.predict(current_pop)
        order_pred = np.argsort(pred)
        surrogate_part = current_pop[order_pred[:min(n_surrogate, len(current_pop))]]
        if len(surrogate_part) < n_surrogate:
            extra = rng.random((n_surrogate - len(surrogate_part), Xz_true.shape[1]))
            surrogate_part = np.vstack([surrogate_part, extra])
        out = np.vstack([true_anchor_pop, surrogate_part])
    else:
        out = true_anchor_pop[:pop_size]
    return out[:pop_size].copy()

def recent_true_improvement(best_history, window_true_fes):
    if len(best_history) < 2:
        return False
    w = min(int(window_true_fes), len(best_history) - 1)
    older = best_history[-w - 1]
    current = best_history[-1]
    threshold = 1e-10 * max(1.0, abs(older))
    return older - current > threshold

def incumbent_basin_index(basins, incumbent_z):
    if len(basins) == 0:
        return -1
    centers = np.vstack([b.center for b in basins])
    d = np.linalg.norm(centers - incumbent_z, axis=1)
    return int(np.argmin(d))

def incumbent_basin_success_rate(basins, incumbent_z):
    idx = incumbent_basin_index(basins, incumbent_z)
    if idx < 0:
        return 0.0
    hist = basins[idx].success_history
    if len(hist) == 0:
        return 0.0
    return float(np.mean(hist))

def generate_incumbent_refinement_candidates(incumbent_z, local_model, Xz_true, rng):
    """
    Dedicated incumbent-centered local surrogate refinement.

    This is separate from the BMR basin population. Candidate generation
    uses a small deterministic/stochastic local cloud; the TRUE objective
    is never used for cheap ranking.
    """
    d = incumbent_z.size
    local_dist = min_distances_blocked(incumbent_z[None, :], Xz_true)[0]
    radius = max(INCUMBENT_REFINE_MIN_RADIUS, min(INCUMBENT_REFINE_RADIUS, max(0.5 * INCUMBENT_REFINE_RADIUS, 2.0 * local_dist)))
    C = incumbent_z + rng.uniform(-radius, radius, size=(INCUMBENT_REFINE_CANDIDATES, d))
    C = clip01(C)
    C = remove_true_duplicates(C, Xz_true)
    if len(C) == 0:
        return np.empty((0, d))
    pred = local_model.predict(C)
    keep = min(max(20, 4 * d), len(C))
    ids = np.argsort(pred)[:keep]
    return C[ids]

def build_local_rbf(Xz, y, center, max_points):
    d = np.linalg.norm(Xz - center, axis=1)
    ids = np.argsort(d)[:min(max_points, len(Xz))]
    if len(ids) < 2:
        return None
    model = RegularizedRBF(ridge=RBF_RIDGE)
    model.fit(Xz[ids], y[ids])
    return model

@njit(cache=True)
def _nc_reduce_kernel(D, ids):
    n, k = ids.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        m = -np.inf
        for j in range(k):
            v = D[i, ids[i, j]]
            if v > m: m = v
        out[i] = m
    return out

def nc_values(C, X_true, n_neighbors):
    """Exact kth-neighbor distances using bounded pairwise-distance blocks."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if len(C) == 0:
        return np.empty(0)
    X_true = np.atleast_2d(np.asarray(X_true, dtype=float))
    k = min(int(n_neighbors), len(X_true))
    if k < 1:
        raise ValueError("nc_values requires a nonempty archive and n_neighbors >= 1")
    result = np.empty(len(C), dtype=float)
    for i in range(0, len(C), DISTANCE_BLOCK_ROWS):
        stop = min(i + DISTANCE_BLOCK_ROWS, len(C))
        nearest = np.empty((stop - i, 0), dtype=float)
        for j in range(0, len(X_true), DISTANCE_BLOCK_COLS):
            distances = cdist(C[i:stop], X_true[j:j + DISTANCE_BLOCK_COLS])
            merged = np.concatenate((nearest, distances), axis=1)
            del distances
            keep = min(k, merged.shape[1])
            merged.partition(keep - 1, axis=1)
            nearest = merged[:, :keep].copy()
            del merged
        result[i:stop] = np.max(nearest, axis=1)
    return result

@njit(cache=True)
def _pareto_scan_kernel(order, f2):
    selected = np.empty(order.size, dtype=np.int64); nsel = 0; best_f2 = np.inf
    for q in range(order.size):
        idx = order[q]
        if f2[idx] < best_f2:
            selected[nsel] = idx; nsel += 1; best_f2 = f2[idx]
    return selected[:nsel]

def pareto_front_2d(f1, f2):
    """First nondominated front for minimization of f1 and f2."""
    order = np.lexsort((f2, f1))
    return _pareto_scan_kernel(order.astype(np.int64), np.asarray(f2, dtype=np.float64))

class ReliabilityTracker:
    """
    Rank-dominant reliability:
        70% Spearman ranking agreement
        30% robust absolute-error score
    """

    def __init__(self, window=40):
        self.window = int(window)
        self.true_y = []
        self.pred_y = []

    def add(self, true_y, pred_y):
        self.true_y.append(float(true_y))
        self.pred_y.append(float(pred_y))
        if len(self.true_y) > self.window:
            self.true_y = self.true_y[-self.window:]
            self.pred_y = self.pred_y[-self.window:]

    def reliability(self):
        if len(self.true_y) < 8:
            return 0.4
        yt = np.asarray(self.true_y, dtype=float)
        yp = np.asarray(self.pred_y, dtype=float)
        rho, _ = spearmanr(yt, yp)
        if not np.isfinite(rho):
            rho = 0.0
        rank_score = (rho + 1.0) / 2.0
        scale = max(np.percentile(yt, 75) - np.percentile(yt, 25), np.std(yt), 1e-12)
        mae = np.median(np.abs(yt - yp))
        abs_score = np.exp(-mae / (scale + 1e-12))
        R = 0.7 * rank_score + 0.3 * abs_score
        return float(np.clip(R, 0.0, 1.0))

def basin_credit_rate(basin):
    """
    Archive-backed/persistent BMR productivity estimate.
    """
    if len(basin.credit_history) > 0:
        return float(np.mean(basin.credit_history))
    if basin.true_bmr_evals <= 0:
        return 0.0
    return float(basin.true_bmr_successes / basin.true_bmr_evals)

def incumbent_basin_credit(basins, incumbent_z):
    idx = incumbent_basin_index(basins, incumbent_z)
    if idx < 0:
        return 0.0
    return basin_credit_rate(basins[idx])

def improvement_over_window(best_history, window_true_fes):
    if len(best_history) < 2:
        return 0.0
    w = min(int(window_true_fes), len(best_history) - 1)
    older = float(best_history[-w - 1])
    current = float(best_history[-1])
    return (older - current) / max(abs(older), 1.0)

def should_intensify(reliability, best_history, incumbent_credit):
    """
    Intensification requires:
      * reasonably high model reliability,
      * some recent verified progress in the broader window,
      * weak progress in the short window,
      * positive incumbent-basin productivity.
    """
    broad_progress = recent_true_improvement(best_history, RECENT_IMPROVEMENT_WINDOW_TRUE_FES)
    short_rel_improvement = improvement_over_window(best_history, INTENSIFY_STALL_WINDOW_TRUE_FES)
    return reliability >= INTENSIFY_RELIABILITY_MIN and broad_progress and (short_rel_improvement <= 0.0001) and (incumbent_credit >= LOCAL_BASIN_CREDIT_MIN)

def generate_multiscale_incumbent_candidates(incumbent_z, local_model, Xz_true, rng):
    """
    V6 multi-scale incumbent refinement.

    Generates three nested local clouds around the verified incumbent.
    """
    groups = []
    for radius in INTENSIFY_RADII:
        C = incumbent_z + rng.uniform(-radius, radius, size=(INTENSIFY_CANDIDATES_PER_RADIUS, incumbent_z.size))
        C = clip01(C)
        C = remove_true_duplicates(C, Xz_true)
        if len(C) == 0:
            continue
        pred = local_model.predict(C)
        keep = min(max(20, 3 * incumbent_z.size), len(C))
        ids = np.argsort(pred)[:keep]
        groups.append(C[ids])
    if len(groups) == 0:
        return np.empty((0, incumbent_z.size))
    return unique_rows(np.vstack(groups))

def elite_archive_points(Xz_true, y_true, fraction=RESTART_ELITE_FRACTION):
    n = max(1, int(np.ceil(fraction * len(y_true))))
    ids = np.argsort(y_true)[:n]
    return Xz_true[ids]

def true_stall_age(current_fe, last_global_improvement_fe):
    return max(0, int(current_fe) - int(last_global_improvement_fe))

def progress_triggered_intensify(reliability, stall_age, recent_local_success=False):
    """
    V8 persistent intensification.

    There is NO upper stall-age cutoff.

    Intensification activates when:
      * the surrogate is sufficiently reliable, and
      * the search has stalled for a moderate number of TRUE FEs,
        OR there was a verified local-refinement success recently.
    """
    return reliability >= INTENSIFY_RELIABILITY_MIN and (stall_age >= INTENSIFY_START_STALL_FES or recent_local_success)

def generate_coordinate_poll_candidates(incumbent_z, Xz_true, stall_age=0):
    """
    Structured coordinate poll around the TRUE incumbent.

    V8 adds one adaptive radius:
        radius_adapt = clip(0.05 / sqrt(1 + stall_age/20), min, max)

    This gradually enables finer local moves during prolonged
    stagnation without using the known optimum.
    """
    incumbent_z = np.asarray(incumbent_z, dtype=float)
    d = incumbent_z.size
    adaptive_radius = 0.05 / np.sqrt(1.0 + max(0.0, float(stall_age)) / 20.0)
    adaptive_radius = float(np.clip(adaptive_radius, ADAPTIVE_POLL_MIN, ADAPTIVE_POLL_MAX))
    radii = sorted(set(list(POLL_RADII) + [adaptive_radius]), reverse=True)
    points = []
    for radius in radii:
        for j in range(d):
            zp = incumbent_z.copy()
            zm = incumbent_z.copy()
            zp[j] += radius
            zm[j] -= radius
            points.append(clip01(zp))
            points.append(clip01(zm))
    if len(points) == 0:
        return np.empty((0, d))
    C = unique_rows(np.vstack(points))
    return remove_true_duplicates(C, Xz_true)

def surrogate_local_optimum_candidates(incumbent_z, local_model, Xz_true, rng, radius=SURR_LOCAL_RADIUS, n_restarts=None):
    """
    Optimize the LOCAL RBF surrogate using multi-start L-BFGS-B.

    IMPORTANT:
    This optimizer NEVER calls the true CEC objective. All optimization
    here is on the cheap surrogate. Only selected resulting points are
    later sent to TRUE evaluation.
    """
    incumbent_z = np.asarray(incumbent_z, dtype=float)
    lo = clip01(incumbent_z - radius)
    hi = clip01(incumbent_z + radius)
    bounds = list(zip(lo.tolist(), hi.tolist()))
    if n_restarts is None:
        n_restarts = SURR_LOCAL_RESTARTS
    n_restarts = max(1, int(n_restarts))
    starts = [incumbent_z.copy()]
    for _ in range(n_restarts - 1):
        starts.append(lo + rng.random(incumbent_z.size) * (hi - lo))
    candidates = []

    def surr_obj(z):
        return float(local_model.predict(np.asarray(z, dtype=float)[None, :])[0])
    for x0 in starts:
        try:
            res = minimize(surr_obj, x0=np.asarray(x0, dtype=float), method='L-BFGS-B', bounds=bounds, options={'maxiter': SURR_LOCAL_MAXITER, 'ftol': 1e-12})
            z = clip01(np.asarray(res.x, dtype=float))
            candidates.append(z)
        except Exception:
            candidates.append(np.asarray(x0, dtype=float))
    C = unique_rows(np.vstack(candidates))
    return remove_true_duplicates(C, Xz_true)

def generate_novelty_candidates(Xz_true, dimension, rng, n_keep=NOVELTY_KEEP):
    """
    Low-discrepancy farthest-point recovery candidates.

    A Sobol pool is generated cheaply and ranked only by decision-space
    distance from TRUE evaluated samples. No known optimum is used.
    """
    pool_size = max(n_keep, NOVELTY_POOL_MULTIPLIER * n_keep)
    m = int(np.ceil(np.log2(max(2, pool_size))))
    seed = int(rng.integers(0, 2 ** 31 - 1))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    C = sampler.random_base2(m=m)
    C = remove_true_duplicates(C, Xz_true)
    if len(C) <= n_keep:
        return C
    min_d = min_distances_blocked(C, Xz_true)
    ids = np.argsort(-min_d)[:n_keep]
    return C[ids]

def generate_deep_novelty_candidates(Xz_true, dimension, rng):
    """
    Stronger farthest-point low-discrepancy search for DEEP_RECOVERY.
    """
    n_keep = DEEP_NOVELTY_KEEP
    pool_size = max(n_keep, DEEP_NOVELTY_POOL_MULTIPLIER * n_keep)
    m = int(np.ceil(np.log2(max(2, pool_size))))
    seed = int(rng.integers(0, 2 ** 31 - 1))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    C = sampler.random_base2(m=m)
    C = remove_true_duplicates(C, Xz_true)
    if len(C) <= n_keep:
        return C
    min_d = min_distances_blocked(C, Xz_true)
    ids = np.argsort(-min_d)[:n_keep]
    return C[ids]

def reseed_global_population_for_deep_recovery(global_pop, Xz_true, y_true, deep_novelty, pop_size, rng):
    """
    Preserve TRUE elites but replace part of the global population with
    novelty points at DEEP_RECOVERY entry.
    """
    n_replace = int(round(DEEP_RECOVERY_RESEED_FRACTION * pop_size))
    n_replace = min(max(1, n_replace), pop_size)
    n_keep = pop_size - n_replace
    order = np.argsort(y_true)
    true_elites = Xz_true[order[:min(n_keep, len(y_true))]]
    if len(true_elites) < n_keep:
        extra = rng.random((n_keep - len(true_elites), Xz_true.shape[1]))
        true_elites = np.vstack([true_elites, extra])
    if len(deep_novelty) >= n_replace:
        novelty = deep_novelty[:n_replace]
    else:
        novelty = np.vstack([deep_novelty, rng.random((n_replace - len(deep_novelty), Xz_true.shape[1]))])
    return np.vstack([true_elites[:n_keep], novelty[:n_replace]])

def v17_shock_nonincumbent_basins(basin_manager, incumbent_z, Xz_true, y_true, rng, current_fe):
    """Re-seed non-incumbent basins with far Sobol points after prolonged meaningful stagnation."""
    basins = basin_manager.basins
    if len(basins) <= 1: return 0
    inc_idx = incumbent_basin_index(basins, incumbent_z)
    C = generate_deep_novelty_candidates(Xz_true, incumbent_z.size, rng)
    if len(C) == 0: return 0
    centers = np.vstack([b.center for b in basins])
    score = min_distances_blocked(C, Xz_true)
    order = np.argsort(-score)
    replace_ids = [i for i in range(len(basins)) if i != inc_idx]
    n_replace = max(1, int(np.ceil(V17_BASIN_SHOCK_FRACTION * len(replace_ids))))
    n_replace = min(n_replace, len(replace_ids), len(order))
    count=0
    for bi, ci in zip(replace_ids[:n_replace], order[:n_replace]):
        z0 = C[int(ci)].copy(); b = basins[bi]
        b.center=z0.copy(); b.best_z=z0.copy(); b.best_y=np.inf; b.radius=TR_RADIUS_INITIAL
        b.failure_count=0; b.last_improvement_fe=int(current_fe); b.restarted_this_cycle=True
        b.success_history=[]; b.true_bmr_evals=0; b.true_bmr_successes=0; b.credit_history=[]
        b.population=basin_manager._init_local_population(z0,b.radius); count += 1
    return count


def v18_is_quarantined(basin, current_fe):
    return int(current_fe) < int(getattr(basin, 'quarantine_until_fe', 0))

def v18_basin_success_rate(basin):
    hist = list(getattr(basin, 'credit_history', []))
    if len(hist) == 0:
        if int(getattr(basin, 'true_bmr_evals', 0)) <= 0:
            return 0.0
        return float(getattr(basin, 'true_bmr_successes', 0) / max(1, getattr(basin, 'true_bmr_evals', 0)))
    return float(np.mean(hist))

def v18_detect_saturated_basins(basin_manager, current_fe, meaningful_stall_age):
    """Identify locally solved/unproductive basins using TRUE data only."""
    saturated = []
    if meaningful_stall_age < V18_SATURATION_TRIGGER_FES:
        return saturated
    for i, b in enumerate(basin_manager.basins):
        if v18_is_quarantined(b, current_fe):
            continue
        n_eval = int(getattr(b, 'true_bmr_evals', 0))
        sig_fe = int(getattr(b, 'last_significant_improvement_fe', b.last_improvement_fe))
        age = int(current_fe) - sig_fe
        sr = v18_basin_success_rate(b)
        if n_eval >= V18_SATURATION_MIN_BMR_EVALS and age >= V18_SATURATION_TRIGGER_FES and sr <= V18_SATURATION_MAX_SUCCESS_RATE:
            saturated.append(i)
    return saturated

def v18_maximin_sobol_seeds(Xz_true, existing_centers, n_seeds, dimension, rng):
    """Sequential maximin Sobol seeds far from TRUE archive and surviving centers."""
    n_seeds = int(max(0, n_seeds))
    if n_seeds == 0:
        return np.empty((0, dimension))
    pool_size = max(V18_MAXIMIN_POOL, 64 * n_seeds)
    m = int(np.ceil(np.log2(max(2, pool_size))))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
    C = sampler.random_base2(m=m)
    C = remove_true_duplicates(C, Xz_true)
    if len(C) == 0:
        return np.empty((0, dimension))
    reference = np.asarray(Xz_true, dtype=float)
    if existing_centers is not None and len(existing_centers) > 0:
        reference = np.vstack([reference, np.asarray(existing_centers, dtype=float)])
    selected = []
    available = np.ones(len(C), dtype=bool)
    for _ in range(n_seeds):
        ids = np.where(available)[0]
        if len(ids) == 0:
            break
        P = C[ids]
        if len(reference) > 0:
            dref = min_distances_blocked(P, reference)
        else:
            dref = np.ones(len(P))
        if len(selected) > 0:
            dsel = min_distances_blocked(P, np.vstack(selected))
            score = np.minimum(dref, dsel)
        else:
            score = dref
        # Prefer genuinely unexplored points, but fall back to the farthest.
        valid = np.where(score >= V18_MAXIMIN_MIN_ARCHIVE_DISTANCE)[0]
        jloc = int(valid[np.argmax(score[valid])]) if len(valid) else int(np.argmax(score))
        j = int(ids[jloc])
        selected.append(C[j].copy())
        available[j] = False
    return np.vstack(selected) if len(selected) else np.empty((0, dimension))

def v18_quarantine_and_regenerate(basin_manager, saturated_ids, incumbent_z, Xz_true, rng, current_fe):
    """
    Preserve a saturated incumbent basin in the archive but quarantine it from
    local refinement. Regenerate saturated non-incumbent basins and, when the
    incumbent itself is saturated, regenerate most alternative basins.
    """
    basins = basin_manager.basins
    if len(basins) == 0 or len(saturated_ids) == 0:
        return {'quarantined': 0, 'regenerated': 0, 'incumbent_saturated': False}
    inc_idx = incumbent_basin_index(basins, incumbent_z)
    sat = set(int(i) for i in saturated_ids if 0 <= int(i) < len(basins))
    incumbent_saturated = inc_idx in sat
    quarantined = 0
    # Incumbent basin is never destroyed; only local exploitation is paused.
    if incumbent_saturated and inc_idx >= 0:
        b = basins[inc_idx]
        b.quarantine_until_fe = int(current_fe + V18_QUARANTINE_FES)
        b.saturation_count = int(getattr(b, 'saturation_count', 0)) + 1
        quarantined += 1
    replace_ids = [i for i in sat if i != inc_idx]
    if incumbent_saturated:
        alternatives = [i for i in range(len(basins)) if i != inc_idx]
        n_alt = max(1, int(np.ceil(V18_REGENERATE_FRACTION * len(alternatives)))) if alternatives else 0
        # Least productive alternatives are replaced first.
        alternatives = sorted(alternatives, key=lambda i: (v18_basin_success_rate(basins[i]), - (current_fe - basins[i].last_improvement_fe)))
        for i in alternatives[:n_alt]:
            if i not in replace_ids:
                replace_ids.append(i)
    if len(replace_ids) == 0:
        return {'quarantined': quarantined, 'regenerated': 0, 'incumbent_saturated': incumbent_saturated}
    surviving_centers = [b.center for i, b in enumerate(basins) if i not in replace_ids]
    seeds = v18_maximin_sobol_seeds(Xz_true, surviving_centers, len(replace_ids), incumbent_z.size, rng)
    regenerated = 0
    for bi, z0 in zip(replace_ids, seeds):
        b = basins[int(bi)]
        b.center = z0.copy()
        b.best_z = z0.copy()
        b.best_y = np.inf
        b.radius = TR_RADIUS_INITIAL
        b.failure_count = 0
        b.last_improvement_fe = int(current_fe)
        b.last_significant_improvement_fe = int(current_fe)
        b.quarantine_until_fe = 0
        b.saturation_count = int(getattr(b, 'saturation_count', 0)) + 1
        b.restarted_this_cycle = True
        b.success_history = []
        b.true_bmr_evals = 0
        b.true_bmr_successes = 0
        b.credit_history = []
        b.population = basin_manager._init_local_population(z0, b.radius)
        regenerated += 1
    return {'quarantined': quarantined, 'regenerated': regenerated, 'incumbent_saturated': incumbent_saturated}

def recent_local_success_active(current_fe, last_local_success_fe):
    return int(current_fe) - int(last_local_success_fe) <= LOCAL_SUCCESS_MEMORY_FES

class SourceCreditTracker:
    """Rolling TRUE-evaluation productivity by source.

    Productivity = smoothed success probability multiplied by a bounded
    verified relative-gain factor.  Surrogate predictions never create credit.
    """

    def __init__(self, window=PRODUCTIVITY_WINDOW):
        self.window = int(window)
        self.events = []

    def add(self, source, improved, relative_gain=0.0):
        self.events.append((str(source), 1 if improved else 0, max(0.0, float(relative_gain))))
        if len(self.events) > self.window:
            self.events = self.events[-self.window:]

    def _ev(self, sources):
        s = set((str(x) for x in sources))
        return [e for e in self.events if e[0] in s]

    def productivity(self, sources):
        ev = self._ev(sources)
        succ = sum((e[1] for e in ev))
        ps = (succ + PRODUCTIVITY_PRIOR_SUCCESS) / (len(ev) + PRODUCTIVITY_PRIOR_TOTAL)
        gains = [e[2] for e in ev if e[1] > 0]
        mg = float(np.mean(gains)) if gains else 0.0
        return float(ps * (1.0 + PRODUCTIVITY_GAIN_WEIGHT * np.tanh(20.0 * mg)))

    def source_productivity(self, source):
        return self.productivity((source,))

    def local_rate(self):
        return self.productivity(LOCAL_SOURCES)

    def global_rate(self):
        return self.productivity(GLOBAL_SOURCES)

    def all_source_productivities(self):
        sources = ('BWR', 'BMR', 'INCUMBENT', 'POLL', 'SURR_LOCAL', 'QUAD_TR', 'ANISO', 'PRECISION', 'NOVELTY', 'COVERAGE')
        return {s: self.source_productivity(s) for s in sources}

class TrueLocalQuadratic:
    """
    Local quadratic surrogate fitted only to TRUE evaluated samples.

    For high dimensions or limited data, a diagonal quadratic is used:
        1, s_j, s_j^2

    When enough samples exist, cross terms are added:
        s_i s_j

    The model is ridge-regularized and fitted in normalized decision
    coordinates centered on the current TRUE incumbent.
    """

    def __init__(self, ridge=QUAD_RIDGE):
        self.ridge = float(ridge)
        self.center = None
        self.scale = None
        self.beta = None
        self.full_quadratic = False
        self.y_mean = 0.0
        self.y_std = 1.0
        self.fitted = False

    def _features(self, S):
        S = np.atleast_2d(np.asarray(S, dtype=float))
        n, d = S.shape
        cols = [np.ones(n)]
        for j in range(d):
            cols.append(S[:, j])
        for j in range(d):
            cols.append(S[:, j] ** 2)
        if self.full_quadratic:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(S[:, i] * S[:, j])
        return np.column_stack(cols)

    def fit(self, X, y, center):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        center = np.asarray(center, dtype=float)
        if len(X) < QUAD_LOCAL_MIN_POINTS:
            return False
        self.center = center.copy()
        spread = np.std(X - center, axis=0)
        self.scale = np.maximum(spread, 0.001)
        S = (X - center) / self.scale
        d = X.shape[1]
        diagonal_terms = 1 + 2 * d
        full_terms = 1 + 2 * d + d * (d - 1) // 2
        self.full_quadratic = len(X) >= max(full_terms + 5, 2 * full_terms)
        Phi = self._features(S)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y))
        if self.y_std < 1e-12:
            self.y_std = 1.0
        yn = (y - self.y_mean) / self.y_std
        A = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        b = Phi.T @ yn
        try:
            self.beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(A, b, rcond=None)[0]
        self.fitted = True
        return True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError('TrueLocalQuadratic is not fitted.')
        X = np.atleast_2d(np.asarray(X, dtype=float))
        S = (X - self.center) / self.scale
        Phi = self._features(S)
        yn = Phi @ self.beta
        return self.y_mean + self.y_std * yn

def build_true_local_quadratic(Xz_true, y_true, incumbent_z):
    """
    Fit the local quadratic model using nearest TRUE archive samples.
    """
    if len(y_true) < QUAD_LOCAL_MIN_POINTS:
        return None
    d = np.linalg.norm(Xz_true - incumbent_z, axis=1)
    ids = np.argsort(d)[:min(QUAD_LOCAL_MAX_POINTS, len(Xz_true))]
    model = TrueLocalQuadratic(ridge=QUAD_RIDGE)
    ok = model.fit(Xz_true[ids], y_true[ids], center=incumbent_z)
    if not ok:
        return None
    return model

def quadratic_tr_candidates(incumbent_z, quad_model, Xz_true, rng, stall_age):
    """
    Generate generic local candidates from a TRUE-data quadratic model.

    No CEC function is called here.
    """
    if quad_model is None:
        return np.empty((0, incumbent_z.size))
    radius = QUAD_TR_RADIUS_INITIAL / np.sqrt(1.0 + max(0, stall_age) / 50.0)
    radius = float(np.clip(radius, QUAD_TR_RADIUS_MIN, QUAD_TR_RADIUS_MAX))
    lo = clip01(incumbent_z - radius)
    hi = clip01(incumbent_z + radius)
    bounds = list(zip(lo.tolist(), hi.tolist()))
    starts = [incumbent_z.copy()]
    for j in range(incumbent_z.size):
        zp = incumbent_z.copy()
        zm = incumbent_z.copy()
        zp[j] = min(hi[j], zp[j] + 0.5 * radius)
        zm[j] = max(lo[j], zm[j] - 0.5 * radius)
        starts.append(zp)
        starts.append(zm)
    n_random = max(4, incumbent_z.size // 2)
    for _ in range(n_random):
        starts.append(lo + rng.random(incumbent_z.size) * (hi - lo))
    candidates = []

    def qobj(z):
        return float(quad_model.predict(np.asarray(z, dtype=float)[None, :])[0])
    for x0 in starts:
        try:
            res = minimize(qobj, x0=np.asarray(x0, dtype=float), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-13})
            candidates.append(clip01(np.asarray(res.x, dtype=float)))
        except Exception:
            pass
    for _ in range(max(10, 2 * incumbent_z.size)):
        candidates.append(clip01(incumbent_z + rng.normal(0.0, radius / 3.0, size=incumbent_z.size)))
    if len(candidates) == 0:
        return np.empty((0, incumbent_z.size))
    C = unique_rows(np.vstack(candidates))
    C = remove_true_duplicates(C, Xz_true)
    if len(C) == 0:
        return C
    pred = quad_model.predict(C)
    keep = min(max(20, 4 * incumbent_z.size), len(C))
    ids = np.argsort(pred)[:keep]
    return C[ids]

def micro_local_trigger(reliability, stall_age, local_credit, global_credit):
    """
    Generic precision-refinement trigger.

    The state is favored when:
      * the model is reasonably reliable,
      * a moderate stall exists,
      * local sources have been at least as productive as global sources.
    """
    return reliability >= 0.65 and stall_age >= MICRO_LOCAL_TRIGGER_FES and (stall_age < MICRO_LOCAL_MAX_FES) and (local_credit >= global_credit or global_credit <= 0.02)

class AnisotropicLocalModel:
    """
    Learns an ellipsoidal trust region from TRUE archive samples near
    the incumbent. It is generic: no function ID, optimum, or hidden
    benchmark information is used.
    """

    def __init__(self):
        self.center = None
        self.axis_radius = None
        self.quad = None
        self.ready = False

    def fit(self, Xz_true, y_true, incumbent_z, stall_age):
        Xz_true = np.asarray(Xz_true, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        incumbent_z = np.asarray(incumbent_z, dtype=float)
        if len(y_true) < ANISO_MIN_POINTS:
            return False
        dist = np.linalg.norm(Xz_true - incumbent_z, axis=1)
        ids = np.argsort(dist)[:min(ANISO_MAX_POINTS, len(Xz_true))]
        Xloc = Xz_true[ids]
        yloc = y_true[ids]
        quad = TrueLocalQuadratic(ridge=QUAD_RIDGE)
        if not quad.fit(Xloc, yloc, center=incumbent_z):
            return False
        self.center = incumbent_z.copy()
        self.quad = quad
        delta = Xloc - incumbent_z
        spread = np.median(np.abs(delta), axis=0)
        spread = np.maximum(spread, 0.0001)
        spread_scale = spread / max(np.median(spread), 1e-12)
        base = float(quad.predict(incumbent_z[None, :])[0])
        curvature = np.zeros(incumbent_z.size)
        h = 0.001
        for j in range(incumbent_z.size):
            zp = incumbent_z.copy()
            zm = incumbent_z.copy()
            zp[j] = min(1.0, zp[j] + h)
            zm[j] = max(0.0, zm[j] - h)
            fp = float(quad.predict(zp[None, :])[0])
            fm = float(quad.predict(zm[None, :])[0])
            hp = max(zp[j] - incumbent_z[j], 1e-08)
            hm = max(incumbent_z[j] - zm[j], 1e-08)
            curvature[j] = abs(2.0 * (hm * fp - (hp + hm) * base + hp * fm) / max(hp * hm * (hp + hm), 1e-12))
        curv_scale = 1.0 / np.sqrt(curvature + 1e-12)
        curv_scale /= max(np.median(curv_scale), 1e-12)
        scale = ANISO_CURVATURE_WEIGHT * curv_scale + ANISO_SPREAD_WEIGHT * spread_scale
        scale = np.clip(scale, 1.0 / ANISO_AXIS_RATIO_MAX, ANISO_AXIS_RATIO_MAX)
        base_radius = ANISO_RADIUS_INITIAL / np.sqrt(1.0 + max(0.0, float(stall_age)) / 40.0)
        self.axis_radius = np.clip(base_radius * scale, ANISO_RADIUS_MIN, ANISO_RADIUS_MAX)
        self.ready = True
        return True

    def bounds(self):
        lo = clip01(self.center - self.axis_radius)
        hi = clip01(self.center + self.axis_radius)
        return (lo, hi)

def build_anisotropic_local_model(Xz_true, y_true, incumbent_z, stall_age):
    model = AnisotropicLocalModel()
    if model.fit(Xz_true, y_true, incumbent_z, stall_age):
        return model
    return None

def anisotropic_candidates(incumbent_z, aniso_model, local_model, Xz_true, rng):
    if aniso_model is None:
        return np.empty((0, incumbent_z.size))
    lo, hi = aniso_model.bounds()
    d = incumbent_z.size
    points = []
    for j in range(d):
        for sign in (-1.0, 1.0):
            z = incumbent_z.copy()
            z[j] += sign * aniso_model.axis_radius[j]
            points.append(np.minimum(np.maximum(z, lo), hi))
    for _ in range(ANISO_RANDOM_CANDIDATES):
        u = rng.normal(0.0, 1.0, size=d)
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            continue
        u /= nu
        radial = rng.random() ** (1.0 / d)
        z = incumbent_z + radial * u * aniso_model.axis_radius
        points.append(np.minimum(np.maximum(z, lo), hi))
    bounds = list(zip(lo.tolist(), hi.tolist()))
    starts = [incumbent_z.copy()]
    for _ in range(ANISO_LBFGSB_RESTARTS - 1):
        starts.append(lo + rng.random(d) * (hi - lo))

    def surr_obj(z):
        return float(local_model.predict(np.asarray(z, dtype=float)[None, :])[0])
    for x0 in starts:
        try:
            res = minimize(surr_obj, x0=np.asarray(x0, dtype=float), method='L-BFGS-B', bounds=bounds, options={'maxiter': ANISO_LBFGSB_MAXITER, 'ftol': 1e-13})
            points.append(clip01(np.asarray(res.x, dtype=float)))
        except Exception:
            pass
    if not points:
        return np.empty((0, d))
    C = unique_rows(np.vstack(points))
    C = remove_true_duplicates(C, Xz_true)
    if len(C) == 0:
        return C
    pred = local_model.predict(C)
    keep = min(max(30, 5 * d), len(C))
    return C[np.argsort(pred)[:keep]]

def anisotropic_mode_trigger(reliability, stall_age, local_productivity, global_productivity, recent_aniso_success):
    if recent_aniso_success:
        return True
    return reliability >= 0.55 and stall_age >= ANISO_TRIGGER_STALL_FES and (local_productivity >= global_productivity or stall_age >= ANISO_DEEP_TRIGGER_FES)

def precision_mode_trigger(reliability, stall_age, local_productivity, global_productivity, recent_precision_success):
    """
    Generic terminal-refinement trigger.

    No function ID and no known optimum are used.
    """
    if recent_precision_success:
        return True
    return reliability >= PRECISION_RELIABILITY_MIN and stall_age >= PRECISION_TRIGGER_STALL_FES and (local_productivity >= global_productivity)

def precision_candidates(incumbent_z, aniso_model, local_model, Xz_true, rng, stall_age, explicit_base_radius=None):
    """
    Very fine anisotropic candidate generator around the TRUE incumbent.

    The base radius contracts continuously with TRUE stagnation age,
    then is modulated by learned anisotropic axis scales.
    """
    incumbent_z = np.asarray(incumbent_z, dtype=float)
    d = incumbent_z.size
    if explicit_base_radius is None:
        base_radius = PRECISION_RADIUS_INITIAL / np.sqrt(1.0 + max(0.0, float(stall_age)) / 25.0)
    else:
        base_radius = float(explicit_base_radius)
    base_radius = float(np.clip(base_radius, PRECISION_MEMORY_MIN_RADIUS, PRECISION_MEMORY_MAX_RADIUS))
    if aniso_model is not None and aniso_model.ready:
        axis_scale = aniso_model.axis_radius / max(np.median(aniso_model.axis_radius), 1e-15)
    else:
        axis_scale = np.ones(d)
    axis_scale = np.clip(axis_scale, 1.0 / ANISO_AXIS_RATIO_MAX, ANISO_AXIS_RATIO_MAX)
    points = []
    for level in PRECISION_AXIS_LEVELS:
        step = np.clip(base_radius * float(level) * axis_scale, PRECISION_RADIUS_MIN, PRECISION_RADIUS_MAX)
        for j in range(d):
            for sign in (-1.0, 1.0):
                z = incumbent_z.copy()
                z[j] += sign * step[j]
                points.append(clip01(z))
    fine_axis = np.clip(base_radius * axis_scale, PRECISION_RADIUS_MIN, PRECISION_RADIUS_MAX)
    for _ in range(PRECISION_RANDOM_CANDIDATES):
        u = rng.normal(0.0, 1.0, size=d)
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            continue
        u /= norm
        radial = rng.random() ** (1.0 / d)
        z = incumbent_z + radial * u * fine_axis
        points.append(clip01(z))
    if len(points) == 0:
        return np.empty((0, d))
    C = unique_rows(np.vstack(points))
    C = remove_true_duplicates(C, Xz_true)
    if len(C) == 0:
        return C
    pred = local_model.predict(C)
    keep = min(max(30, 4 * d), len(C))
    ids = np.argsort(pred)[:keep]
    return C[ids]

def magnitude_aware_success(old_best, new_value, rel_tol=MAG_SUCCESS_REL_GAIN, abs_tol=MAG_SUCCESS_ABS_GAIN):
    """
    Scale-aware TRUE progress test.

    V15 deliberately uses a single threshold
        gain >= max(abs_tol, rel_tol * max(|old_best|, 1))
    instead of V14's OR rule.  Consequently a numerically tiny improvement
    can still update the archive best, but it does not earn productivity
    credit or reset the meaningful-progress clock.
    """
    old_best = float(old_best)
    new_value = float(new_value)
    gain = max(0.0, old_best - new_value)
    scale = max(abs(old_best), 1.0)
    rel_gain = gain / scale
    threshold = max(float(abs_tol), float(rel_tol) * scale)
    success = new_value < old_best and gain >= threshold
    return (bool(success), float(gain), float(rel_gain))

def slow_progress(best_history, window=SLOW_PROGRESS_WINDOW_FES, rel_tol=SLOW_PROGRESS_REL_TOL):
    """Detect genuine slow progress even when tiny improvements continue."""
    h = np.asarray(best_history, dtype=float)
    if h.size < window + 1:
        return (False, np.inf)
    old = float(h[-window - 1])
    new = float(h[-1])
    gain = max(0.0, old - new)
    rel_gain = gain / max(abs(old), 1.0)
    return (bool(rel_gain < rel_tol), float(rel_gain))

class FullRotatedQuadraticTR:
    """
    Full ridge-regularized local quadratic fitted only to TRUE samples.

    q(s) = c + g^T s + 1/2 s^T H s

    The symmetric Hessian H is eigendecomposed to obtain rotated search
    directions. No function ID, reference optimum, shift vector, or CEC
    transformation matrix is used.
    """

    def __init__(self, ridge=ROT_QUAD_RIDGE):
        self.ridge = float(ridge)
        self.center = None
        self.scale = None
        self.beta = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.gradient_s = None
        self.hessian_s = None
        self.eigvals = None
        self.eigvecs_s = None
        self.axis_radius_s = None
        self.base_radius = ROT_TR_RADIUS_INITIAL
        self.ready = False

    @staticmethod
    def _features(S):
        S = np.atleast_2d(np.asarray(S, dtype=float))
        n, d = S.shape
        cols = [np.ones(n)]
        cols.extend([S[:, j] for j in range(d)])
        cols.extend([S[:, j] ** 2 for j in range(d)])
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(S[:, i] * S[:, j])
        return np.column_stack(cols)

    def _unpack(self, d):
        b = np.asarray(self.beta, dtype=float)
        k = 1
        g = b[k:k + d].copy()
        k += d
        diag_coef = b[k:k + d].copy()
        k += d
        H = np.zeros((d, d), dtype=float)
        for j in range(d):
            H[j, j] = 2.0 * diag_coef[j]
        for i in range(d):
            for j in range(i + 1, d):
                H[i, j] = b[k]
                H[j, i] = b[k]
                k += 1
        self.gradient_s = g
        self.hessian_s = 0.5 * (H + H.T)

    def fit(self, Xz_true, y_true, incumbent_z, stall_age=0):
        Xz_true = np.asarray(Xz_true, dtype=float)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        incumbent_z = np.asarray(incumbent_z, dtype=float)
        if len(y_true) < ROT_QUAD_MIN_POINTS:
            return False
        dist = np.linalg.norm(Xz_true - incumbent_z, axis=1)
        ids = np.argsort(dist)[:min(ROT_QUAD_MAX_POINTS, len(y_true))]
        X = Xz_true[ids]
        y = y_true[ids]
        self.center = incumbent_z.copy()
        spread = np.std(X - incumbent_z, axis=0)
        self.scale = np.maximum(spread, 0.002)
        S = (X - incumbent_z) / self.scale
        Phi = self._features(S)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y))
        if self.y_std < 1e-12:
            self.y_std = 1.0
        yn = (y - self.y_mean) / self.y_std
        r = np.linalg.norm(S, axis=1)
        med = max(float(np.median(r)), 1e-06)
        w = np.exp(-(r / (1.5 * med)) ** 2)
        sw = np.sqrt(np.maximum(w, 1e-08))
        Pw = Phi * sw[:, None]
        yw = yn * sw
        A = Pw.T @ Pw + self.ridge * np.eye(Phi.shape[1])
        b = Pw.T @ yw
        try:
            self.beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(A, b, rcond=None)[0]
        d = incumbent_z.size
        self._unpack(d)
        try:
            eigvals, eigvecs = np.linalg.eigh(self.hessian_s)
        except np.linalg.LinAlgError:
            return False
        curv = np.maximum(np.abs(eigvals), 1e-08)
        invsqrt = 1.0 / np.sqrt(curv)
        invsqrt /= max(float(np.median(invsqrt)), 1e-12)
        invsqrt = np.clip(invsqrt, 1.0 / ROT_TR_AXIS_RATIO_MAX, ROT_TR_AXIS_RATIO_MAX)
        self.base_radius = float(np.clip(ROT_TR_RADIUS_INITIAL / np.sqrt(1.0 + max(0.0, stall_age) / 80.0), ROT_TR_RADIUS_MIN, ROT_TR_RADIUS_MAX))
        self.axis_radius_s = np.clip(self.base_radius * invsqrt, ROT_TR_RADIUS_MIN, ROT_TR_RADIUS_MAX)
        self.eigvals = eigvals
        self.eigvecs_s = eigvecs
        self.ready = True
        return True

    def predict(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        S = (Z - self.center) / self.scale
        yn = self._features(S) @ self.beta
        return self.y_mean + self.y_std * yn

    def rotated_step_to_z(self, t):
        s_step = self.eigvecs_s @ np.asarray(t, dtype=float)
        return clip01(self.center + self.scale * s_step)

    def candidate_bounds_t(self, base_radius=None):
        if base_radius is None:
            axes = self.axis_radius_s.copy()
        else:
            ratio = self.axis_radius_s / max(self.base_radius, 1e-15)
            axes = np.clip(float(base_radius) * ratio, ROT_TR_RADIUS_MIN, ROT_TR_RADIUS_MAX)
        return (-axes, axes)

class TrueNewtonTRMemory:
    """Persistent normalized-space trust radius for V18 Newton escalation."""

    def __init__(self):
        self.radius = float(NEWTON_TR_RADIUS_INITIAL)

    def on_good(self):
        self.radius = float(np.clip(self.radius * NEWTON_TR_EXPAND, NEWTON_TR_RADIUS_MIN, NEWTON_TR_RADIUS_MAX))

    def on_bad(self):
        self.radius = float(np.clip(self.radius * NEWTON_TR_SHRINK, NEWTON_TR_RADIUS_MIN, NEWTON_TR_RADIUS_MAX))

@njit(cache=True)
def _safe_symmetric_step(center, direction, requested_h):
    """Largest symmetric +/- step along direction that stays inside [0,1]^D."""
    # Numba-safe: inputs are already float64 arrays.  Avoid dtype=float,
    # which can be interpreted as a Numba Function object in some releases.
    d = direction
    nd = np.linalg.norm(d)
    if nd < 1e-14:
        return 0.0
    d = d / nd
    hmax = np.inf
    for j, dj in enumerate(d):
        if abs(dj) < 1e-15:
            continue
        if dj > 0.0:
            hp = (1.0 - center[j]) / dj
            hm = center[j] / dj
        else:
            hp = center[j] / -dj
            hm = (1.0 - center[j]) / -dj
        hmax = min(hmax, hp, hm)
    if not np.isfinite(hmax):
        return 0.0
    return float(max(0.0, min(float(requested_h), 0.9 * hmax)))

def _true_eval_z(problem, archive, best_history, z, max_true_fes):
    """Evaluate one normalized point with strict TRUE-FE accounting."""
    if problem.true_fe >= max_true_fes:
        return None
    z = clip01(np.asarray(z, dtype=float))
    x = denormalize_x(z, problem.lb, problem.ub)
    y = float(problem.evaluate(x))
    archive.add(x, y)
    best_history.append(archive.best_y())
    return y

def _fallback_newton_directions(Xz, y, center):
    """Function-agnostic PCA/coordinate basis when rotated quadratic geometry is unavailable."""
    Xz = np.asarray(Xz, dtype=float); y = np.asarray(y, dtype=float); center = np.asarray(center, dtype=float)
    d = center.size
    if len(Xz) >= 3:
        dist = np.linalg.norm(Xz - center, axis=1)
        ids = np.argsort(dist)[:min(len(Xz), V17_NEWTON_PCA_POINTS)]
        S = Xz[ids] - center
        if len(S) >= 3 and np.any(np.std(S, axis=0) > 1e-12):
            w = np.exp(-dist[ids] / max(float(np.median(dist[ids][dist[ids] > 0])) if np.any(dist[ids] > 0) else 1.0, 1e-12))
            Sw = S * np.sqrt(np.maximum(w, 1e-12))[:, None]
            C = Sw.T @ Sw / max(float(np.sum(w)), 1e-12)
            try:
                vals, vecs = np.linalg.eigh(C)
                order = np.argsort(vals)[::-1]
                Q = vecs[:, order]
                dirs = []
                for j in range(d):
                    v = Q[:, j]
                    nv = np.linalg.norm(v)
                    if nv > 1e-14: dirs.append(v / nv)
                if len(dirs) >= d: return dirs[:d]
            except np.linalg.LinAlgError:
                pass
    dirs=[]
    for j in range(d):
        e=np.zeros(d); e[j]=1.0; dirs.append(e)
    return dirs

def run_true_newton_tr_episode(problem, archive, best_history, rng, max_true_fes, memory, max_iters=None):
    """
    V18 robust counted TRUE-evaluation Newton trust-region episode.

    1) Fit the archive-based full rotated quadratic only to learn a local
       eigenbasis Q around the TRUE incumbent.
    2) Along each learned direction q_i, evaluate TRUE f(x +/- h q_i).
       These 2D samples estimate directional gradient and curvature.
    3) Build a regularized diagonal Hessian in the rotated basis and take a
       bounded Newton step. In original coordinates Q diag(kappa) Q^T is a
       full rotated Hessian.
    4) Evaluate the Newton proposal on the TRUE benchmark and update the TR
       using actual/predicted reduction rho.

    Every objective evaluation goes through problem.evaluate and therefore
    consumes the official TRUE-FE budget. Known optimum/reference is unused.
    """
    if max_iters is None:
        max_iters = NEWTON_TR_MAX_ITERS
    start_fe = int(problem.true_fe)
    start_best = float(archive.best_y())
    rhos = []
    accepted = 0
    for _ in range(int(max_iters)):
        if problem.true_fe >= max_true_fes:
            break
        X = archive.X_array
        y = archive.y_array
        Xz = normalize_x(X, problem.lb, problem.ub)
        ib = int(np.argmin(y))
        center = Xz[ib].copy()
        f0 = float(y[ib])
        orient = FullRotatedQuadraticTR()
        d = center.size
        directions = []
        if orient.fit(Xz, y, center, stall_age=0):
            Qs = np.asarray(orient.eigvecs_s, dtype=float)
            scale = np.asarray(orient.scale, dtype=float)
            for j in range(d):
                vz = scale * Qs[:, j]
                nv = np.linalg.norm(vz)
                if nv > 1e-14:
                    directions.append(vz / nv)
        if len(directions) < d:
            directions = _fallback_newton_directions(Xz, y, center)
        directions = directions[:d]
        g = np.zeros(d, dtype=float)
        kappa = np.zeros(d, dtype=float)
        valid = np.zeros(d, dtype=bool)
        h_used = np.zeros(d, dtype=float)
        for j, v in enumerate(directions):
            if problem.true_fe + 2 > max_true_fes:
                break
            h = _safe_symmetric_step(center, v, max(memory.radius, V17_NEWTON_FD_RADIUS_MIN))
            if h < NEWTON_TR_RADIUS_MIN:
                continue
            zp = center + h * v
            zm = center - h * v
            fp = _true_eval_z(problem, archive, best_history, zp, max_true_fes)
            fm = _true_eval_z(problem, archive, best_history, zm, max_true_fes)
            if fp is None or fm is None:
                break
            g[j] = (fp - fm) / (2.0 * h)
            kappa[j] = (fp - 2.0 * f0 + fm) / (h * h)
            h_used[j] = h
            valid[j] = np.isfinite(g[j]) and np.isfinite(kappa[j])
        if np.count_nonzero(valid) < V17_NEWTON_MIN_VALID_DIRS:
            memory.on_bad()
            break
        valid_curv = np.abs(kappa[valid])
        curv_scale = max(float(np.median(valid_curv)) if valid_curv.size else 1.0, NEWTON_TR_CURV_FLOOR)
        floor = max(NEWTON_TR_CURV_FLOOR, 1e-06 * curv_scale)
        # Trust-region modified Newton: negative/near-zero curvature receives a positive floor.
        # This avoids treating strong negative curvature as reliable positive curvature.
        kreg = np.where(valid, np.maximum(kappa, floor), np.inf)
        a = np.zeros(d, dtype=float)
        a[valid] = -g[valid] / kreg[valid]
        step = np.zeros(d, dtype=float)
        for j, v in enumerate(directions):
            step += a[j] * v
        sn = np.linalg.norm(step)
        max_norm = NEWTON_TR_MAX_STEP_NORM_FACTOR * memory.radius
        if sn > max_norm and sn > 1e-15:
            scale_step = max_norm / sn
            a *= scale_step
            step *= scale_step
        ztrial = clip01(center + step)
        actual_step = ztrial - center
        if np.linalg.norm(actual_step) < NEWTON_TR_RADIUS_MIN:
            memory.on_bad()
            break
        pred_change = 0.0
        for j in range(d):
            if valid[j]:
                pred_change += g[j] * a[j] + 0.5 * kreg[j] * a[j] * a[j]
        pred_red = max(0.0, -float(pred_change))
        if pred_red <= NEWTON_TR_MIN_PRED_REDUCTION:
            grad_z = np.zeros(d, dtype=float)
            for j, v in enumerate(directions):
                if valid[j]:
                    grad_z += g[j] * v
            ng = np.linalg.norm(grad_z)
            if ng < 1e-14:
                memory.on_bad()
                break
            ztrial = clip01(center - memory.radius * grad_z / ng)
            pred_red = max(NEWTON_TR_MIN_PRED_REDUCTION, memory.radius * ng)
        if problem.true_fe >= max_true_fes:
            break
        ftrial = _true_eval_z(problem, archive, best_history, ztrial, max_true_fes)
        if ftrial is None:
            break
        actual_red = float(f0 - ftrial)
        rho = actual_red / max(pred_red, NEWTON_TR_MIN_PRED_REDUCTION)
        rhos.append(float(rho))
        if actual_red > 0.0 and rho >= NEWTON_TR_ETA_ACCEPT:
            accepted += 1
            if rho >= NEWTON_TR_ETA_EXPAND:
                memory.on_good()
        else:
            memory.on_bad()
        if memory.radius <= NEWTON_TR_RADIUS_MIN * 1.01:
            break
    end_best = float(archive.best_y())
    return {'used': int(problem.true_fe - start_fe), 'best_before': start_best, 'best_after': end_best, 'improvement': float(start_best - end_best), 'accepted': int(accepted), 'rho_last': float(rhos[-1]) if rhos else np.nan, 'rho_mean': float(np.mean(rhos)) if rhos else np.nan, 'radius': float(memory.radius)}

def build_anisotropic_local_model(Xz_true, y_true, incumbent_z, stall_age):
    """V14 override: full rotated quadratic geometry from TRUE samples."""
    model = FullRotatedQuadraticTR()
    if model.fit(Xz_true, y_true, incumbent_z, stall_age=stall_age):
        return model
    return None

def _rotated_tr_candidates(incumbent_z, model, Xz_true, rng, base_radius=None, n_random=None):
    if model is None or not getattr(model, 'ready', False):
        return np.empty((0, incumbent_z.size))
    d = incumbent_z.size
    lo_t, hi_t = model.candidate_bounds_t(base_radius)
    points = [incumbent_z.copy()]
    for j in range(d):
        for sign in (-1.0, 1.0):
            t = np.zeros(d)
            t[j] = sign * hi_t[j]
            points.append(model.rotated_step_to_z(t))
    if n_random is None:
        n_random = ROT_TR_RANDOM_CANDIDATES
    for _ in range(int(n_random)):
        u = rng.normal(size=d)
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            continue
        u /= nu
        radial = rng.random() ** (1.0 / d)
        t = radial * u * hi_t
        points.append(model.rotated_step_to_z(t))
    bounds = list(zip(lo_t.tolist(), hi_t.tolist()))
    starts = [np.zeros(d)]
    for _ in range(ROT_TR_RESTARTS - 1):
        starts.append(lo_t + rng.random(d) * (hi_t - lo_t))

    def qobj(t):
        z = model.rotated_step_to_z(t)
        return float(model.predict(z[None, :])[0])
    for t0 in starts:
        try:
            res = minimize(qobj, x0=np.asarray(t0, dtype=float), method='L-BFGS-B', bounds=bounds, options={'maxiter': ROT_TR_MAXITER, 'ftol': 1e-13})
            points.append(model.rotated_step_to_z(res.x))
        except Exception:
            pass
    C = unique_rows(np.vstack(points))
    C = remove_true_duplicates(C, Xz_true)
    if len(C) == 0:
        return C
    pred = model.predict(C)
    keep = min(max(40, 6 * d), len(C))
    return C[np.argsort(pred)[:keep]]

def anisotropic_candidates(incumbent_z, aniso_model, local_model, Xz_true, rng):
    """V14 override: candidates follow learned rotated Hessian axes."""
    return _rotated_tr_candidates(incumbent_z, aniso_model, Xz_true, rng, base_radius=None, n_random=ROT_TR_RANDOM_CANDIDATES)

def precision_candidates(incumbent_z, aniso_model, local_model, Xz_true, rng, stall_age, explicit_base_radius=None):
    """V14 override: precision search uses the same rotated geometry."""
    if aniso_model is None:
        return np.empty((0, incumbent_z.size))
    if explicit_base_radius is None:
        base_radius = PRECISION_RADIUS_INITIAL / np.sqrt(1.0 + max(0.0, float(stall_age)) / 60.0)
    else:
        base_radius = float(explicit_base_radius)
    base_radius = float(np.clip(base_radius, PRECISION_MEMORY_MIN_RADIUS, PRECISION_MEMORY_MAX_RADIUS))
    return _rotated_tr_candidates(incumbent_z, aniso_model, Xz_true, rng, base_radius=base_radius, n_random=PRECISION_RANDOM_CANDIDATES)

class LocalModeSuccessTracker:
    """
    Rolling TRUE-evaluation success history for ANISO and PRECISION.

    A success means the evaluated candidate improved the GLOBAL TRUE best
    immediately before that evaluation.
    """

    def __init__(self, window=LOCAL_MODE_SUCCESS_WINDOW):
        self.window = int(window)
        self.events = {'ANISO': [], 'PRECISION': []}

    def add(self, source, success):
        if source not in self.events:
            return
        arr = self.events[source]
        arr.append(1 if success else 0)
        if len(arr) > self.window:
            self.events[source] = arr[-self.window:]

    def ratio(self, source):
        arr = self.events.get(source, [])
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr))

    def count(self, source):
        return len(self.events.get(source, []))

class PrecisionRadiusMemory:
    """
    Adaptive precision scale controlled only by verified TRUE outcomes.
    """

    def __init__(self):
        self.radius = float(PRECISION_MEMORY_INITIAL_RADIUS)
        self.last_success_radius = float(PRECISION_MEMORY_INITIAL_RADIUS)

    def on_success(self):
        self.last_success_radius = float(self.radius)
        self.radius = float(np.clip(self.radius * PRECISION_SUCCESS_SHRINK, PRECISION_MEMORY_MIN_RADIUS, PRECISION_MEMORY_MAX_RADIUS))

    def on_failure_episode(self):
        self.radius = float(np.clip(self.radius * PRECISION_FAILURE_EXPAND, PRECISION_MEMORY_MIN_RADIUS, PRECISION_MEMORY_MAX_RADIUS))

    def retry_radius(self):
        return float(np.clip(self.last_success_radius * PRECISION_RETRY_SCALE, PRECISION_MEMORY_MIN_RADIUS, PRECISION_MEMORY_MAX_RADIUS))

def aniso_success_ratio_ok(tracker):
    n = tracker.count('ANISO')
    if n < 4:
        return True
    return tracker.ratio('ANISO') >= ANISO_MIN_SUCCESS_RATIO

def precision_success_ratio_ok(tracker):
    n = tracker.count('PRECISION')
    if n < 4:
        return True
    return tracker.ratio('PRECISION') >= PRECISION_MIN_SUCCESS_RATIO

def choose_state(reliability, recovery_remaining, deep_recovery_remaining, recent_improvement, stall_age, intensify_flag, micro_local_flag, local_productivity, global_productivity, anisotropic_flag, precision_flag, precision_retry_flag):
    """
    V13 controller.

    Priority:
        DEEP_RECOVERY
        RECOVERY
        PRECISION_RETRY
        PRECISION_LOCAL
        ANISO_LOCAL
        MICRO_LOCAL
        GLOBAL / LOCAL / BALANCED
    """
    if deep_recovery_remaining > 0:
        return 'DEEP_RECOVERY'
    if recovery_remaining > 0:
        return 'RECOVERY'
    if precision_retry_flag:
        return 'PRECISION_RETRY'
    if precision_flag:
        return 'PRECISION_LOCAL'
    if anisotropic_flag:
        return 'ANISO_LOCAL'
    if stall_age >= MICRO_LOCAL_TRIGGER_FES:
        if local_productivity >= global_productivity and local_productivity > 0.0:
            return 'MICRO_LOCAL'
        if stall_age >= FORCE_NOVELTY_AFTER_FES:
            return 'RECOVERY'
    if local_productivity > global_productivity and reliability >= 0.45:
        return 'LOCAL'
    if global_productivity > local_productivity:
        return 'GLOBAL'
    return 'BALANCED'

@dataclass
class CandidateRecord:
    z: np.ndarray
    pred: float
    nc: float
    source: str
    basin_idx: int
    restart_priority: bool = False

def evaluate_candidate_information(C, source, basin_idx, global_model, local_models, Xz_true):
    if len(C) == 0:
        return []
    C = np.asarray(C, dtype=float)
    if source in ('BMR', 'INCUMBENT', 'POLL', 'SURR_LOCAL', 'QUAD_TR', 'ANISO', 'PRECISION') and basin_idx >= 0 and (basin_idx in local_models):
        pred = local_models[basin_idx].predict(C)
    else:
        pred = global_model.predict(C)
    nc = nc_values(C, Xz_true, NC_NEIGHBORS)
    records = []
    for i in range(len(C)):
        records.append(CandidateRecord(z=C[i].copy(), pred=float(pred[i]), nc=float(nc[i]), source=source, basin_idx=int(basin_idx), restart_priority=False))
    return records

def dsds_admissible(records):
    if len(records) == 0:
        return []
    pred = np.asarray([r.pred for r in records])
    nc = np.asarray([r.nc for r in records])
    p = robust_scale(pred)
    n = robust_scale(nc)
    ids = pareto_front_2d(p, n)
    return [records[i] for i in ids]

def v19_global_confidence(Xz_true, y_true, incumbent_z):
    """Cheap basin-level confidence; independent of known CEC optimum.

    High values require good archive support around the incumbent AND evidence
    that spatially separated archive regions are not similarly competitive.
    This prevents local surrogate reliability from being mistaken for global
    basin certainty.
    """
    Xz_true = np.asarray(Xz_true, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    incumbent_z = np.asarray(incumbent_z, dtype=float)
    if len(y_true) < max(20, 2 * incumbent_z.size):
        return 0.25
    dinc = np.linalg.norm(Xz_true - incumbent_z, axis=1)
    local_support = float(np.mean(dinc <= V19_GLOBAL_CONFIDENCE_DIVERSITY_RADIUS))
    local_support = min(1.0, local_support / 0.20)
    order = np.argsort(y_true)
    top = order[:max(10, int(0.10 * len(order)))]
    if len(top) <= 1:
        separation = 0.0
    else:
        dt = np.linalg.norm(Xz_true[top] - incumbent_z, axis=1)
        separated = dt >= V19_GLOBAL_CONFIDENCE_DIVERSITY_RADIUS
        if np.any(separated):
            y0 = float(y_true[order[0]])
            ys = float(np.min(y_true[top][separated]))
            rel_gap = max(0.0, ys - y0) / max(abs(y0), 1.0)
            separation = float(np.tanh(10.0 * rel_gap))
        else:
            separation = 0.25
    return float(np.clip(0.45 * local_support + 0.55 * separation, 0.0, 1.0))


def adaptive_acquisition_weights(source_productivities, reliability,
                                 global_confidence, meaningful_stall_age):
    """Compute parameter-free, evidence-adaptive acquisition weights.

    No CEC function ID, known optimum value/vector, shift, or rotation is used.
    The four non-negative utilities are obtained only from quantities already
    produced by the search, then normalized so the weights sum exactly to 1.

    Returns
    -------
    dict with keys: productivity, prediction, novelty, diversity
    """
    eps = np.finfo(float).eps

    reliability = float(np.clip(reliability, 0.0, 1.0))
    global_confidence = float(np.clip(global_confidence, 0.0, 1.0))

    # Aggregate VERIFIED productivity evidence across currently known sources.
    # If no source has evidence yet, use neutral evidence rather than a tuned
    # acquisition coefficient.
    vals = [float(np.clip(v, 0.0, 1.0))
            for v in source_productivities.values()
            if np.isfinite(v)]
    productivity_evidence = float(np.mean(vals)) if vals else 0.5

    # Dimensionless stagnation pressure in [0,1).  V19_LONG_STALL_FES is an
    # existing search-state time scale, not an acquisition weight.
    stall = max(0.0, float(meaningful_stall_age))
    scale = max(1.0, float(V19_LONG_STALL_FES))
    stagnation_pressure = stall / (stall + scale)

    # Evidence utilities. Exploration becomes stronger when the model/global
    # basin evidence is weak or verified progress has stalled.
    u_productivity = eps + productivity_evidence
    u_prediction = eps + reliability
    u_novelty = eps + (1.0 - reliability) + stagnation_pressure
    u_diversity = eps + (1.0 - global_confidence) + stagnation_pressure

    U = np.asarray([u_productivity, u_prediction,
                    u_novelty, u_diversity], dtype=float)
    U = np.maximum(U, eps)
    W = U / np.sum(U)

    return {
        'productivity': float(W[0]),
        'prediction': float(W[1]),
        'novelty': float(W[2]),
        'diversity': float(W[3]),
    }


def adaptive_restart_bonus(reliability, global_confidence, meaningful_stall_age):
    """Evidence-adaptive bonus for stale-basin restart candidates.

    Replaces the former fixed +0.50 restart-priority term.  The bonus becomes
    strong only when the surrogate is unreliable, global basin confidence is
    weak, and/or meaningful stagnation is high.  It is bounded in [0, 0.50].

        B_R = (0.50 / 3) * [(1 - R) + (1 - G) + S]
        S   = A_stall / (A_stall + V19_LONG_STALL_FES)

    No known optimum, function ID, shift, or rotation information is used.
    """
    reliability = float(np.clip(reliability, 0.0, 1.0))
    global_confidence = float(np.clip(global_confidence, 0.0, 1.0))

    stall = max(0.0, float(meaningful_stall_age))
    scale = max(1.0, float(V19_LONG_STALL_FES))
    stagnation_pressure = stall / (stall + scale)

    bonus = (0.50 / 3.0) * (
        (1.0 - reliability)
        + (1.0 - global_confidence)
        + stagnation_pressure
    )
    return float(np.clip(bonus, 0.0, 0.50))


def v19_source_score(source, source_productivities, reliability, global_confidence,
                     state, meaningful_stall_age, adaptive_weights=None):
    """Evidence score used only to allocate TRUE evaluations; never uses f*.

    The source-productivity contribution now uses the online adaptive
    productivity weight instead of a fixed coefficient.
    """
    if adaptive_weights is None:
        adaptive_weights = adaptive_acquisition_weights(
            source_productivities, reliability, global_confidence,
            meaningful_stall_age
        )

    p = float(np.clip(source_productivities.get(source, 0.0), 0.0, 1.0))
    global_sources = {'BWR', 'NOVELTY', 'COVERAGE'}
    local_sources = {'BMR', 'INCUMBENT', 'POLL', 'SURR_LOCAL', 'QUAD_TR', 'ANISO', 'PRECISION'}

    score = adaptive_weights['productivity'] * p

    # Existing V19 state-control bonuses are intentionally preserved. They are
    # not the four acquisition mixture weights and therefore retain V19 logic.
    if source in global_sources:
        score += V19_LOW_RELIABILITY_EXPLORATION_BONUS * max(
            0.0, V19_LOW_RELIABILITY_THRESHOLD - reliability)
        score += 0.20 * (1.0 - global_confidence)
        if state in ('RECOVERY', 'DEEP_RECOVERY', 'ESCAPE'):
            score += 0.15
        if meaningful_stall_age >= V19_LONG_STALL_FES:
            score += 0.15
    elif source in local_sources:
        score += 0.15 * reliability + 0.10 * global_confidence
        if state in ('LOCAL', 'INTENSIFY', 'MICRO_LOCAL', 'ANISO_LOCAL',
                     'PRECISION_LOCAL', 'PRECISION_RETRY'):
            score += 0.12
    return float(score)


def select_batch(records, state, batch_size, allow_extra_bmr=False,
                 source_productivities=None, reliability=0.5,
                 global_confidence=0.5, meaningful_stall_age=0,
                 audit_log=None, cycle=None, true_fe=None):
    """V19 evidence-adaptive TRUE batch selection with adaptive weights.

    Guarantees one global and one local slot when both classes are available;
    all remaining slots compete using verified source productivity, surrogate
    ranking, novelty and within-batch diversity.  Their four mixture weights
    are recomputed online from current search evidence and sum to one.
    """
    if batch_size <= 0 or len(records) == 0:
        return []
    if source_productivities is None:
        source_productivities = {}

    weights = adaptive_acquisition_weights(
        source_productivities=source_productivities,
        reliability=reliability,
        global_confidence=global_confidence,
        meaningful_stall_age=meaningful_stall_age,
    )

    admissible = dsds_admissible(records)
    base = admissible if len(admissible) else records
    selected = []
    global_sources = {'BWR', 'NOVELTY', 'COVERAGE'}
    local_sources = {'BMR', 'INCUMBENT', 'POLL', 'SURR_LOCAL', 'QUAD_TR', 'ANISO', 'PRECISION'}

    def duplicate(r):
        return any(np.linalg.norm(r.z - s.z) < 1e-12 for s in selected)

    def source_count(source):
        return sum(1 for r in selected if r.source == source)

    def allowed(r):
        if duplicate(r):
            return False
        if r.source == 'BMR' and not allow_extra_bmr and source_count('BMR') >= MAX_BMR_TRUE_SLOTS:
            return False
        return True

    selection_stage = [0]

    def rank_pool(pool, pool_role='COMPETITIVE'):
        pool = [r for r in pool if allowed(r)]
        if not pool:
            return None

        pred = np.asarray([r.pred for r in pool], dtype=float)
        nov = np.asarray([r.nc for r in pool], dtype=float)
        pred_good = 1.0 - robust_scale(pred)
        nov_good = robust_scale(nov)

        if selected:
            S = np.vstack([r.z for r in selected])
            P = np.vstack([r.z for r in pool])
            div_good = robust_scale(min_distances_blocked(P, S))
        else:
            div_good = np.ones(len(pool))

        # Evidence-adaptive restart pressure. This replaces the old fixed
        # +0.50 restart bonus while preserving restart_priority as an indicator.
        restart_bonus = adaptive_restart_bonus(
            reliability=reliability,
            global_confidence=global_confidence,
            meaningful_stall_age=meaningful_stall_age,
        )

        scores = np.empty(len(pool), dtype=float)
        for i, r in enumerate(pool):
            src = v19_source_score(
                r.source, source_productivities, reliability,
                global_confidence, state, meaningful_stall_age,
                adaptive_weights=weights
            )
            scores[i] = (
                src
                + weights['prediction'] * pred_good[i]
                + weights['novelty'] * nov_good[i]
                + weights['diversity'] * div_good[i]
                + restart_bonus * float(bool(r.restart_priority))
            )
        winner = int(np.argmax(scores))
        if audit_log is not None:
            selection_stage[0] += 1
            for i, r in enumerate(pool):
                src = v19_source_score(r.source, source_productivities, reliability, global_confidence, state, meaningful_stall_age, adaptive_weights=weights)
                row = {
                    'cycle': cycle, 'true_fe_before_batch': true_fe, 'selection_stage': selection_stage[0],
                    'pool_role': pool_role, 'pool_size': len(pool), 'source': r.source, 'basin_idx': r.basin_idx,
                    'restart_priority': bool(r.restart_priority), 'predicted_f': float(r.pred), 'novelty_nc': float(r.nc),
                    'prediction_good': float(pred_good[i]), 'novelty_good': float(nov_good[i]), 'diversity_good': float(div_good[i]),
                    'source_productivity': float(source_productivities.get(r.source, 0.0)), 'source_score': float(src),
                    'weight_productivity': float(weights['productivity']), 'weight_prediction': float(weights['prediction']),
                    'weight_novelty': float(weights['novelty']), 'weight_diversity': float(weights['diversity']),
                    'weighted_prediction': float(weights['prediction'] * pred_good[i]),
                    'weighted_novelty': float(weights['novelty'] * nov_good[i]),
                    'weighted_diversity': float(weights['diversity'] * div_good[i]),
                    'restart_bonus': float(restart_bonus if r.restart_priority else 0.0), 'total_selection_score': float(scores[i]),
                    'selected_at_this_stage': bool(i == winner), 'reliability': float(reliability),
                    'global_confidence': float(global_confidence), 'meaningful_stall_age': int(meaningful_stall_age)
                }
                for j, val in enumerate(r.z): row[f'z{j+1}'] = float(val)
                audit_log.append(row)
        return pool[winner]

    # Preserve structural exploration/exploitation: one global and one local
    # TRUE-evaluation slot when both classes are available.
    for srcset in (global_sources, local_sources):
        if len(selected) >= batch_size:
            break
        r = rank_pool([q for q in base if q.source in srcset], 'GUARANTEED_GLOBAL' if srcset is global_sources else 'GUARANTEED_LOCAL')
        if r is None:
            r = rank_pool([q for q in records if q.source in srcset], 'FALLBACK_GLOBAL' if srcset is global_sources else 'FALLBACK_LOCAL')
        if r is not None:
            selected.append(r)

    # Remaining TRUE FEs are fully evidence-adaptive.
    while len(selected) < batch_size:
        r = rank_pool(base, 'COMPETITIVE_ADMISSIBLE')
        if r is None:
            r = rank_pool(records, 'COMPETITIVE_FALLBACK')
        if r is None:
            break
        selected.append(r)

    return selected[:batch_size]


class V19RecoveryMemory:
    """Tracks whether recovery episodes create meaningful verified progress."""
    def __init__(self):
        self.failed_episodes = 0
        self.episode_start_best = None
        self.in_recovery = False

    def enter(self, best):
        if not self.in_recovery:
            self.episode_start_best = float(best)
            self.in_recovery = True

    def exit(self, best):
        if not self.in_recovery or self.episode_start_best is None:
            return False
        rel = max(0.0, self.episode_start_best - float(best)) / max(abs(self.episode_start_best), 1.0)
        success = rel >= V19_RECOVERY_SUCCESS_REL
        self.failed_episodes = 0 if success else self.failed_episodes + 1
        self.in_recovery = False
        self.episode_start_best = None
        return success

    def hard_reset_due(self):
        return self.failed_episodes >= V19_RECOVERY_FAIL_LIMIT

def stagnating(best_history, current_fe, last_recovery_fe):
    if current_fe - last_recovery_fe < RECOVERY_COOLDOWN_TRUE_FES:
        return False
    if len(best_history) < STALL_TRUE_FES + 1:
        return False
    older = best_history[-STALL_TRUE_FES - 1]
    current = best_history[-1]
    rel_imp = (older - current) / max(abs(older), 1.0)
    return rel_imp <= 1e-08

class _DiscardAudit:
    def append(self, row):
        pass


class _StreamingAudit:
    """Bounded CSV buffer: complete rows on disk, only winners retained."""
    def __init__(self, path, keep_winners=False):
        self.path = path
        self.fields = None
        self.buffer = []
        self.winners = []
        self.keep_winners = keep_winners
        if os.path.exists(path):
            os.remove(path)

    def append(self, row):
        if self.fields is None:
            self.fields = list(row)
        self.buffer.append(row)
        if self.keep_winners and row.get('selected_at_this_stage', False):
            self.winners.append(row)
        if len(self.buffer) >= 512:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        exists = os.path.exists(self.path)
        with open(self.path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields)
            if not exists:
                writer.writeheader()
            writer.writerows(self.buffer)
        self.buffer.clear()


def _trajectory_digest(archive):
    digest = hashlib.sha256()
    digest.update(np.asarray(archive.X_array, dtype='<f8').tobytes())
    digest.update(np.asarray(archive.y_array, dtype='<f8').tobytes())
    return digest.hexdigest()


def mb_dtaa_pfsa_v19(function_id=FUNCTION_ID, dimension=DIM, seed=MASTER_SEED, max_true_fes=MAX_TRUE_FES, verbose=True, result_dir=None, run_id=None, show_plot=False, log_prefix='', detailed=False):
    # result_dir/run_id affect ONLY file naming; optimizer logic is unchanged.
    if result_dir is None:
        result_dir = RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    run_suffix = f"run{int(run_id):02d}_" if run_id is not None else ""
    rng = np.random.default_rng(seed)
    max_true_fes = resolve_true_fe_budget(dimension=dimension, requested_budget=max_true_fes)
    problem = CEC2017Problem(function_id=function_id, dimension=dimension)
    if verbose:
        problem.print_metadata()
    archive = TrueArchive(dim=dimension)
    # COMPLETE RESEARCH AUDIT LOGS: reporting only; search logic is unchanged.
    doe_stepwise_log = [] if detailed else _DiscardAudit()
    rbf_fit_log = [] if detailed else _DiscardAudit()
    basin_state_log = [] if detailed else _DiscardAudit()
    prefix = f'F{function_id}_D{dimension}_{run_suffix}seed{seed}'
    candidate_ranking_log = (_StreamingAudit(os.path.join(result_dir, prefix + '_04_all_generated_candidates.csv')) if detailed else None)
    exact_selection_audit_log = (_StreamingAudit(os.path.join(result_dir, prefix + '_05_exact_sequential_selection_scores.csv'), keep_winners=True) if detailed else None)
    selected_eval_log = [] if detailed else _DiscardAudit()
    recovery_event_log = [] if detailed else _DiscardAudit()
    reliability_update_log = [] if detailed else _DiscardAudit()
    reliability = ReliabilityTracker(RELIABILITY_WINDOW)
    source_credit = SourceCreditTracker(SOURCE_CREDIT_WINDOW)
    basin_manager = BasinManager(MAX_BASINS, LOCAL_POP_SIZE, rng)
    t0 = time.time()
    n_init = min(INITIAL_DOE, max_true_fes)
    X0 = lhs(n=n_init, dim=dimension, lb=problem.lb, ub=problem.ub, seed=seed)
    best_history = []
    if verbose:
        print()
        print('=' * 112)
        print(f'MB-DTASR-PFSA V19 | CEC-2017 F{function_id}')
        print('=' * 112)
        print(f'Dimension                  : {dimension}')
        print(f'TRUE FE budget             : {max_true_fes}')
        print(f'Initial DOE                : {n_init}')
        print(f'Global population          : {GLOBAL_POP_SIZE}')
        print(f'Local population/basin     : {LOCAL_POP_SIZE}')
        print(f'Maximum basins             : {MAX_BASINS}')
        print(f'TR radius initial/min/max  : {TR_RADIUS_INITIAL:.3f}/{TR_RADIUS_MIN:.3f}/{TR_RADIUS_MAX:.3f}')
        print(f'TR shrink patience         : {FAILURE_PATIENCE} BMR failures')
        print(f'Stale basin threshold      : {STALE_BASIN_TRUE_FES} TRUE FEs')
        print('Global optimizer            : TRUE-anchored BWR')
        print('Local optimizer             : BMR TR + full rotated quadratic TR + magnitude-aware precision')
        print('Terminal escalation          : counted TRUE rotated Newton trust-region')
        print(f'Newton trigger/cooldown      : {NEWTON_TR_TRIGGER_FES}/{NEWTON_TR_COOLDOWN_FES} TRUE FEs')
        print('Selection                   : role-aware DSDS + productivity + slow-progress recovery')
        print('Mandatory TRUE slots        : BWR + BMR in every state')
        print('Stale-basin replacement     : BWR/Sobol novelty + DEEP_RECOVERY reseeding')
        print('Reference used by search    : NO')
        print('=' * 112)
    for x in X0:
        y0 = problem.evaluate(x)
        archive.add(x, y0)
        best_history.append(archive.best_y())
        drow={'true_fe': int(problem.true_fe), 'phase':'INITIAL_DOE', 'true_f':float(y0), 'best_so_far_f':float(archive.best_y())}
        z0=normalize_x(np.asarray(x)[None,:], problem.lb, problem.ub)[0]
        for j,v in enumerate(x): drow[f'x{j+1}']=float(v)
        for j,v in enumerate(z0): drow[f'z{j+1}']=float(v)
        doe_stepwise_log.append(drow)
    if verbose:
        print(f'\nInitial TRUE best = {archive.best_y():.12e}\n')
    X = archive.X_array
    y = archive.y_array
    Xz = normalize_x(X, problem.lb, problem.ub)
    global_pop = initialize_global_population(Xz, y, GLOBAL_POP_SIZE, rng)
    basin_manager.update_from_archive(Xz, y, seed=seed, current_fe=problem.true_fe)
    cycle = 0
    last_global_improvement_fe = problem.true_fe
    last_meaningful_improvement_fe = problem.true_fe
    last_global_best = archive.best_y()
    last_meaningful_best = archive.best_y()
    last_local_success_fe = -10 ** 9
    last_aniso_success_fe = -10 ** 9
    last_precision_success_fe = -10 ** 9
    local_mode_success = LocalModeSuccessTracker(LOCAL_MODE_SUCCESS_WINDOW)
    precision_memory = PrecisionRadiusMemory()
    newton_memory = TrueNewtonTRMemory()
    last_newton_fe = -NEWTON_TR_COOLDOWN_FES
    aniso_failed_cycles = 0
    precision_retry_failed_cycles = 0
    precision_retry_flag = False
    recovery_remaining = 0
    deep_recovery_remaining = 0
    last_deep_recovery_fe = -DEEP_RECOVERY_COOLDOWN_TRUE_FES
    last_recovery_fe = -RECOVERY_COOLDOWN_TRUE_FES
    escape_remaining = 0
    last_escape_fe = -V17_ESCAPE_COOLDOWN_FES
    last_basin_shock_fe = -V17_BASIN_SHOCK_COOLDOWN_FES
    last_saturation_action_fe = -V18_SATURATION_COOLDOWN_FES
    v19_recovery_memory = V19RecoveryMemory()
    v19_hard_resets = 0
    cycle_log = []
    while problem.true_fe < max_true_fes:
        cycle += 1
        X = archive.X_array
        y = archive.y_array
        Xz = normalize_x(X, problem.lb, problem.ub)
        Xg, yg = select_global_training(Xz, y, GLOBAL_RBF_MAX_POINTS, rng)
        global_model = RegularizedRBF(ridge=RBF_RIDGE)
        global_model.fit(Xg, yg)
        rbf_fit_log.append({'cycle':cycle,'true_fe':int(problem.true_fe),'model':'GLOBAL_RBF','n_archive':len(y),'n_training':len(yg),'sigma':float(global_model.sigma),'y_mean':float(global_model.y_mean),'y_std':float(global_model.y_std),'ridge':float(global_model.ridge)})
        basin_manager.update_from_archive(Xz, y, seed=seed + cycle, current_fe=problem.true_fe)
        for bi,b in enumerate(basin_manager.basins):
            brow={'cycle':cycle,'true_fe':int(problem.true_fe),'basin_idx':bi,'best_y':float(b.best_y),'radius':float(b.radius),'failure_count':int(b.failure_count),'last_improvement_fe':int(b.last_improvement_fe),'true_bmr_evals':int(b.true_bmr_evals),'true_bmr_successes':int(b.true_bmr_successes),'quarantine_until_fe':int(b.quarantine_until_fe),'saturation_count':int(b.saturation_count)}
            for j,v in enumerate(b.center): brow[f'center_z{j+1}']=float(v)
            for j,v in enumerate(b.best_z): brow[f'best_z{j+1}']=float(v)
            basin_state_log.append(brow)
        R_before = reliability.reliability()
        if recovery_remaining == 0 and stagnating(best_history, problem.true_fe, last_recovery_fe):
            recovery_remaining = RECOVERY_CYCLES
            last_recovery_fe = problem.true_fe
        current_best_z = Xz[int(np.argmin(y))].copy()
        recent_improvement_flag = recent_true_improvement(best_history, RECENT_IMPROVEMENT_WINDOW_TRUE_FES)
        incumbent_credit = incumbent_basin_credit(basin_manager.basins, current_best_z)
        stall_age = true_stall_age(current_fe=problem.true_fe, last_global_improvement_fe=last_global_improvement_fe)
        meaningful_stall_age = true_stall_age(current_fe=problem.true_fe, last_global_improvement_fe=last_meaningful_improvement_fe)
        slow_progress_flag, slow_progress_rel_gain = slow_progress(best_history, window=SLOW_PROGRESS_WINDOW_FES, rel_tol=SLOW_PROGRESS_REL_TOL)
        # V17: temporary global escape after prolonged lack of meaningful progress.
        if escape_remaining == 0 and meaningful_stall_age >= V17_ESCAPE_TRIGGER_FES and problem.true_fe - last_escape_fe >= V17_ESCAPE_COOLDOWN_FES:
            escape_remaining = V17_ESCAPE_CYCLES
            last_escape_fe = problem.true_fe
        # V17: shock only non-incumbent basins, preserving the verified incumbent basin.
        basin_shock_count = 0
        if meaningful_stall_age >= V17_BASIN_SHOCK_TRIGGER_FES and problem.true_fe - last_basin_shock_fe >= V17_BASIN_SHOCK_COOLDOWN_FES:
            basin_shock_count = v17_shock_nonincumbent_basins(basin_manager, current_best_z, Xz, y, rng, problem.true_fe)
            last_basin_shock_fe = problem.true_fe
        # V18: detect locally solved/wrong basins and quarantine/regenerate them.
        saturation_info = {'quarantined': 0, 'regenerated': 0, 'incumbent_saturated': False}
        saturated_ids = v18_detect_saturated_basins(basin_manager, problem.true_fe, meaningful_stall_age)
        if len(saturated_ids) > 0 and problem.true_fe - last_saturation_action_fe >= V18_SATURATION_COOLDOWN_FES:
            saturation_info = v18_quarantine_and_regenerate(basin_manager, saturated_ids, current_best_z, Xz, rng, problem.true_fe)
            last_saturation_action_fe = int(problem.true_fe)
            if saturation_info['incumbent_saturated']:
                escape_remaining = max(escape_remaining, V18_SATURATION_ESCAPE_CYCLES)
                last_escape_fe = int(problem.true_fe)
        inc_idx_pre_newton = incumbent_basin_index(basin_manager.basins, current_best_z)
        inc_quarantined_pre_newton = inc_idx_pre_newton >= 0 and v18_is_quarantined(basin_manager.basins[inc_idx_pre_newton], problem.true_fe)
        newton_due = (not inc_quarantined_pre_newton) and meaningful_stall_age >= NEWTON_TR_TRIGGER_FES and problem.true_fe - last_newton_fe >= NEWTON_TR_COOLDOWN_FES and (problem.true_fe < max_true_fes)
        if newton_due:
            nbefore = float(archive.best_y())
            nout = run_true_newton_tr_episode(problem=problem, archive=archive, best_history=best_history, rng=rng, max_true_fes=max_true_fes, memory=newton_memory, max_iters=NEWTON_TR_MAX_ITERS)
            last_newton_fe = int(problem.true_fe)
            nafter = float(archive.best_y())
            if verbose:
                print(f"[V19 TRUE-NEWTON-TR] FE {problem.true_fe:5d}/{max_true_fes} | used={nout['used']:3d} | accepted={nout['accepted']} | Rtr={nout['radius']:.3e} | rho={nout['rho_last']:.3e} | best {nbefore:.12e} -> {nafter:.12e} | dF={nbefore - nafter:.3e}")
            if nafter < last_global_best:
                old_global_best = float(last_global_best)
                last_global_best = nafter
                last_global_improvement_fe = problem.true_fe
                meaningful_now, _, _ = magnitude_aware_success(old_global_best, nafter)
                if meaningful_now:
                    last_meaningful_best = nafter
                    last_meaningful_improvement_fe = problem.true_fe
                    last_local_success_fe = problem.true_fe
            if nout['used'] > 0:
                continue
        meaningful_recovery_due = meaningful_stall_age >= MEANINGFUL_STALL_RECOVERY_FES and (slow_progress_flag or meaningful_stall_age >= 2 * MEANINGFUL_STALL_RECOVERY_FES) and (problem.true_fe - last_recovery_fe >= MEANINGFUL_RECOVERY_COOLDOWN_FES)
        if meaningful_recovery_due and recovery_remaining == 0:
            recovery_remaining = SLOW_PROGRESS_RECOVERY_CYCLES
            last_recovery_fe = problem.true_fe
        local_success_active = recent_local_success_active(current_fe=problem.true_fe, last_local_success_fe=last_local_success_fe)
        local_credit_rate = source_credit.local_rate()
        global_credit_rate = source_credit.global_rate()
        source_productivities = source_credit.all_source_productivities()
        global_confidence = v19_global_confidence(Xz, y, current_best_z)
        recent_aniso_success = problem.true_fe - last_aniso_success_fe <= ANISO_SUCCESS_MEMORY_FES
        recent_precision_success = problem.true_fe - last_precision_success_fe <= PRECISION_SUCCESS_MEMORY_FES
        precision_flag = precision_mode_trigger(reliability=R_before, stall_age=stall_age, local_productivity=local_credit_rate, global_productivity=global_credit_rate, recent_precision_success=recent_precision_success)
        precision_flag = precision_flag and precision_success_ratio_ok(local_mode_success)
        anisotropic_flag = anisotropic_mode_trigger(reliability=R_before, stall_age=stall_age, local_productivity=local_credit_rate, global_productivity=global_credit_rate, recent_aniso_success=recent_aniso_success)
        anisotropic_flag = anisotropic_flag and aniso_success_ratio_ok(local_mode_success)
        if aniso_failed_cycles >= ANISO_FAILURE_CYCLE_LIMIT:
            precision_retry_flag = True
            anisotropic_flag = False
            precision_flag = False
        intensify_flag = progress_triggered_intensify(reliability=R_before, stall_age=stall_age, recent_local_success=local_success_active)
        micro_local_flag = micro_local_trigger(reliability=R_before, stall_age=stall_age, local_credit=local_credit_rate, global_credit=global_credit_rate)
        if deep_recovery_remaining == 0 and (stall_age >= DEEP_RECOVERY_TRIGGER_FES or meaningful_stall_age >= MEANINGFUL_STALL_DEEP_FES) and (problem.true_fe - last_deep_recovery_fe >= DEEP_RECOVERY_COOLDOWN_TRUE_FES):
            deep_recovery_remaining = DEEP_RECOVERY_CYCLES
            last_deep_recovery_fe = problem.true_fe
        state = choose_state(reliability=R_before, recovery_remaining=recovery_remaining, deep_recovery_remaining=deep_recovery_remaining, recent_improvement=recent_improvement_flag, stall_age=stall_age, intensify_flag=intensify_flag, micro_local_flag=micro_local_flag, local_productivity=local_credit_rate, global_productivity=global_credit_rate, anisotropic_flag=anisotropic_flag, precision_flag=precision_flag, precision_retry_flag=precision_retry_flag)
        if escape_remaining > 0 and deep_recovery_remaining == 0:
            state = 'ESCAPE'
        recovery_state = state in ('RECOVERY', 'DEEP_RECOVERY', 'ESCAPE')
        if recovery_state:
            v19_recovery_memory.enter(archive.best_y())
        elif v19_recovery_memory.in_recovery:
            v19_recovery_memory.exit(archive.best_y())
        if v19_recovery_memory.hard_reset_due():
            # Generic escalation after repeated failed recovery episodes.
            # It changes search distribution without any function-ID/optimum information.
            n_reset = max(1, int(np.ceil(V19_GLOBAL_RESET_FRACTION * max(1, len(basin_manager.basins)))))
            reset_ids = list(range(len(basin_manager.basins)))
            inc_reset_idx = incumbent_basin_index(basin_manager.basins, current_best_z)
            reset_ids = [i for i in reset_ids if i != inc_reset_idx][:n_reset]
            if reset_ids:
                tmp_info = v18_quarantine_and_regenerate(basin_manager, reset_ids, current_best_z, Xz, rng, problem.true_fe)
                v19_hard_resets += int(tmp_info.get('regenerated', 0) > 0)
            escape_remaining = max(escape_remaining, V17_ESCAPE_CYCLES)
            deep_recovery_remaining = max(deep_recovery_remaining, DEEP_RECOVERY_CYCLES)
            v19_recovery_memory.failed_episodes = 0
        global_pop = truth_anchor_global_population(current_pop=global_pop, global_model=global_model, Xz_true=Xz, y_true=y, pop_size=GLOBAL_POP_SIZE, rng=rng)
        global_pred = global_model.predict(global_pop)
        if state == 'DEEP_RECOVERY':
            n_global = int(DEEP_RECOVERY_BWR_OFFSPRING_MULTIPLIER * GLOBAL_BWR_OFFSPRING)
        elif state == 'ESCAPE':
            n_global = int(V17_ESCAPE_BWR_MULTIPLIER * GLOBAL_BWR_OFFSPRING)
        elif state in ('GLOBAL', 'RECOVERY'):
            n_global = int(1.5 * GLOBAL_BWR_OFFSPRING)
        else:
            n_global = GLOBAL_BWR_OFFSPRING
        C_bwr = bwr_offspring(global_pop, global_pred, n_global, rng)
        C_bwr = remove_true_duplicates(C_bwr, Xz)
        if len(C_bwr) == 0:
            C_bwr = rng.random((max(20, dimension * 2), dimension))
        pred_bwr = global_model.predict(C_bwr)
        global_pop = update_global_population(global_pop, global_pred, C_bwr, pred_bwr, GLOBAL_POP_SIZE)
        restart_seeds = basin_manager.restart_stale_from_bwr(C_bwr, pred_bwr, current_fe=problem.true_fe, Xz_true=Xz, y_true=y)
        basins = basin_manager.basins
        local_models = {}
        for b_idx, b in enumerate(basins):
            if v18_is_quarantined(b, problem.true_fe):
                continue
            model = build_local_rbf(Xz, y, b.best_z, LOCAL_RBF_MAX_POINTS)
            if model is not None:
                local_models[b_idx] = model
        local_candidate_lists = []
        for b_idx, b in enumerate(basins):
            if v18_is_quarantined(b, problem.true_fe):
                continue
            local_model = local_models.get(b_idx, global_model)
            lo = clip01(b.best_z - b.radius)
            hi = clip01(b.best_z + b.radius)
            if len(b.population) != LOCAL_POP_SIZE:
                b.population = basin_manager._init_local_population(b.best_z, b.radius)
            b.population = np.minimum(np.maximum(b.population, lo), hi)
            local_pred = local_model.predict(b.population)
            n_local = LOCAL_BMR_OFFSPRING
            if state in ('LOCAL', 'INTENSIFY', 'MICRO_LOCAL', 'ANISO_LOCAL', 'PRECISION_LOCAL', 'PRECISION_RETRY'):
                n_local = int(1.5 * n_local)
            elif state == 'RECOVERY':
                n_local = max(30, int(0.6 * n_local))
            Cb = bmr_trust_region_offspring(b.population, local_pred, b.best_z, b.radius, n_local, rng)
            Cb = remove_true_duplicates(Cb, Xz)
            if len(Cb) == 0:
                continue
            pred_cb = local_model.predict(Cb)
            P = np.vstack([b.population, Cb])
            Pf = np.concatenate([local_pred, pred_cb])
            ids = np.argsort(Pf)[:LOCAL_POP_SIZE]
            b.population = P[ids].copy()
            local_candidate_lists.append((b_idx, Cb))
        n_cov = COVERAGE_CANDIDATES
        if state in ('GLOBAL', 'RECOVERY', 'ESCAPE'):
            n_cov *= 2
        C_cov = rng.random((n_cov, dimension))
        C_cov = remove_true_duplicates(C_cov, Xz)
        if state == 'DEEP_RECOVERY':
            C_novel = generate_deep_novelty_candidates(Xz_true=Xz, dimension=dimension, rng=rng)
            global_pop = reseed_global_population_for_deep_recovery(global_pop=global_pop, Xz_true=Xz, y_true=y, deep_novelty=C_novel, pop_size=GLOBAL_POP_SIZE, rng=rng)
        elif state == 'ESCAPE':
            C_novel = generate_novelty_candidates(Xz_true=Xz, dimension=dimension, rng=rng, n_keep=V17_ESCAPE_NOVELTY_KEEP)
        else:
            C_novel = generate_novelty_candidates(Xz_true=Xz, dimension=dimension, rng=rng, n_keep=NOVELTY_KEEP)
        incumbent_z = Xz[int(np.argmin(y))].copy()
        incumbent_basin_idx = incumbent_basin_index(basins, incumbent_z)
        incumbent_quarantined = incumbent_basin_idx >= 0 and v18_is_quarantined(basins[incumbent_basin_idx], problem.true_fe)
        if incumbent_basin_idx >= 0 and incumbent_basin_idx in local_models and not incumbent_quarantined:
            incumbent_model = local_models[incumbent_basin_idx]
        else:
            incumbent_model = global_model
        quad_model = None if incumbent_quarantined else build_true_local_quadratic(Xz_true=Xz, y_true=y, incumbent_z=incumbent_z)
        aniso_model = None if incumbent_quarantined else build_anisotropic_local_model(Xz_true=Xz, y_true=y, incumbent_z=incumbent_z, stall_age=stall_age)
        if incumbent_quarantined:
            C_aniso = np.empty((0, dimension))
            C_precision = np.empty((0, dimension))
            C_quad = np.empty((0, dimension))
            C_inc = np.empty((0, dimension))
            C_poll = np.empty((0, dimension))
            C_surr_local = np.empty((0, dimension))
        else:
            C_aniso = anisotropic_candidates(incumbent_z=incumbent_z, aniso_model=aniso_model, local_model=incumbent_model, Xz_true=Xz, rng=rng)
            precision_base_radius = precision_memory.retry_radius() if state == 'PRECISION_RETRY' else precision_memory.radius
            C_precision = precision_candidates(incumbent_z=incumbent_z, aniso_model=aniso_model, local_model=incumbent_model, Xz_true=Xz, rng=rng, stall_age=stall_age, explicit_base_radius=precision_base_radius)
            C_quad = quadratic_tr_candidates(incumbent_z=incumbent_z, quad_model=quad_model, Xz_true=Xz, rng=rng, stall_age=stall_age)
            if state in ('INTENSIFY', 'LOCAL', 'MICRO_LOCAL'):
                C_inc = generate_multiscale_incumbent_candidates(incumbent_z=incumbent_z, local_model=incumbent_model, Xz_true=Xz, rng=rng)
            else:
                C_inc = generate_incumbent_refinement_candidates(incumbent_z=incumbent_z, local_model=incumbent_model, Xz_true=Xz, rng=rng)
            C_poll = generate_coordinate_poll_candidates(incumbent_z=incumbent_z, Xz_true=Xz, stall_age=stall_age)
            C_surr_local = surrogate_local_optimum_candidates(incumbent_z=incumbent_z, local_model=incumbent_model, Xz_true=Xz, rng=rng, radius=max(INCUMBENT_REFINE_MIN_RADIUS, SURR_LOCAL_RADIUS / np.sqrt(1.0 + stall_age / 40.0)) if state in ('INTENSIFY', 'LOCAL') else SURR_LOCAL_RADIUS, n_restarts=PERSISTENT_SURR_LOCAL_RESTARTS if state == 'INTENSIFY' else SURR_LOCAL_RESTARTS)
        records = []
        records.extend(evaluate_candidate_information(C_bwr, source='BWR', basin_idx=-1, global_model=global_model, local_models=local_models, Xz_true=Xz))
        for b_idx, Cb in local_candidate_lists:
            records.extend(evaluate_candidate_information(Cb, source='BMR', basin_idx=b_idx, global_model=global_model, local_models=local_models, Xz_true=Xz))
        records.extend(evaluate_candidate_information(C_inc, source='INCUMBENT', basin_idx=incumbent_basin_idx, global_model=global_model, local_models=local_models, Xz_true=Xz))
        records.extend(evaluate_candidate_information(C_precision, source='PRECISION', basin_idx=incumbent_basin_idx, global_model=global_model, local_models=local_models, Xz_true=Xz))
        records.extend(evaluate_candidate_information(C_aniso, source='ANISO', basin_idx=incumbent_basin_idx, global_model=global_model, local_models=local_models, Xz_true=Xz))
        if len(C_quad) > 0:
            quad_pred = quad_model.predict(C_quad) if quad_model is not None else global_model.predict(C_quad)
            quad_nc = nc_values(C_quad, Xz, NC_NEIGHBORS)
            for i in range(len(C_quad)):
                records.append(CandidateRecord(z=C_quad[i].copy(), pred=float(quad_pred[i]), nc=float(quad_nc[i]), source='QUAD_TR', basin_idx=int(incumbent_basin_idx), restart_priority=False))
        records.extend(evaluate_candidate_information(C_poll, source='POLL', basin_idx=incumbent_basin_idx, global_model=global_model, local_models=local_models, Xz_true=Xz))
        records.extend(evaluate_candidate_information(C_surr_local, source='SURR_LOCAL', basin_idx=incumbent_basin_idx, global_model=global_model, local_models=local_models, Xz_true=Xz))
        records.extend(evaluate_candidate_information(C_novel, source='NOVELTY', basin_idx=-1, global_model=global_model, local_models=local_models, Xz_true=Xz))
        records.extend(evaluate_candidate_information(C_cov, source='COVERAGE', basin_idx=-1, global_model=global_model, local_models=local_models, Xz_true=Xz))
        for restart_basin_idx, restart_z, restart_pred in restart_seeds:
            restart_nc = float(nc_values(restart_z[None, :], Xz, NC_NEIGHBORS)[0])
            records.append(CandidateRecord(z=restart_z.copy(), pred=float(restart_pred), nc=restart_nc, source='BWR', basin_idx=int(restart_basin_idx), restart_priority=True))
        remaining = max_true_fes - problem.true_fe
        batch_n = min(BATCH_SIZE, remaining)
        cycle_weights = adaptive_acquisition_weights(source_productivities, R_before, global_confidence, meaningful_stall_age)
        batch = select_batch(records=records, state=state, batch_size=batch_n, allow_extra_bmr=recent_improvement_flag and incumbent_credit >= LOCAL_BASIN_CREDIT_MIN, source_productivities=source_productivities, reliability=R_before, global_confidence=global_confidence, meaningful_stall_age=meaningful_stall_age, audit_log=exact_selection_audit_log, cycle=cycle, true_fe=problem.true_fe)
        # One-row-per-generated-candidate cycle audit.
        if detailed and records:
            cpred=np.asarray([r.pred for r in records],float); cnov=np.asarray([r.nc for r in records],float)
            cpg=1.0-robust_scale(cpred); cng=robust_scale(cnov)
            for ci,r in enumerate(records):
                crow={'cycle':cycle,'true_fe_before_batch':int(problem.true_fe),'candidate_id':ci,'state':state,'source':r.source,'basin_idx':r.basin_idx,'restart_priority':bool(r.restart_priority),'selected_for_true_eval':bool(any(np.linalg.norm(r.z-q.z)<1e-12 for q in batch)),'predicted_f':float(r.pred),'novelty_nc':float(r.nc),'prediction_good_scaled':float(cpg[ci]),'novelty_good_scaled':float(cng[ci]),'source_productivity':float(source_productivities.get(r.source,0.0)),'reliability':float(R_before),'global_confidence':float(global_confidence),'weight_productivity':float(cycle_weights['productivity']),'weight_prediction':float(cycle_weights['prediction']),'weight_novelty':float(cycle_weights['novelty']),'weight_diversity':float(cycle_weights['diversity'])}
                for j,v in enumerate(r.z): crow[f'z{j+1}']=float(v)
                candidate_ranking_log.append(crow)
        if detailed:
            candidate_ranking_log.flush()
            exact_selection_audit_log.flush()
        best_before_cycle = archive.best_y()
        source_count = {'BWR': 0, 'BMR': 0, 'INCUMBENT': 0, 'POLL': 0, 'SURR_LOCAL': 0, 'QUAD_TR': 0, 'ANISO': 0, 'PRECISION': 0, 'NOVELTY': 0, 'COVERAGE': 0}
        restart_true_count = 0
        bmr_successes = 0
        bmr_failures = 0
        for rec in batch:
            x = denormalize_x(rec.z, problem.lb, problem.ub)
            best_before_this_eval = archive.best_y()
            true_y = problem.evaluate(x)
            archive.add(x, true_y)
            if rec.source in ('INCUMBENT', 'POLL', 'SURR_LOCAL', 'QUAD_TR', 'ANISO', 'PRECISION') and true_y < best_before_this_eval:
                last_local_success_fe = problem.true_fe
            if rec.source == 'ANISO' and true_y < best_before_this_eval:
                last_aniso_success_fe = problem.true_fe
            mag_success, absolute_gain_this_eval, relative_gain_this_eval = magnitude_aware_success(best_before_this_eval, true_y)
            if rec.source == 'PRECISION' and mag_success:
                last_precision_success_fe = problem.true_fe
                precision_retry_failed_cycles = 0
                precision_retry_flag = False
            reliability.add(true_y, rec.pred)
            erow={'cycle':cycle,'true_fe':int(problem.true_fe),'state':state,'source':rec.source,'basin_idx':int(rec.basin_idx),'restart_priority':bool(rec.restart_priority),'predicted_f':float(rec.pred),'true_f':float(true_y),'prediction_error':float(true_y-rec.pred),'abs_prediction_error':float(abs(true_y-rec.pred)),'best_before_eval':float(best_before_this_eval),'best_after_eval':float(archive.best_y()),'improvement':float(best_before_this_eval-archive.best_y()),'reliability_before_cycle':float(R_before),'global_confidence':float(global_confidence)}
            for j,v in enumerate(x): erow[f'x{j+1}']=float(v)
            for j,v in enumerate(rec.z): erow[f'z{j+1}']=float(v)
            selected_eval_log.append(erow)
            reliability_update_log.append({'cycle':cycle,'true_fe':int(problem.true_fe),'source':rec.source,'true_f':float(true_y),'predicted_f':float(rec.pred),'abs_prediction_error':float(abs(true_y-rec.pred)),'reliability_after_update':float(reliability.reliability())})
            best_history.append(archive.best_y())
            source_count[rec.source] += 1
            improved_global_this_eval = true_y < best_before_this_eval
            source_success = mag_success if rec.source in ('ANISO', 'PRECISION', 'QUAD_TR') else improved_global_this_eval
            source_credit.add(rec.source, source_success, relative_gain=relative_gain_this_eval)
            local_mode_success.add(rec.source, source_success)
            if rec.restart_priority:
                restart_true_count += 1
            if rec.source == 'BMR' and rec.basin_idx >= 0:
                improved = basin_manager.record_true_bmr_result(basin_idx=rec.basin_idx, z=rec.z, true_y=true_y, current_fe=problem.true_fe)
                if improved:
                    bmr_successes += 1
                else:
                    bmr_failures += 1
            if rec.restart_priority and rec.basin_idx >= 0 and (rec.basin_idx < len(basin_manager.basins)):
                b = basin_manager.basins[rec.basin_idx]
                b.best_y = float(true_y)
                b.best_z = rec.z.copy()
                b.center = rec.z.copy()
                b.last_improvement_fe = problem.true_fe
                b.failure_count = 0
            if problem.true_fe >= max_true_fes:
                break
        if recovery_remaining > 0:
            recovery_remaining -= 1
        if deep_recovery_remaining > 0:
            deep_recovery_remaining -= 1
        if escape_remaining > 0:
            escape_remaining -= 1
        best_after_cycle = archive.best_y()
        improvement = best_before_cycle - best_after_cycle
        cycle_rel_gain = max(0.0, improvement) / max(abs(best_before_cycle), 1.0)
        cycle_gain_threshold = max(MAG_SUCCESS_ABS_GAIN, MAG_SUCCESS_REL_GAIN * max(abs(best_before_cycle), 1.0))
        meaningful_cycle_improvement = improvement > 0.0 and improvement >= cycle_gain_threshold
        if state in ('PRECISION_LOCAL', 'PRECISION_RETRY'):
            if meaningful_cycle_improvement:
                precision_memory.on_success()
            elif improvement <= 0.0:
                precision_memory.on_failure_episode()
        if state == 'ANISO_LOCAL':
            if meaningful_cycle_improvement:
                aniso_failed_cycles = 0
            else:
                aniso_failed_cycles += 1
        elif improvement > 0.0:
            aniso_failed_cycles = 0
        if state == 'PRECISION_RETRY':
            if meaningful_cycle_improvement:
                precision_retry_failed_cycles = 0
                precision_retry_flag = False
            else:
                precision_retry_failed_cycles += 1
                if precision_retry_failed_cycles >= PRECISION_RETRY_FAILURE_LIMIT:
                    precision_memory.on_failure_episode()
                    precision_retry_failed_cycles = 0
                    precision_retry_flag = False
                    aniso_failed_cycles = 0
                    recovery_remaining = max(recovery_remaining, V13_RECOVERY_CYCLES)
        if best_after_cycle < last_global_best:
            old_global_best = float(last_global_best)
            last_global_best = best_after_cycle
            last_global_improvement_fe = problem.true_fe
            meaningful_now, _, _ = magnitude_aware_success(old_global_best, best_after_cycle)
            if meaningful_now:
                last_meaningful_best = best_after_cycle
                last_meaningful_improvement_fe = problem.true_fe
        R_after = reliability.reliability()
        radii = [b.radius for b in basin_manager.basins]
        failures = [b.failure_count for b in basin_manager.basins]
        stale_now = basin_manager.stale_indices(problem.true_fe)
        cycle_log.append({'cycle': cycle, 'true_fe': problem.true_fe, 'state': state, 'reliability_before': R_before, 'reliability_after': R_after, 'best_true_f': best_after_cycle, 'improvement': improvement, 'n_basins': len(basin_manager.basins), 'mean_tr_radius': float(np.mean(radii)) if len(radii) else np.nan, 'min_tr_radius': float(np.min(radii)) if len(radii) else np.nan, 'max_tr_radius': float(np.max(radii)) if len(radii) else np.nan, 'mean_failure_count': float(np.mean(failures)) if len(failures) else np.nan, 'true_BWR': source_count['BWR'], 'true_BMR': source_count['BMR'], 'true_INCUMBENT': source_count['INCUMBENT'], 'true_POLL': source_count['POLL'], 'true_SURR_LOCAL': source_count['SURR_LOCAL'], 'true_QUAD_TR': source_count['QUAD_TR'], 'true_ANISO': source_count['ANISO'], 'true_PRECISION': source_count['PRECISION'], 'true_NOVELTY': source_count['NOVELTY'], 'local_source_credit': float(local_credit_rate), 'global_source_credit': float(global_credit_rate), 'prod_BWR': float(source_productivities.get('BWR', 0.0)), 'prod_BMR': float(source_productivities.get('BMR', 0.0)), 'prod_INC': float(source_productivities.get('INCUMBENT', 0.0)), 'prod_POLL': float(source_productivities.get('POLL', 0.0)), 'prod_SR': float(source_productivities.get('SURR_LOCAL', 0.0)), 'prod_Q': float(source_productivities.get('QUAD_TR', 0.0)), 'prod_ANISO': float(source_productivities.get('ANISO', 0.0)), 'prod_PRECISION': float(source_productivities.get('PRECISION', 0.0)), 'prod_NOV': float(source_productivities.get('NOVELTY', 0.0)), 'micro_local_flag': bool(micro_local_flag), 'anisotropic_flag': bool(anisotropic_flag), 'precision_flag': bool(precision_flag), 'recent_aniso_success': bool(recent_aniso_success), 'recent_precision_success': bool(recent_precision_success), 'precision_retry_flag': bool(precision_retry_flag), 'aniso_failed_cycles': int(aniso_failed_cycles), 'precision_retry_failed_cycles': int(precision_retry_failed_cycles), 'aniso_success_ratio': float(local_mode_success.ratio('ANISO')), 'precision_success_ratio': float(local_mode_success.ratio('PRECISION')), 'precision_memory_radius': float(precision_memory.radius), 'true_COVERAGE': source_count['COVERAGE'], 'recent_true_improvement': bool(recent_improvement_flag), 'incumbent_basin_credit': float(incumbent_credit), 'stall_age': int(stall_age), 'meaningful_stall_age': int(meaningful_stall_age), 'intensify_flag': bool(intensify_flag), 'local_success_active': bool(local_success_active), 'deep_recovery_remaining': int(deep_recovery_remaining), 'bmr_successes': bmr_successes, 'bmr_failures': bmr_failures, 'basin_restarts_generated': len(restart_seeds), 'basin_restarts_true_verified': restart_true_count, 'currently_stale_basins': len(stale_now), 'recovery_remaining': recovery_remaining, 'last_recovery_fe': last_recovery_fe, 'escape_remaining': int(escape_remaining), 'basin_shock_count': int(basin_shock_count), 'saturated_basins': int(len(saturated_ids)), 'quarantined_basins': int(sum(v18_is_quarantined(b, problem.true_fe) for b in basin_manager.basins)), 'v18_regenerated_basins': int(saturation_info['regenerated']), 'v18_incumbent_quarantined': bool(incumbent_quarantined), 'v19_global_confidence': float(global_confidence), 'weight_productivity': float(cycle_weights['productivity']), 'weight_prediction': float(cycle_weights['prediction']), 'weight_novelty': float(cycle_weights['novelty']), 'weight_diversity': float(cycle_weights['diversity']), 'weight_sum': float(sum(cycle_weights.values())), 'v19_failed_recovery_episodes': int(v19_recovery_memory.failed_episodes), 'v19_hard_resets': int(v19_hard_resets)})
        if verbose:
            tr_mean = np.mean(radii) if len(radii) else np.nan
            tr_min = np.min(radii) if len(radii) else np.nan
            tr_max = np.max(radii) if len(radii) else np.nan
            print(f"{log_prefix}Cycle {cycle:4d} | FE {problem.true_fe:5d}/{max_true_fes} | {state:8s} | R={R_after:5.3f} | Basins={len(basins):2d} | TR={tr_min:5.3f}/{tr_mean:5.3f}/{tr_max:5.3f} | BWR/BMR/INC/POLL/SR/Q/AN/PX/NOV={source_count['BWR']}/{source_count['BMR']}/{source_count['INCUMBENT']}/{source_count['POLL']}/{source_count['SURR_LOCAL']}/{source_count['QUAD_TR']}/{source_count['ANISO']}/{source_count['PRECISION']}/{source_count['NOVELTY']} | Prog={int(recent_improvement_flag)} | Age={stall_age:3d} | MAge={meaningful_stall_age:3d} | BCR={incumbent_credit:4.2f} | INT={int(intensify_flag)} | MIC={int(micro_local_flag)} | AN={int(anisotropic_flag)} | PX={int(precision_flag)} | PR={int(precision_retry_flag)} | AF={aniso_failed_cycles:02d} | ASR={local_mode_success.ratio('ANISO'):4.2f} | PSR={local_mode_success.ratio('PRECISION'):4.2f} | Pr={precision_memory.radius:.1e} | LP/GP={local_credit_rate:4.2f}/{global_credit_rate:4.2f} | GC={global_confidence:4.2f} | W[P/F/N/D]={cycle_weights['productivity']:.3f}/{cycle_weights['prediction']:.3f}/{cycle_weights['novelty']:.3f}/{cycle_weights['diversity']:.3f} | LS={int(local_success_active)} | DR={deep_recovery_remaining} | ER={escape_remaining} | Q={sum(v18_is_quarantined(b, problem.true_fe) for b in basin_manager.basins)} | RG={saturation_info['regenerated']} | BMRsucc={bmr_successes} | Restart={restart_true_count} | best={best_after_cycle:.10e} | dF={improvement:.3e}", flush=True)
    runtime = time.time() - t0
    best_x = archive.best_x()
    best_f = archive.best_y()
    optimum = float(problem.reference_optimum)
    error = best_f - optimum if np.isfinite(optimum) else np.nan
    trajectory_sha256 = _trajectory_digest(archive)
    if not detailed:
        hist = np.asarray(best_history, dtype=float)
        conv = pd.DataFrame({'true_fe': np.arange(1, len(hist)+1),
                             'best_so_far_f': hist, 'error': hist - optimum})
        conv.to_csv(os.path.join(result_dir, 'convergence.csv'), index=False)
        # Compact cycle data also supports the suite's existing convergence reader.
        pd.DataFrame(cycle_log)[['cycle','true_fe','best_true_f']].to_csv(
            os.path.join(result_dir, prefix + '_cycles.csv'), index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(conv['true_fe'], np.maximum(hist-optimum, 1e-16) if np.isfinite(optimum) else np.maximum(np.abs(hist), 1e-16))
        ax.set(xlabel='TRUE function evaluations', ylabel='Best-so-far error',
               title=f'Convergence | F{function_id} | D={dimension} | Seed {seed}')
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, 'convergence.png'), dpi=450, bbox_inches='tight')
        fig.savefig(os.path.join(result_dir, 'convergence.pdf'), bbox_inches='tight')
        plt.close(fig)
        return {'method': 'MB-DTASR-PFSA V19', 'benchmark_metadata': problem.metadata(),
                'best_x': best_x, 'best_f': best_f, 'error': error, 'true_fes': problem.true_fe,
                'runtime': runtime, 'nonfinite_count': int(problem.nonfinite_count),
                'archive': pd.DataFrame(), 'cycles': pd.DataFrame(), 'best_history': hist,
                'trajectory_sha256': trajectory_sha256, 'detailed': False}
    candidate_ranking_log.flush()
    exact_selection_audit_log.flush()
    archive_df = pd.DataFrame(archive.X_array, columns=[f'x{i + 1}' for i in range(dimension)])
    archive_df['f'] = archive.y_array
    archive_df['error_postrun'] = archive_df['f'] - optimum
    archive_path = os.path.join(result_dir, f'F{function_id}_D{dimension}_{run_suffix}seed{seed}_archive.csv')
    archive_df.to_csv(archive_path, index=False)
    cycle_df = pd.DataFrame(cycle_log)
    cycle_path = os.path.join(result_dir, f'F{function_id}_D{dimension}_{run_suffix}seed{seed}_cycles.csv')
    cycle_df.to_csv(cycle_path, index=False)

    # V20: save the exact adaptive acquisition weights used cycle-by-cycle.
    weight_cols = ['cycle', 'true_fe', 'state', 'reliability_before',
                   'v19_global_confidence', 'meaningful_stall_age',
                   'weight_productivity', 'weight_prediction',
                   'weight_novelty', 'weight_diversity', 'weight_sum']
    weight_df = cycle_df[[c for c in weight_cols if c in cycle_df.columns]].copy()
    weight_path = os.path.join(result_dir, f'F{function_id}_D{dimension}_{run_suffix}seed{seed}_adaptive_weights.csv')
    weight_df.to_csv(weight_path, index=False)

    prefix = f'F{function_id}_D{dimension}_{run_suffix}seed{seed}'
    pd.DataFrame(doe_stepwise_log).to_csv(os.path.join(result_dir, f'{prefix}_01_initial_DOE_stepwise.csv'), index=False)
    pd.DataFrame(rbf_fit_log).to_csv(os.path.join(result_dir, f'{prefix}_02_RBF_fit_diagnostics.csv'), index=False)
    pd.DataFrame(basin_state_log).to_csv(os.path.join(result_dir, f'{prefix}_03_basin_state_stepwise.csv'), index=False)
    pd.DataFrame(selected_eval_log).to_csv(os.path.join(result_dir, f'{prefix}_06_selected_TRUE_evaluation_diagnostics.csv'), index=False)
    pd.DataFrame(reliability_update_log).to_csv(os.path.join(result_dir, f'{prefix}_07_reliability_update_stepwise.csv'), index=False)
    # Complete TRUE-FE numerical trajectory, including DOE and adaptive search.
    true_step_df = archive_df.copy()
    true_step_df.insert(0, 'true_fe', np.arange(1, len(true_step_df)+1, dtype=int))
    true_step_df.insert(1, 'phase', np.where(true_step_df['true_fe'] <= n_init, 'INITIAL_DOE', 'ADAPTIVE_SEARCH'))
    true_step_df['best_so_far_f'] = true_step_df['f'].cummin()
    true_step_df['improvement_from_previous_best'] = true_step_df['best_so_far_f'].shift(1) - true_step_df['best_so_far_f']
    true_step_df.loc[0, 'improvement_from_previous_best'] = 0.0
    true_step_df.to_csv(os.path.join(result_dir, f'{prefix}_08_TRUE_FE_complete_stepwise_numerical_values.csv'), index=False)

    if len(weight_df):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(weight_df['true_fe'], weight_df['weight_productivity'], label='Productivity')
        ax.plot(weight_df['true_fe'], weight_df['weight_prediction'], label='Prediction')
        ax.plot(weight_df['true_fe'], weight_df['weight_novelty'], label='Novelty')
        ax.plot(weight_df['true_fe'], weight_df['weight_diversity'], label='Diversity')
        ax.set_xlabel('True Function Evaluations')
        ax.set_ylabel('Adaptive acquisition weight')
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f'Adaptive Acquisition Weights | CEC-2017 F{function_id} | D={dimension}')
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f'F{function_id}_D{dimension}_{run_suffix}seed{seed}_adaptive_weights.png'), dpi=450, bbox_inches='tight')
        plt.close(fig)
    # COMPLETE PER-RUN RESEARCH VISUALIZATIONS (PNG + PDF).
    def _save_audit_fig(fig, stem):
        fig.tight_layout(); fig.savefig(os.path.join(result_dir, f'{prefix}_{stem}.png'), dpi=450, bbox_inches='tight'); fig.savefig(os.path.join(result_dir, f'{prefix}_{stem}.pdf'), bbox_inches='tight'); plt.close(fig)
    if len(cycle_df):
        plot_specs=[
            ('best_true_f','Best-so-far objective','09_best_objective_convergence'),
            ('improvement','Verified improvement','10_verified_improvement'),
            ('reliability_after','Surrogate reliability','11_surrogate_reliability'),
            ('v19_global_confidence','Global basin confidence','12_global_confidence'),
            ('mean_tr_radius','Mean trust-region radius','13_trust_region_evolution'),
            ('n_basins','Number of basins','14_basin_count'),
            ('meaningful_stall_age','Meaningful stagnation age','15_stagnation_history'),
            ('local_source_credit','Local source credit','16_local_source_credit'),
            ('global_source_credit','Global source credit','17_global_source_credit'),
            ('v19_hard_resets','Cumulative hard resets','18_hard_reset_activity')]
        for col,ylab,stem in plot_specs:
            if col in cycle_df.columns:
                fig,ax=plt.subplots(figsize=(10,6)); ax.plot(cycle_df['true_fe'],cycle_df[col]); ax.set_xlabel('TRUE function evaluations'); ax.set_ylabel(ylab); ax.set_title(f'{ylab} | CEC-2017 F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); _save_audit_fig(fig,stem)
        src_cols=[c for c in ['true_BWR','true_BMR','true_INCUMBENT','true_POLL','true_SURR_LOCAL','true_QUAD_TR','true_ANISO','true_PRECISION','true_NOVELTY','true_COVERAGE'] if c in cycle_df.columns]
        if src_cols:
            fig,ax=plt.subplots(figsize=(11,6));
            for c in src_cols: ax.plot(cycle_df['true_fe'],cycle_df[c].cumsum(),label=c.replace('true_',''))
            ax.set_xlabel('TRUE function evaluations'); ax.set_ylabel('Cumulative TRUE evaluations'); ax.set_title(f'TRUE-FE Allocation by Source | F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); ax.legend(ncol=2,fontsize=15); _save_audit_fig(fig,'19_true_fe_allocation_by_source')
        prod_cols=[c for c in ['prod_BWR','prod_BMR','prod_INC','prod_POLL','prod_SR','prod_Q','prod_ANISO','prod_PRECISION','prod_NOV'] if c in cycle_df.columns]
        if prod_cols:
            fig,ax=plt.subplots(figsize=(11,6));
            for c in prod_cols: ax.plot(cycle_df['true_fe'],cycle_df[c],label=c.replace('prod_',''))
            ax.set_xlabel('TRUE function evaluations'); ax.set_ylabel('Verified productivity'); ax.set_title(f'Source Productivity | F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); ax.legend(ncol=2,fontsize=15); _save_audit_fig(fig,'20_source_productivity')
    sedf=pd.DataFrame(selected_eval_log)
    if len(sedf):
        fig,ax=plt.subplots(figsize=(7,7)); ax.scatter(sedf['predicted_f'],sedf['true_f'],s=18,alpha=.7); lo=min(sedf['predicted_f'].min(),sedf['true_f'].min()); hi=max(sedf['predicted_f'].max(),sedf['true_f'].max()); ax.plot([lo,hi],[lo,hi],'--',linewidth=1); ax.set_xlabel('Predicted objective'); ax.set_ylabel('TRUE objective'); ax.set_title(f'Predicted vs TRUE | F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); _save_audit_fig(fig,'21_predicted_vs_true')
        fig,ax=plt.subplots(figsize=(10,6)); ax.plot(sedf['true_fe'],sedf['abs_prediction_error']); ax.set_yscale('log'); ax.set_xlabel('TRUE function evaluations'); ax.set_ylabel('Absolute prediction error'); ax.set_title(f'Prediction Error History | F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); _save_audit_fig(fig,'22_prediction_error_history')
    bdf=pd.DataFrame(basin_state_log)
    if len(bdf):
        fig,ax=plt.subplots(figsize=(10,6));
        for bid,g in bdf.groupby('basin_idx'): ax.plot(g['true_fe'],g['radius'],label=f'Basin {bid}')
        ax.set_xlabel('TRUE function evaluations'); ax.set_ylabel('Trust-region radius'); ax.set_title(f'Per-Basin Radius Evolution | F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); ax.legend(); _save_audit_fig(fig,'23_per_basin_radius')
    adf=pd.DataFrame(exact_selection_audit_log.winners)
    if len(adf):
        win=adf[adf['selected_at_this_stage']==True]
        if len(win):
            fig,ax=plt.subplots(figsize=(10,6)); ax.plot(np.arange(1,len(win)+1),win['total_selection_score']); ax.set_xlabel('Sequential selection stage'); ax.set_ylabel('Winning acquisition score'); ax.set_title(f'Exact Winning Selection Score | F{function_id} | D={dimension}'); ax.grid(True,alpha=.25); _save_audit_fig(fig,'24_exact_selection_score_history')

    if np.isfinite(optimum):
        error_hist = np.maximum(np.asarray(best_history) - optimum, 1e-16)
    else:
        error_hist = np.maximum(np.abs(np.asarray(best_history)), 1e-16)
    plt.figure(figsize=(10, 6))
    plt.semilogy(np.arange(1, len(error_hist) + 1), error_hist)
    plt.xlabel('True Function Evaluations')
    plt.ylabel('Best-so-far error')
    plt.title(f'MB-DTASR-PFSA | CEC-2017 F{function_id} | D={dimension}')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plot_path = os.path.join(result_dir, f'F{function_id}_D{dimension}_{run_suffix}seed{seed}_convergence.png')
    plt.savefig(plot_path, dpi=450, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print()
    print('=' * 112)
    print('FINAL RESULT')
    print('=' * 112)
    print(f'CEC function          : F{function_id}')
    print(f'Dimension             : {dimension}')
    print(f'TRUE evaluations      : {problem.true_fe}')
    print(f'Final TRUE best       : {best_f:.15e}')
    if np.isfinite(optimum):
        print(f'Backend f_global      : {optimum:.15e}')
        print(f'POST-RUN error        : {error:.15e}')
    else:
        print('Backend f_global      : UNAVAILABLE')
        print('POST-RUN error        : NaN (reference unavailable)')
    print(f'Reference source      : {problem.reference_source}')
    print(f'Runtime               : {runtime:.3f} s')
    print(f'Non-finite backend FEs: {problem.nonfinite_count}')
    print('\nBest x:')
    print(np.array2string(best_x, precision=10, suppress_small=False))
    print('=' * 112)
    print('\nKnown optimum was used ONLY after optimization to calculate the reporting error.')
    return {'method': 'MB-DTASR-PFSA V19', 'benchmark_metadata': problem.metadata(), 'best_x': best_x, 'best_f': best_f, 'error': error, 'true_fes': problem.true_fe, 'runtime': runtime, 'nonfinite_count': int(problem.nonfinite_count), 'archive': archive_df, 'cycles': cycle_df, 'best_history': np.asarray(best_history), 'trajectory_sha256': trajectory_sha256, 'detailed': True}



def _atomic_csv_write(df, path):
    """Write CSV atomically so Windows Explorer never sees a half-written summary."""
    tmp = str(path) + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _safe_research_plot(fig, run_dir, stem):
    png = os.path.join(run_dir, stem + '.png')
    pdf = os.path.join(run_dir, stem + '.pdf')
    fig.tight_layout()
    fig.savefig(png, dpi=450, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return [png, pdf]


def _expanded_run_research_visualizations(out, run_dir, function_id, dimension, run_id, seed):
    """Reporting-only research figures derived from already-recorded run histories."""
    artifacts=[]
    prefix=f'F{int(function_id)}_D{int(dimension)}_run{int(run_id):02d}_seed{int(seed)}'
    cyc=out.get('cycles', pd.DataFrame()).copy()
    arc=out.get('archive', pd.DataFrame()).copy()
    hist=np.asarray(out.get('best_history', []), dtype=float)
    ref=float(out['benchmark_metadata'].get('reference_optimum', np.nan))
    def save(fig, name): artifacts.extend(_safe_research_plot(fig, run_dir, prefix+'_'+name))
    if len(cyc):
        x=cyc['true_fe'].to_numpy(float)
        wcols=['weight_productivity','weight_prediction','weight_novelty','weight_diversity']
        if all(c in cyc for c in wcols):
            bal=pd.DataFrame({'cycle':cyc['cycle'],'true_fe':cyc['true_fe'],
                'exploration_index':cyc['weight_novelty']+cyc['weight_diversity'],
                'exploitation_index':cyc['weight_productivity']+cyc['weight_prediction']})
            bal['difference_explore_minus_exploit']=bal['exploration_index']-bal['exploitation_index']
            p=os.path.join(run_dir,prefix+'_25_exploration_exploitation.csv'); bal.to_csv(p,index=False); artifacts.append(p)
            fig,ax=plt.subplots(figsize=(10,6)); ax.plot(x,bal['exploration_index'],label='Exploration'); ax.plot(x,bal['exploitation_index'],label='Exploitation'); ax.axhline(.5,ls='--',lw=1); ax.set(xlabel='TRUE function evaluations',ylabel='Combined adaptive weight',title='Exploration–Exploitation Balance'); ax.grid(True,alpha=.25); ax.legend(); save(fig,'25_exploration_exploitation')
            fig,ax=plt.subplots(figsize=(10,6)); ax.stackplot(x,*[cyc[c] for c in wcols],labels=['Productivity','Prediction','Novelty','Diversity']); ax.set(xlabel='TRUE function evaluations',ylabel='Weight share',title='Adaptive Acquisition Weight Composition'); ax.set_ylim(0,1); ax.legend(loc='upper right'); save(fig,'26_adaptive_weight_stack')
        if 'state' in cyc:
            st=pd.crosstab(cyc['true_fe'],cyc['state']).reset_index(); p=os.path.join(run_dir,prefix+'_27_search_state_onehot.csv'); st.to_csv(p,index=False); artifacts.append(p)
            cats=list(pd.unique(cyc['state'].astype(str))); mp={v:i for i,v in enumerate(cats)}
            fig,ax=plt.subplots(figsize=(11,4)); ax.step(x,[mp[str(v)] for v in cyc['state']],where='post'); ax.set_yticks(range(len(cats)),cats); ax.set(xlabel='TRUE function evaluations',ylabel='Search state',title='Search-State Timeline'); ax.grid(True,alpha=.25); save(fig,'27_search_state_timeline')
            cnt=cyc['state'].value_counts().rename_axis('state').reset_index(name='cycles'); cnt['percent']=100*cnt['cycles']/cnt['cycles'].sum(); p=os.path.join(run_dir,prefix+'_28_search_state_usage.csv'); cnt.to_csv(p,index=False); artifacts.append(p)
            fig,ax=plt.subplots(figsize=(9,5)); ax.bar(cnt['state'],cnt['cycles']); ax.set(xlabel='Search state',ylabel='Cycles',title='Search-State Usage'); ax.tick_params(axis='x',rotation=30); ax.grid(True,axis='y',alpha=.25); save(fig,'28_search_state_usage')
        flag_cols=[c for c in ['micro_local_flag','anisotropic_flag','precision_flag','intensify_flag','local_success_active','precision_retry_flag'] if c in cyc]
        if flag_cols:
            fd=cyc[['cycle','true_fe']+flag_cols].copy(); p=os.path.join(run_dir,prefix+'_29_mechanism_activation.csv'); fd.to_csv(p,index=False); artifacts.append(p)
            fig,ax=plt.subplots(figsize=(11,6));
            for i,c in enumerate(flag_cols): ax.step(x,fd[c].astype(int)+i*1.15,where='post',label=c)
            ax.set(xlabel='TRUE function evaluations',ylabel='Activation (offset for visibility)',title='Mechanism Activation Timeline'); ax.grid(True,alpha=.2); ax.legend(ncol=2,fontsize=15); save(fig,'29_mechanism_activation')
        rec_cols=[c for c in ['recovery_remaining','deep_recovery_remaining','escape_remaining','saturated_basins','quarantined_basins','v18_regenerated_basins','v19_failed_recovery_episodes','v19_hard_resets'] if c in cyc]
        if rec_cols:
            rd=cyc[['cycle','true_fe']+rec_cols].copy(); p=os.path.join(run_dir,prefix+'_30_recovery_escape_quarantine.csv'); rd.to_csv(p,index=False); artifacts.append(p)
            fig,ax=plt.subplots(figsize=(11,6));
            for c in rec_cols: ax.plot(x,rd[c],label=c)
            ax.set(xlabel='TRUE function evaluations',ylabel='Count / remaining cycles',title='Recovery, Escape, Saturation and Quarantine'); ax.grid(True,alpha=.25); ax.legend(ncol=2,fontsize=15); save(fig,'30_recovery_escape_quarantine')
        trcols=[c for c in ['min_tr_radius','mean_tr_radius','max_tr_radius','precision_memory_radius'] if c in cyc]
        if trcols:
            td=cyc[['cycle','true_fe']+trcols].copy(); p=os.path.join(run_dir,prefix+'_31_trust_region_envelope.csv'); td.to_csv(p,index=False); artifacts.append(p)
            fig,ax=plt.subplots(figsize=(10,6));
            for c in trcols: ax.plot(x,td[c],label=c)
            ax.set(xlabel='TRUE function evaluations',ylabel='Radius',title='Trust-Region Radius Envelope'); ax.set_yscale('log'); ax.grid(True,alpha=.25); ax.legend(); save(fig,'31_trust_region_envelope')
        suc=[c for c in ['aniso_success_ratio','precision_success_ratio','local_source_credit','global_source_credit','incumbent_basin_credit'] if c in cyc]
        if suc:
            sd=cyc[['cycle','true_fe']+suc].copy(); p=os.path.join(run_dir,prefix+'_32_success_credit_evolution.csv'); sd.to_csv(p,index=False); artifacts.append(p)
            fig,ax=plt.subplots(figsize=(10,6));
            for c in suc: ax.plot(x,sd[c],label=c)
            ax.set(xlabel='TRUE function evaluations',ylabel='Rate / credit',title='Success Ratios and Source Credits'); ax.grid(True,alpha=.25); ax.legend(fontsize=15); save(fig,'32_success_credit_evolution')
    # complete TRUE-evaluation trajectory
    if len(arc):
        a=arc.copy(); a.insert(0,'true_fe',np.arange(1,len(a)+1)); a['best_so_far_f']=a['f'].cummin(); a['is_new_best']=a['f'].le(a['best_so_far_f']);
        p=os.path.join(run_dir,prefix+'_33_all_true_evaluations_plot_data.csv'); a.to_csv(p,index=False); artifacts.append(p)
        fig,ax=plt.subplots(figsize=(11,6)); ax.scatter(a['true_fe'],a['f'],s=8,alpha=.35,label='Every TRUE evaluation'); ax.plot(a['true_fe'],a['best_so_far_f'],lw=2,label='Best-so-far'); ax.set(xlabel='TRUE function evaluations',ylabel='Objective',title='Every TRUE Evaluation and Best-so-Far'); ax.grid(True,alpha=.25); ax.legend(); save(fig,'33_every_true_evaluation')
        xcols=[c for c in a if c.startswith('x') and c[1:].isdigit()]
        if xcols:
            # trajectory heat map, normalized to [-100,100] CEC domain for comparability
            vals=a[xcols].to_numpy(float).T
            fig,ax=plt.subplots(figsize=(12,max(4,min(9,.35*len(xcols))))); im=ax.imshow(vals,aspect='auto',origin='lower',extent=[1,len(a),1,len(xcols)]); ax.set(xlabel='TRUE function evaluation',ylabel='Decision-variable index',title='Decision-Variable Trajectory Heat Map'); fig.colorbar(im,ax=ax,label='Decision-variable value'); save(fig,'34_design_variable_heatmap')
            if len(xcols)<=20:
                fig,ax=plt.subplots(figsize=(11,6));
                for c in xcols: ax.plot(a['true_fe'],a[c],lw=.8,label=c)
                ax.set(xlabel='TRUE function evaluations',ylabel='Decision-variable value',title='Decision-Variable Trajectories'); ax.grid(True,alpha=.2); ax.legend(ncol=4,fontsize=15); save(fig,'35_design_variable_trajectories')
    # Raw audit CSVs generated by optimizer: visualize additional step-level information.
    def read_suffix(suffix):
        fs=list(Path(run_dir).glob('*'+suffix));
        try: return pd.read_csv(fs[0]) if fs else pd.DataFrame()
        except Exception: return pd.DataFrame()
    doe=read_suffix('_01_initial_DOE_stepwise.csv')
    if len(doe):
        fig,ax=plt.subplots(figsize=(10,6)); ax.scatter(doe['true_fe'],doe['true_f'],s=16,alpha=.65,label='DOE objective'); ax.plot(doe['true_fe'],doe['best_so_far_f'],lw=2,label='DOE best-so-far'); ax.set(xlabel='DOE TRUE evaluation',ylabel='Objective',title='Initial DOE Step-by-Step'); ax.grid(True,alpha=.25); ax.legend(); save(fig,'36_initial_DOE_stepwise')
    rbf=read_suffix('_02_RBF_fit_diagnostics.csv')
    if len(rbf):
        fig,ax=plt.subplots(figsize=(10,6)); ax.plot(rbf['true_fe'],rbf['sigma'],label='RBF sigma'); ax.set(xlabel='TRUE function evaluations',ylabel='Gaussian RBF sigma',title='RBF Kernel-Width Evolution'); ax.grid(True,alpha=.25); ax.legend(); save(fig,'37_RBF_sigma')
        fig,ax=plt.subplots(figsize=(10,6)); ax.plot(rbf['true_fe'],rbf['n_archive'],label='Archive size'); ax.plot(rbf['true_fe'],rbf['n_training'],label='Training size'); ax.set(xlabel='TRUE function evaluations',ylabel='Points',title='RBF Training-Set Evolution'); ax.grid(True,alpha=.25); ax.legend(); save(fig,'38_RBF_training_size')
    candidate_files=list(Path(run_dir).glob('*_04_all_generated_candidates.csv'))
    parts=[]; samples=[]; sampled=0
    if candidate_files:
        for chunk in pd.read_csv(candidate_files[0], chunksize=10000):
            parts.append(chunk.groupby(['cycle','true_fe_before_batch','source'],as_index=False).agg(
                generated=('candidate_id','size'), selected=('selected_for_true_eval','sum'),
                predicted_sum=('predicted_f','sum'), novelty_sum=('novelty_nc','sum')))
            # Bound the scatter independently of raw log size; selection uses no optimizer RNG.
            if sampled < 10000:
                sample=chunk.iloc[::100].head(10000-sampled)
                samples.append(sample); sampled += len(sample)
        cs=pd.concat(parts,ignore_index=True).groupby(['cycle','true_fe_before_batch','source'],as_index=False).sum()
        cs['mean_predicted_f']=cs.pop('predicted_sum')/cs['generated']
        cs['mean_novelty']=cs.pop('novelty_sum')/cs['generated']
        p=os.path.join(run_dir,prefix+'_39_candidate_generation_summary.csv'); cs.to_csv(p,index=False); artifacts.append(p)
        piv=cs.pivot_table(index='true_fe_before_batch',columns='source',values='generated',aggfunc='sum',fill_value=0)
        fig,ax=plt.subplots(figsize=(11,6))
        for c in piv.columns: ax.plot(piv.index,piv[c],label=str(c))
        ax.set(xlabel='TRUE function evaluations before batch',ylabel='Generated candidates',title='Candidate Generation by Source')
        ax.grid(True,alpha=.25); ax.legend(ncol=3,fontsize=15); save(fig,'39_candidate_generation_by_source')
        cand=pd.concat(samples,ignore_index=True)
        fig,ax=plt.subplots(figsize=(10,6)); ax.scatter(cand['novelty_nc'],cand['predicted_f'],c=cand['selected_for_true_eval'].astype(int),s=8,alpha=.35)
        ax.set(xlabel='Novelty',ylabel='Predicted objective',title='Candidate Fitness vs Novelty (bounded sample)')
        ax.grid(True,alpha=.2); save(fig,'40_candidate_fitness_novelty')
        del cand, parts, samples
    selection_files=list(Path(run_dir).glob('*_05_exact_sequential_selection_scores.csv'))
    winners=[]
    if selection_files:
        for chunk in pd.read_csv(selection_files[0],chunksize=10000):
            winners.append(chunk.loc[chunk['selected_at_this_stage'] == True].copy())
    sel=pd.concat(winners,ignore_index=True) if winners else pd.DataFrame()
    if len(sel):
        win=sel[sel['selected_at_this_stage'].astype(bool)].copy()
        if len(win):
            cols=[c for c in ['weighted_prediction','weighted_novelty','weighted_diversity','source_score','restart_bonus'] if c in win]
            fig,ax=plt.subplots(figsize=(11,6));
            for c in cols: ax.plot(np.arange(1,len(win)+1),win[c],label=c)
            ax.set(xlabel='Sequential winning selection',ylabel='Score contribution',title='Winning Acquisition-Score Components'); ax.grid(True,alpha=.25); ax.legend(fontsize=15); save(fig,'41_winning_score_components')
            fig,ax=plt.subplots(figsize=(10,6)); ax.plot(np.arange(1,len(win)+1),win['pool_size']); ax.set(xlabel='Sequential winning selection',ylabel='Candidate pool size',title='Selection Pool Size'); ax.grid(True,alpha=.25); save(fig,'42_selection_pool_size')
    bdf=read_suffix('_03_basin_state_stepwise.csv')
    if len(bdf):
        for col,stem,ylab in [('best_y','43_per_basin_best','Basin best objective'),('failure_count','44_per_basin_failure_count','Failure count'),('true_bmr_evals','45_per_basin_BMR_evaluations','Cumulative BMR TRUE evaluations'),('true_bmr_successes','46_per_basin_BMR_successes','Cumulative BMR successes')]:
            if col in bdf:
                fig,ax=plt.subplots(figsize=(11,6));
                for bid,g in bdf.groupby('basin_idx'): ax.plot(g['true_fe'],g[col],label=f'Basin {bid}')
                ax.set(xlabel='TRUE function evaluations',ylabel=ylab,title=ylab+' by Basin'); ax.grid(True,alpha=.25); ax.legend(fontsize=15); save(fig,stem)
    rel=read_suffix('_07_reliability_update_stepwise.csv')
    if len(rel):
        fig,ax=plt.subplots(figsize=(10,6)); ax.plot(rel['true_fe'],rel['reliability_after_update']); ax.set(xlabel='TRUE function evaluations',ylabel='Reliability',title='Surrogate Reliability After Every Verified Update'); ax.set_ylim(0,1); ax.grid(True,alpha=.25); save(fig,'47_reliability_every_update')
    # manuscript dashboard
    if len(cyc):
        fig,axs=plt.subplots(3,2,figsize=(14,14)); axs=axs.ravel();
        axs[0].plot(cyc['true_fe'],cyc['best_true_f']); axs[0].set_title('Best objective')
        if 'reliability_after' in cyc: axs[1].plot(cyc['true_fe'],cyc['reliability_after'],label='Reliability')
        if 'v19_global_confidence' in cyc: axs[1].plot(cyc['true_fe'],cyc['v19_global_confidence'],label='Global confidence'); axs[1].legend()
        if all(c in cyc for c in ['weight_productivity','weight_prediction','weight_novelty','weight_diversity']):
            for c in ['weight_productivity','weight_prediction','weight_novelty','weight_diversity']: axs[2].plot(cyc['true_fe'],cyc[c],label=c.replace('weight_','')); axs[2].legend(fontsize=15)
        if 'mean_tr_radius' in cyc: axs[3].plot(cyc['true_fe'],cyc['mean_tr_radius'])
        if 'meaningful_stall_age' in cyc: axs[4].plot(cyc['true_fe'],cyc['meaningful_stall_age'])
        if len(arc): axs[5].semilogy(np.arange(1,len(hist)+1),np.maximum(hist-ref,1e-16) if np.isfinite(ref) else np.maximum(np.abs(hist),1e-16))
        titles=['Best objective','Reliability / confidence','Adaptive weights','Mean TR radius','Meaningful stagnation','Best-so-far error']
        for ax,t in zip(axs,titles): ax.set_title(t); ax.grid(True,alpha=.2); ax.set_xlabel('TRUE FE')
        save(fig,'48_COMPLETE_RESEARCH_DASHBOARD')
    return artifacts


def _finalize_run_package(out, run_dir, function_id, dimension, run_id, seed):
    """Create a self-contained seed package immediately after the seed finishes."""
    os.makedirs(run_dir, exist_ok=True)
    row={'function':int(function_id),'dimension':int(dimension),'run':int(run_id),'seed':int(seed),
         'reference_optimum':out['benchmark_metadata']['reference_optimum'],'best_f':out['best_f'],'error':out['error'],
         'abs_error':abs(out['error']) if np.isfinite(out['error']) else np.nan,'true_fes':out['true_fes'],
         'runtime_s':out['runtime'],'nonfinite_count':out['nonfinite_count'], 'trajectory_sha256':out['trajectory_sha256']}
    for j,v in enumerate(np.asarray(out.get('best_x',[])).reshape(-1)): row[f'x{j+1}']=float(v)
    _atomic_csv_write(pd.DataFrame([row]),os.path.join(run_dir,'RUN_FINAL_SUMMARY.csv'))
    if out.get('best_x') is not None:
        _atomic_csv_write(pd.DataFrame([np.asarray(out['best_x']).reshape(-1)],columns=[f'x{i+1}' for i in range(dimension)]),os.path.join(run_dir,'RUN_BEST_SOLUTION.csv'))
    if out.get("detailed", False):
        _expanded_run_research_visualizations(out,run_dir,function_id,dimension,run_id,seed)
    files=sorted([p for p in Path(run_dir).rglob('*') if p.is_file()])
    inv=pd.DataFrame([{'artifact':p.name,'relative_path':str(p.relative_to(run_dir)),'extension':p.suffix.lower(),'size_bytes':p.stat().st_size} for p in files])
    _atomic_csv_write(inv,os.path.join(run_dir,'RUN_ALL_ARTIFACTS_INDEX.csv'))
    _atomic_csv_write(inv[inv['extension']=='.csv'].copy(),os.path.join(run_dir,'RUN_ALL_CSV_FILES_INDEX.csv'))
    _atomic_csv_write(inv[inv['extension'].isin(['.png','.pdf'])].copy(),os.path.join(run_dir,'RUN_ALL_FIGURES_INDEX.csv'))
    with open(os.path.join(run_dir,'RUN_COMPLETE.txt'),'w',encoding='utf-8') as fh:
        fh.write(f'COMPLETED\\nFunction=F{function_id}\\nDimension={dimension}\\nRun={run_id}\\nSeed={seed}\\nBest_f={out["best_f"]:.17e}\\nError={out["error"]:.17e}\\nTRUE_FEs={out["true_fes"]}\\nRuntime_s={out["runtime"]:.6f}\\n')
    return row


def _save_completed_so_far(rows, function_dir, function_id, dimension):
    if not rows: return
    df=pd.DataFrame(rows).sort_values('run').reset_index(drop=True)
    _atomic_csv_write(df,os.path.join(function_dir,'COMPLETED_RUNS_CHECKPOINT.csv'))
    _atomic_csv_write(df,os.path.join(function_dir,'RUNS_COMPLETED_SO_FAR.csv'))
    _atomic_csv_write(df,os.path.join(function_dir,f'F{function_id}_D{dimension}_{len(df)}runs_COMPLETED_SO_FAR.csv'))
    # Reuse full statistical routine on completed subset, then retain clearly named live copies.
    _function_30run_statistics(df,function_id,dimension,function_dir)
    for src,dst in [('30run_statistical_summary.csv','CUMULATIVE_STATISTICAL_SUMMARY_SO_FAR.csv'),('30run_metric_statistics.csv','CUMULATIVE_METRIC_STATISTICS_SO_FAR.csv')]:
        p=os.path.join(function_dir,src)
        if os.path.isfile(p):
            pd.read_csv(p).to_csv(os.path.join(function_dir,dst),index=False)
    last=df.iloc[-1]
    with open(os.path.join(function_dir,'LATEST_COMPLETED_RUN.txt'),'w',encoding='utf-8') as fh:
        fh.write(f'run={int(last.run)}\\nseed={int(last.seed)}\\nbest_f={float(last.best_f):.17e}\\nfolder=RUN_{int(last.run):02d}_SEED_{int(last.seed)}\\n')

def _run_one_parallel_job(function_id, dimension, run_id, seed, max_true_fes, verbose, function_dir):
    """Windows/Spyder-safe worker; every seed writes into its own folder immediately."""
    run_dir=os.path.join(function_dir,f'RUN_{int(run_id):02d}_SEED_{int(seed)}')
    os.makedirs(run_dir,exist_ok=True)
    pd.DataFrame([{'status':'RUNNING','function':int(function_id),'dimension':int(dimension),'run':int(run_id),'seed':int(seed)}]).to_csv(os.path.join(run_dir,'RUN_STATUS.csv'),index=False)
    print('\n'+'#'*112,flush=True)
    print(f'[WORKER] CEC-2017 F{function_id} | D={dimension} | RUN {run_id}/{RUNS_PER_FUNCTION} | seed={seed}',flush=True)
    print(f'[RUN DIRECTORY] {os.path.abspath(run_dir)}',flush=True)
    print('#'*112,flush=True)
    log_prefix=f'[F{int(function_id):02d} RUN {int(run_id):02d} SEED {int(seed)}] '
    try:
        out=mb_dtaa_pfsa_v19(function_id=function_id,dimension=dimension,seed=seed,max_true_fes=max_true_fes,verbose=verbose,result_dir=run_dir,run_id=run_id,show_plot=False,log_prefix=log_prefix)
        row=_finalize_run_package(out,run_dir,function_id,dimension,run_id,seed)
        row.update({'backend_class':out['benchmark_metadata']['backend_class'],'problem_name':out['benchmark_metadata']['problem_name'],'reference_source':out['benchmark_metadata']['reference_source'],'run_dir':os.path.abspath(run_dir)})
        pd.DataFrame([{'status':'COMPLETED','function':int(function_id),'dimension':int(dimension),'run':int(run_id),'seed':int(seed),'best_f':out['best_f'],'error':out['error'],'true_fes':out['true_fes'],'runtime_s':out['runtime']}]).to_csv(os.path.join(run_dir,'RUN_STATUS.csv'),index=False)
        print(f'{log_prefix}RUN FINISHED AND SAVED | {os.path.abspath(run_dir)}',flush=True)
        return row
    except Exception as exc:
        pd.DataFrame([{'status':'FAILED','function':int(function_id),'dimension':int(dimension),'run':int(run_id),'seed':int(seed),'error_message':repr(exc)}]).to_csv(os.path.join(run_dir,'RUN_STATUS.csv'),index=False)
        raise


def _function_30run_statistics(df, function_id, dimension, function_dir):
    """Save a function-local descriptive statistical summary of the 30 runs."""
    bestf = df['best_f'].to_numpy(dtype=float)
    err = df['error'].to_numpy(dtype=float)
    rt = df['runtime_s'].to_numpy(dtype=float)
    fes = df['true_fes'].to_numpy(dtype=float)
    abs_err = np.abs(err)

    finite_err = err[np.isfinite(err)]
    if len(finite_err) >= 3 and not np.allclose(finite_err, finite_err[0]):
        try:
            shapiro_w, shapiro_p = shapiro(finite_err)
        except Exception:
            shapiro_w, shapiro_p = np.nan, np.nan
    else:
        shapiro_w, shapiro_p = np.nan, np.nan

    row = {
        'function': int(function_id),
        'dimension': int(dimension),
        'runs': int(len(df)),
        'seed_start': int(RUN_SEED_START),
        'seed_end': int(RUN_SEED_START + len(df) - 1),
        'parallel_workers': int(PARALLEL_WORKERS),
        'reference_optimum': float(df['reference_optimum'].iloc[0]),
        'best_f_min': float(np.nanmin(bestf)),
        'best_f_mean': float(np.nanmean(bestf)),
        'best_f_median': float(np.nanmedian(bestf)),
        'best_f_std': float(np.nanstd(bestf, ddof=1)) if len(bestf) > 1 else 0.0,
        'best_f_max': float(np.nanmax(bestf)),
        'error_min': float(np.nanmin(err)),
        'error_mean': float(np.nanmean(err)),
        'error_median': float(np.nanmedian(err)),
        'error_std': float(np.nanstd(err, ddof=1)) if len(err) > 1 else 0.0,
        'error_max': float(np.nanmax(err)),
        'abs_error_mean': float(np.nanmean(abs_err)),
        'abs_error_median': float(np.nanmedian(abs_err)),
        'abs_error_std': float(np.nanstd(abs_err, ddof=1)) if len(abs_err) > 1 else 0.0,
        'q1_error': float(np.nanpercentile(err, 25)),
        'q3_error': float(np.nanpercentile(err, 75)),
        'iqr_error': float(np.nanpercentile(err, 75) - np.nanpercentile(err, 25)),
        'mean_true_fes': float(np.nanmean(fes)),
        'std_true_fes': float(np.nanstd(fes, ddof=1)) if len(fes) > 1 else 0.0,
        'mean_runtime_s': float(np.nanmean(rt)),
        'median_runtime_s': float(np.nanmedian(rt)),
        'std_runtime_s': float(np.nanstd(rt, ddof=1)) if len(rt) > 1 else 0.0,
        'success_tolerance': float(PUBLICATION_SUCCESS_TOL),
        'success_count': int(np.sum(np.isfinite(err) & (np.abs(err) <= PUBLICATION_SUCCESS_TOL))),
        'success_rate_percent': float(100.0 * np.mean(np.isfinite(err) & (np.abs(err) <= PUBLICATION_SUCCESS_TOL))),
        'shapiro_W_error': float(shapiro_w),
        'shapiro_p_error': float(shapiro_p),
    }
    stats_df = pd.DataFrame([row])
    stats_df.to_csv(os.path.join(function_dir, '30run_statistical_summary.csv'), index=False)

    metric_rows = []
    for name, values in [('best_f', bestf), ('error', err), ('abs_error', abs_err), ('true_fes', fes), ('runtime_s', rt)]:
        values = np.asarray(values, dtype=float)
        metric_rows.append({
            'metric': name,
            'n': int(np.sum(np.isfinite(values))),
            'min': float(np.nanmin(values)),
            'mean': float(np.nanmean(values)),
            'median': float(np.nanmedian(values)),
            'std': float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            'q1': float(np.nanpercentile(values, 25)),
            'q3': float(np.nanpercentile(values, 75)),
            'max': float(np.nanmax(values)),
        })
    pd.DataFrame(metric_rows).to_csv(os.path.join(function_dir, '30run_metric_statistics.csv'), index=False)
    return stats_df


def _best_run_report(df, function_dir, function_id, dimension, max_true_fes):
    # Ties use the earliest run. Replay is reporting work, never another trial.
    best = df.sort_values(['best_f', 'run'], kind='stable').iloc[0]
    folder = os.path.join(function_dir, 'BEST_RUN_DETAILED')
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame([best]).to_csv(os.path.join(folder, 'SELECTED_ORIGINAL_RUN.csv'), index=False)
    status_path = os.path.join(folder, 'REPLAY_STATUS.csv')
    pd.DataFrame([{'status':'RUNNING', 'seed':int(best.seed)}]).to_csv(status_path,index=False)
    try:
        out = mb_dtaa_pfsa_v19(function_id=function_id, dimension=dimension, seed=int(best.seed),
            max_true_fes=max_true_fes, verbose=False, result_dir=folder, run_id=int(best.run), detailed=True)
        matches = out['trajectory_sha256'] == best.trajectory_sha256
        pd.DataFrame([{'status':'VERIFIED' if matches else 'MISMATCH',
                       'seed':int(best.seed), 'original_best_f':best.best_f, 'replay_best_f':out['best_f'],
                       'original_sha256':best.trajectory_sha256, 'replay_sha256':out['trajectory_sha256'],
                       'replay_true_fes':out['true_fes'], 'replay_runtime_s':out['runtime'],
                       'included_in_statistics':False}]).to_csv(status_path,index=False)
        if not matches:
            raise RuntimeError('Best-seed replay trajectory differs; detailed outputs are NOT verified. Original statistics are unchanged.')
        _finalize_run_package(out, folder, function_id, dimension, int(best.run), int(best.seed))
        print(f'[BEST RUN VERIFIED] seed={int(best.seed)} | {os.path.abspath(folder)}',flush=True)
    except Exception as exc:
        status = pd.read_csv(status_path)
        status['status'] = 'FAILED'
        status['error_message'] = str(exc)
        status.to_csv(status_path,index=False)
        with open(os.path.join(folder, 'REPORT_ERROR.txt'),'w',encoding='utf-8') as fh:
            fh.write(str(exc))
        print(f'[BEST REPORT FAILED; original results retained] {exc}',flush=True)


def _essential_function_reports(df, folder, function_id, dimension):
    outputs=[]
    for row in df.itertuples():
        conv=pd.read_csv(os.path.join(row.run_dir, 'convergence.csv'))
        outputs.append({'best_history':conv['best_so_far_f'].to_numpy()})
    _publication_convergence(outputs, float(df.iloc[0].reference_optimum), folder, function_id, dimension)
    fig, ax = plt.subplots(figsize=(10,6))
    ref = float(df.iloc[0].reference_optimum)
    for row, out in zip(df.itertuples(), outputs):
        h = out['best_history']
        err = np.maximum(h-ref,1e-16) if np.isfinite(ref) else np.maximum(np.abs(h),1e-16)
        ax.semilogy(np.arange(1,len(h)+1),err,alpha=.55,linewidth=.8,label=f'Seed {row.seed}')
    ax.set(xlabel='TRUE function evaluations',ylabel='Best-so-far error',title=f'All {len(df)} Independent Runs | F{function_id} | D={dimension}')
    ax.grid(True,alpha=.25)
    _publication_save_figure(fig,folder,'plot_all_run_convergence')
    _publication_distribution_plots(df,folder,function_id,dimension)
    _summarize_final_runs(df,function_id,dimension,folder)
    _write_comparison_templates(folder)
    for kind, configured in [('comparator', PUBLICATION_COMPARATOR_CSV), ('ablation', PUBLICATION_ABLATION_CSV)]:
        other=_load_optional_method_data(configured, kind+'_results.csv', folder)
        _pairwise_method_statistics(df,other,folder,kind)


def run_function_multiple(function_id, dimension=DIM, n_runs=RUNS_PER_FUNCTION, master_seed=MASTER_SEED,
                          max_true_fes=MAX_TRUE_FES, verbose=True, parallel_workers=PARALLEL_WORKERS):
    """
    Run one CEC function n_runs times using consecutive independent seeds.
    For the default 30 runs, the seed schedule is 42, 43, ..., 71.
    Runs are executed with up to 3 parallel worker processes and all artifacts are
    saved in a separate function-wise folder.
    """
    function_dir = os.path.join(RESULT_DIR, 'BEST_SEED_REPORTING', f'F{int(function_id):02d}_D{int(dimension)}')
    if os.path.isdir(function_dir) and os.listdir(function_dir):
        # Preserve previous experiments and prevent stale detailed artifacts.
        function_dir = os.path.join(function_dir, 'SESSION_' + str(time.time_ns()))
    os.makedirs(function_dir, exist_ok=True)

    seed_start = int(master_seed)
    rows = []
    failures = []
    n_workers = max(1, min(int(parallel_workers), int(n_runs)))

    print('\n' + '=' * 112)
    print(f'F{function_id} | D={dimension} | {n_runs} RUNS | SEEDS={seed_start}..{seed_start + int(n_runs) - 1} | PARALLEL WORKERS={n_workers}')
    print(f'Function result folder: {os.path.abspath(function_dir)}')
    print(f'[WINDOWS EXPLORER] Copy/paste this path: {os.path.abspath(function_dir)}')
    print('[SAVE MODE] Every completed seed is finalized immediately; cumulative CSV/statistics refresh after each seed.')
    print('=' * 112)

    if n_workers == 1:
        for run_id in range(1, int(n_runs) + 1):
            seed = seed_start + run_id - 1
            try:
                row = _run_one_parallel_job(function_id, dimension, run_id, seed,
                                            max_true_fes, verbose, function_dir)
                rows.append(row)
                _save_completed_so_far(rows, function_dir, function_id, dimension)
                print(f'[VISIBLE NOW] {row["run_dir"]}', flush=True)
            except Exception as exc:
                failures.append({'run': run_id, 'seed': seed, 'error_message': repr(exc)})
                print(f'[FAILED] F{function_id} run {run_id}: {exc}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_run = {
                executor.submit(
                    _run_one_parallel_job,
                    int(function_id), int(dimension), int(run_id), seed_start + int(run_id) - 1,
                    int(max_true_fes), bool(verbose), function_dir,
                ): run_id
                for run_id in range(1, int(n_runs) + 1)
            }
            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                try:
                    row = future.result()
                    rows.append(row)
                    _save_completed_so_far(rows, function_dir, function_id, dimension)
                    print(f'[VISIBLE NOW] {row["run_dir"]}', flush=True)
                    print(
                        f"[COMPLETED] F{function_id} run {run_id:02d}/{n_runs} | "
                        f"seed={row['seed']} | best_f={row['best_f']:.12e} | error={row['error']:.3e}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append({'run': run_id, 'seed': seed_start + int(run_id) - 1, 'error_message': repr(exc)})
                    print(f'[FAILED] F{function_id} run {run_id}: {exc}', flush=True)

    if len(rows) == 0:
        raise RuntimeError(f'All {n_runs} runs failed for CEC-2017 F{function_id}.')

    df = pd.DataFrame(rows).sort_values('run').reset_index(drop=True)
    df.to_csv(os.path.join(function_dir, '30run_results.csv'), index=False)
    df.to_csv(os.path.join(function_dir, f'F{function_id}_D{dimension}_{n_runs}run_summary.csv'), index=False)
    _function_30run_statistics(df, function_id, dimension, function_dir)

    if failures:
        pd.DataFrame(failures).sort_values('run').to_csv(
            os.path.join(function_dir, 'failed_runs.csv'), index=False
        )

    _essential_function_reports(df, function_dir, function_id, dimension)
    _best_run_report(df, function_dir, function_id, dimension, max_true_fes)

    print(f'\n[SAVED] F{function_id} 30-run results and statistics -> {os.path.abspath(function_dir)}')
    return df

def _save_show_comparative_figure(fig, filename_stem):
    """Save one comparative figure as PNG and PDF, show it, then close."""
    png_path = os.path.join(RESULT_DIR, filename_stem + '.png')
    pdf_path = os.path.join(RESULT_DIR, filename_stem + '.pdf')
    fig.tight_layout()
    fig.savefig(png_path, dpi=450, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return (png_path, pdf_path)

def _positive_plot_value(v, floor=1e-16):
    """Return finite non-negative values suitable for logarithmic plots."""
    a = np.asarray(v, dtype=float)
    a = np.where(np.isfinite(a), np.maximum(a, floor), np.nan)
    return a

def plot_suite_comparisons(all_df, summary_df, dimension):
    """
    Create publication-style comparative plots for a completed CEC-2017
    suite without changing any optimization logic or FE accounting.

    Plots
    -----
    1. Best and mean final error by CEC function (log scale).
    2. Mean final objective value by CEC function.
    3. Mean runtime by CEC function.
    4. Mean TRUE FE consumed by CEC function.
    5. Median convergence-error comparison across all completed functions.

    All figures are displayed on screen and saved to RESULT_DIR as both
    PNG (300 dpi) and vector PDF.
    """
    if summary_df is None or len(summary_df) == 0:
        return []
    s = summary_df.sort_values('function').copy()
    fids = s['function'].to_numpy(dtype=int)
    labels = [f'F{int(fid)}' for fid in fids]
    saved = []
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.semilogy(fids, _positive_plot_value(s['best_error'].to_numpy(dtype=float)), marker='o', linewidth=1.5, label='Best error')
    ax.semilogy(fids, _positive_plot_value(s['mean_error'].to_numpy(dtype=float)), marker='s', linewidth=1.5, label='Mean error')
    ax.set_xlabel('CEC-2017 function')
    ax.set_ylabel('Final error |f - f*| / post-run error')
    ax.set_title(f'MB-DTASR-PFSA | Final Error Comparison | D={dimension}')
    ax.set_xticks(fids)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(True, alpha=0.25)
    ax.legend()
    saved.extend(_save_show_comparative_figure(fig, f'CEC2017_D{dimension}_comparison_final_error'))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(fids, s['mean_f'].to_numpy(dtype=float), marker='o', linewidth=1.5)
    ax.set_xlabel('CEC-2017 function')
    ax.set_ylabel('Mean final true objective')
    ax.set_title(f'MB-DTASR-PFSA | Mean Final Objective Comparison | D={dimension}')
    ax.set_xticks(fids)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(True, alpha=0.25)
    saved.extend(_save_show_comparative_figure(fig, f'CEC2017_D{dimension}_comparison_mean_objective'))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(fids, s['mean_runtime_s'].to_numpy(dtype=float))
    ax.set_xlabel('CEC-2017 function')
    ax.set_ylabel('Mean runtime (s)')
    ax.set_title(f'MB-DTASR-PFSA | Runtime Comparison | D={dimension}')
    ax.set_xticks(fids)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(True, axis='y', alpha=0.25)
    saved.extend(_save_show_comparative_figure(fig, f'CEC2017_D{dimension}_comparison_runtime'))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(fids, s['mean_FE'].to_numpy(dtype=float))
    ax.set_xlabel('CEC-2017 function')
    ax.set_ylabel('Mean TRUE function evaluations')
    ax.set_title(f'MB-DTASR-PFSA | TRUE-FE Comparison | D={dimension}')
    ax.set_xticks(fids)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(True, axis='y', alpha=0.25)
    saved.extend(_save_show_comparative_figure(fig, f'CEC2017_D{dimension}_comparison_true_fe'))
    if all_df is not None and len(all_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        convergence_found = False
        for fid in sorted(all_df['function'].unique()):
            g = all_df[all_df['function'] == fid]
            ref = float(g['reference_optimum'].iloc[0])
            run_curves = []
            max_fe_seen = 0
            for _, row in g.iterrows():
                seed = int(row['seed'])
                cycle_path = os.path.join(row['run_dir'], f'F{int(fid)}_D{dimension}_run{int(row["run"]):02d}_seed{seed}_cycles.csv')
                if not os.path.exists(cycle_path):
                    continue
                cdf = pd.read_csv(cycle_path)
                if 'true_fe' not in cdf.columns or 'best_true_f' not in cdf.columns or len(cdf) == 0:
                    continue
                fe = cdf['true_fe'].to_numpy(dtype=int)
                bf = cdf['best_true_f'].to_numpy(dtype=float)
                if np.isfinite(ref):
                    err = np.maximum(bf - ref, 1e-16)
                else:
                    err = np.maximum(np.abs(bf), 1e-16)
                if len(fe) > 0:
                    max_fe_seen = max(max_fe_seen, int(fe[-1]))
                    run_curves.append((fe, err))
            if len(run_curves) == 0:
                continue
            convergence_found = True
            n_grid = min(300, max(40, max_fe_seen))
            grid = np.unique(np.linspace(min((int(curve[0][0]) for curve in run_curves)), max_fe_seen, n_grid).astype(int))
            sampled = []
            for fe, err in run_curves:
                ids = np.searchsorted(fe, grid, side='right') - 1
                ids = np.clip(ids, 0, len(err) - 1)
                sampled.append(err[ids])
            sampled = np.vstack(sampled)
            median_curve = np.median(sampled, axis=0)
            ax.semilogy(grid, median_curve, linewidth=1.2, label=f'F{int(fid)}')
        if convergence_found:
            ax.set_xlabel('TRUE function evaluations')
            ax.set_ylabel('Median best-so-far error')
            ax.set_title(f'MB-DTASR-PFSA | Comparative Convergence | D={dimension}')
            ax.grid(True, alpha=0.25)
            if len(fids) <= 15:
                ax.legend(ncol=2, fontsize=15)
            else:
                ax.legend(ncol=3, fontsize=15)
            saved.extend(_save_show_comparative_figure(fig, f'CEC2017_D{dimension}_comparison_convergence'))
        else:
            plt.close(fig)
    print()
    print('-' * 112)
    print('COMPARATIVE PLOTS')
    print('-' * 112)
    for path in saved:
        print(path)
    print('-' * 112)
    return saved

def run_full_suite(function_ids=SUITE_FUNCTION_IDS, dimension=DIM, n_runs=RUNS_PER_FUNCTION, master_seed=MASTER_SEED, max_true_fes=MAX_TRUE_FES, verbose=True, skip_unavailable=SKIP_UNAVAILABLE_FUNCTIONS):
    """
    Run the same MB-DTASR-PFSA V19 framework on a complete CEC-2017 suite
    or any user-selected subset.

    IMPORTANT
    ---------
    No function-specific tuning or known optimum information is passed
    to the optimizer. Every function uses the same framework settings.

    Results saved:
      1. Per-run CSV for each function.
      2. Combined all-runs CSV.
      3. Function-level statistical summary CSV.
    """
    all_rows = []
    failures = []
    for fid in function_ids:
        print()
        print('=' * 112)
        print(f'STARTING CEC-2017 F{fid} | D={dimension} | runs={n_runs}')
        print('=' * 112)
        try:
            df = run_function_multiple(function_id=fid, dimension=dimension, n_runs=n_runs, master_seed=master_seed, max_true_fes=max_true_fes, verbose=verbose)
            all_rows.append(df)
        except Exception as exc:
            failures.append({'function': int(fid), 'error_message': repr(exc)})
            print(f'\n[WARNING] F{fid} failed: {exc}')
            if not skip_unavailable:
                raise
    if len(all_rows) == 0:
        raise RuntimeError('No CEC-2017 function completed successfully.')
    all_df = pd.concat(all_rows, ignore_index=True)
    all_path = os.path.join(RESULT_DIR, f'CEC2017_D{dimension}_all_runs.csv')
    all_df.to_csv(all_path, index=False)
    summary_rows = []
    for fid, g in all_df.groupby('function', sort=True):
        err = g['error'].to_numpy(dtype=float)
        bestf = g['best_f'].to_numpy(dtype=float)
        runtime = g['runtime_s'].to_numpy(dtype=float)
        summary_rows.append({'function': int(fid), 'backend_class': str(g['backend_class'].iloc[0]), 'problem_name': str(g['problem_name'].iloc[0]), 'reference_optimum': float(g['reference_optimum'].iloc[0]) if np.isfinite(g['reference_optimum'].iloc[0]) else np.nan, 'reference_source': str(g['reference_source'].iloc[0]), 'dimension': int(dimension), 'runs': len(g), 'best_f': float(np.min(bestf)), 'mean_f': float(np.mean(bestf)), 'median_f': float(np.median(bestf)), 'std_f': float(np.std(bestf, ddof=1) if len(bestf) > 1 else 0.0), 'best_error': float(np.min(err)), 'mean_error': float(np.mean(err)), 'median_error': float(np.median(err)), 'std_error': float(np.std(err, ddof=1) if len(err) > 1 else 0.0), 'mean_FE': float(np.mean(g['true_fes'])), 'mean_runtime_s': float(np.mean(runtime)), 'log10_mean_error': float(np.log10(max(np.mean(np.maximum(err, 0.0)), 1e-300)))})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULT_DIR, f'CEC2017_D{dimension}_suite_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    plot_suite_comparisons(all_df=all_df, summary_df=summary_df, dimension=dimension)
    if len(failures) > 0:
        failure_df = pd.DataFrame(failures)
        failure_path = os.path.join(RESULT_DIR, f'CEC2017_D{dimension}_failures.csv')
        failure_df.to_csv(failure_path, index=False)
    print()
    print('=' * 112)
    print('CEC-2017 SUITE COMPLETE')
    print('=' * 112)
    print(f'Dimension             : {dimension}')
    print(f'Requested functions   : {len(function_ids)}')
    print(f"Completed functions   : {summary_df['function'].nunique()}")
    print(f'Runs/function         : {n_runs}')
    print(f'TRUE FE/run           : {max_true_fes}')
    print(f'Combined results      : {all_path}')
    print(f'Suite summary         : {summary_path}')
    if len(failures) > 0:
        print(f"Failed/skipped funcs  : {[x['function'] for x in failures]}")
    print('=' * 112)
    return (all_df, summary_df)

def run_cec2017_dimensions(dimensions=(10, 30, 50, 100), function_ids=SUITE_FUNCTION_IDS, n_runs=1, master_seed=MASTER_SEED, max_true_fes=MAX_TRUE_FES, verbose=False):
    """
    Convenience runner for the same generalized framework across
    CEC-2017 dimensions without function-specific tuning.
    """
    summaries = []
    for d in dimensions:
        print()
        print('=' * 112)
        print(f'CEC-2017 MULTI-DIMENSION TEST | D={d}')
        print('=' * 112)
        _, summary = run_full_suite(function_ids=function_ids, dimension=d, n_runs=n_runs, master_seed=master_seed, max_true_fes=max_true_fes, verbose=verbose, skip_unavailable=SKIP_UNAVAILABLE_FUNCTIONS)
        summary = summary.copy()
        summary['test_dimension'] = d
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    path = os.path.join(RESULT_DIR, 'CEC2017_multi_dimension_summary.csv')
    combined.to_csv(path, index=False)
    print(f'\nSaved combined CEC-2017 multi-dimension summary to:\n{path}')
    return combined

def _publication_save_figure(fig, folder, stem):
    """Save a publication figure as 300-dpi PNG + vector PDF and display."""
    os.makedirs(folder, exist_ok=True)
    png = os.path.join(folder, stem + '.png')
    pdf = os.path.join(folder, stem + '.pdf')
    fig.tight_layout()
    fig.savefig(png, dpi=450, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    if PUBLICATION_SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return (png, pdf)

def _bootstrap_ci(values, statistic='mean', confidence=PUBLICATION_CONFIDENCE, n_boot=PUBLICATION_BOOTSTRAP_SAMPLES, seed=MASTER_SEED):
    """Deterministic nonparametric bootstrap CI for mean or median."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(int(seed))
    ids = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    samples = x[ids]
    if statistic == 'median':
        vals = np.median(samples, axis=1)
    else:
        vals = np.mean(samples, axis=1)
    alpha = 1.0 - float(confidence)
    return (float(np.quantile(vals, alpha / 2.0)), float(np.quantile(vals, 1.0 - alpha / 2.0)))

def _vargha_delaney_a12(a, b):
    """
    Vargha-Delaney A12 for a minimization problem.
    A12 > 0.5 means sample a tends to have SMALLER (better) values than b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    wins = 0.0
    for av in a:
        wins += np.sum(av < b)
        wins += 0.5 * np.sum(av == b)
    return float(wins / (len(a) * len(b)))

def _a12_magnitude(a12):
    """Common A12 magnitude bands based on distance from 0.5."""
    if not np.isfinite(a12):
        return 'NA'
    d = abs(float(a12) - 0.5)
    if d < 0.06:
        return 'negligible'
    if d < 0.14:
        return 'small'
    if d < 0.21:
        return 'medium'
    return 'large'

def _holm_adjust(p_values):
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return out
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out

def _step_sample_curve(fe, values, grid):
    fe = np.asarray(fe, dtype=int)
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=int)
    ids = np.searchsorted(fe, grid, side='right') - 1
    ids = np.clip(ids, 0, len(values) - 1)
    return values[ids]

def _summarize_final_runs(run_df, function_id, dimension, folder):
    """Descriptive statistics and reproducibility summary for 30 runs."""
    e = run_df['error'].to_numpy(dtype=float)
    f = run_df['best_f'].to_numpy(dtype=float)
    rt = run_df['runtime_s'].to_numpy(dtype=float)
    fes = run_df['true_fes'].to_numpy(dtype=float)
    ef = e[np.isfinite(e)]
    mean_lo, mean_hi = _bootstrap_ci(ef, 'mean')
    med_lo, med_hi = _bootstrap_ci(ef, 'median')
    if len(ef) >= 3:
        try:
            sw_stat, sw_p = shapiro(ef)
        except Exception:
            sw_stat, sw_p = (np.nan, np.nan)
    else:
        sw_stat, sw_p = (np.nan, np.nan)
    if len(ef):
        q1, q3 = np.quantile(ef, [0.25, 0.75])
        median = np.median(ef)
        mad = np.median(np.abs(ef - median))
    else:
        q1 = q3 = median = mad = np.nan
    success = np.isfinite(e) & (np.abs(e) <= PUBLICATION_SUCCESS_TOL)
    row = {'function': int(function_id), 'dimension': int(dimension), 'n_runs': int(len(run_df)), 'reference_optimum': float(run_df['reference_optimum'].iloc[0]), 'best_f_min': float(np.nanmin(f)), 'best_f_mean': float(np.nanmean(f)), 'best_f_median': float(np.nanmedian(f)), 'best_f_std': float(np.nanstd(f, ddof=1)) if len(f) > 1 else 0.0, 'best_f_worst': float(np.nanmax(f)), 'error_best': float(np.nanmin(e)), 'error_mean': float(np.nanmean(e)), 'error_median': float(np.nanmedian(e)), 'error_std': float(np.nanstd(e, ddof=1)) if len(e) > 1 else 0.0, 'error_worst': float(np.nanmax(e)), 'error_q1': float(q1), 'error_q3': float(q3), 'error_iqr': float(q3 - q1) if np.isfinite(q3) else np.nan, 'error_mad': float(mad), 'mean_error_ci_low': mean_lo, 'mean_error_ci_high': mean_hi, 'median_error_ci_low': med_lo, 'median_error_ci_high': med_hi, 'success_tolerance': float(PUBLICATION_SUCCESS_TOL), 'success_count': int(np.sum(success)), 'success_rate_percent': float(100.0 * np.mean(success)), 'mean_true_fes': float(np.nanmean(fes)), 'std_true_fes': float(np.nanstd(fes, ddof=1)) if len(fes) > 1 else 0.0, 'mean_runtime_s': float(np.nanmean(rt)), 'median_runtime_s': float(np.nanmedian(rt)), 'std_runtime_s': float(np.nanstd(rt, ddof=1)) if len(rt) > 1 else 0.0, 'shapiro_W_error': float(sw_stat), 'shapiro_p_error': float(sw_p)}
    sdf = pd.DataFrame([row])
    sdf.to_csv(os.path.join(folder, 'publication_descriptive_statistics.csv'), index=False)
    return sdf

def _aggregate_cycle_diagnostics(run_outputs, folder, function_id, dimension):
    """Aggregate states, operator use, productivity, reliability and TR traces."""
    state_rows = []
    op_rows = []
    prod_rows = []
    curves = []
    op_cols = ['true_BWR', 'true_BMR', 'true_INCUMBENT', 'true_POLL', 'true_SURR_LOCAL', 'true_QUAD_TR', 'true_ANISO', 'true_PRECISION', 'true_NOVELTY', 'true_COVERAGE']
    prod_cols = ['prod_BWR', 'prod_BMR', 'prod_INC', 'prod_POLL', 'prod_SR', 'prod_Q', 'prod_ANISO', 'prod_PRECISION', 'prod_NOV']
    for run_idx, out in enumerate(run_outputs, start=1):
        cdf = out['cycles'].copy()
        if len(cdf) == 0:
            continue
        counts = cdf['state'].value_counts()
        for state, count in counts.items():
            state_rows.append({'run': run_idx, 'state': str(state), 'cycles': int(count)})
        for col in op_cols:
            if col in cdf.columns:
                op_rows.append({'run': run_idx, 'operator': col.replace('true_', ''), 'true_evaluations': float(cdf[col].sum())})
        for col in prod_cols:
            if col in cdf.columns:
                prod_rows.append({'run': run_idx, 'source': col.replace('prod_', ''), 'mean_productivity': float(cdf[col].mean())})
        if 'true_fe' in cdf.columns:
            curves.append(cdf)
    state_df = pd.DataFrame(state_rows)
    op_df = pd.DataFrame(op_rows)
    prod_df = pd.DataFrame(prod_rows)
    if len(state_df):
        state_df.to_csv(os.path.join(folder, 'state_usage_by_run.csv'), index=False)
        ss = state_df.groupby('state')['cycles'].agg(['mean', 'std', 'median', 'sum']).reset_index()
        ss.to_csv(os.path.join(folder, 'state_usage_summary.csv'), index=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(ss['state'], ss['mean'])
        ax.set_xlabel('Search state')
        ax.set_ylabel('Mean cycles per run')
        ax.set_title(f'Search-State Usage | CEC-2017 F{function_id} | D={dimension}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.25)
        _publication_save_figure(fig, folder, 'plot_state_usage')
    if len(op_df):
        op_df.to_csv(os.path.join(folder, 'operator_true_evaluations_by_run.csv'), index=False)
        osum = op_df.groupby('operator')['true_evaluations'].agg(['mean', 'std', 'median', 'sum']).reset_index()
        osum.to_csv(os.path.join(folder, 'operator_true_evaluations_summary.csv'), index=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(osum['operator'], osum['mean'])
        ax.set_xlabel('Candidate source')
        ax.set_ylabel('Mean TRUE evaluations per run')
        ax.set_title(f'TRUE-Evaluation Allocation by Source | F{function_id} | D={dimension}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.25)
        _publication_save_figure(fig, folder, 'plot_operator_true_evaluation_allocation')
    if len(prod_df):
        prod_df.to_csv(os.path.join(folder, 'source_productivity_by_run.csv'), index=False)
        ps = prod_df.groupby('source')['mean_productivity'].agg(['mean', 'std', 'median']).reset_index()
        ps.to_csv(os.path.join(folder, 'source_productivity_summary.csv'), index=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(ps['source'], ps['mean'])
        ax.set_xlabel('Source')
        ax.set_ylabel('Mean verified productivity')
        ax.set_title(f'Verified Source Productivity | F{function_id} | D={dimension}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.25)
        _publication_save_figure(fig, folder, 'plot_source_productivity')
    if len(curves):
        min_fe = max((int(c['true_fe'].iloc[0]) for c in curves))
        max_fe = min((int(c['true_fe'].iloc[-1]) for c in curves))
        if max_fe >= min_fe:
            grid = np.unique(np.linspace(min_fe, max_fe, min(250, max_fe - min_fe + 1)).astype(int))
            metrics = [('reliability_after', 'Surrogate reliability', 'plot_reliability_trace'), ('mean_tr_radius', 'Mean basin trust-region radius', 'plot_trust_region_trace'), ('meaningful_stall_age', 'Meaningful stagnation age', 'plot_meaningful_stall_trace')]
            trace_rows = []
            for col, ylabel, stem in metrics:
                mats = []
                for cdf in curves:
                    if col not in cdf.columns:
                        continue
                    mats.append(_step_sample_curve(cdf['true_fe'], cdf[col], grid))
                if not mats:
                    continue
                mat = np.vstack(mats)
                med = np.nanmedian(mat, axis=0)
                q1 = np.nanquantile(mat, 0.25, axis=0)
                q3 = np.nanquantile(mat, 0.75, axis=0)
                for fev, a, b, d in zip(grid, q1, med, q3):
                    trace_rows.append({'metric': col, 'true_fe': fev, 'q1': a, 'median': b, 'q3': d})
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(grid, med, linewidth=1.8, label='Median')
                ax.fill_between(grid, q1, q3, alpha=0.2, label='IQR')
                ax.set_xlabel('TRUE function evaluations')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel} | F{function_id} | D={dimension}')
                ax.grid(True, alpha=0.25)
                ax.legend()
                _publication_save_figure(fig, folder, stem)
            if trace_rows:
                pd.DataFrame(trace_rows).to_csv(os.path.join(folder, 'adaptive_controller_traces.csv'), index=False)
    return (state_df, op_df, prod_df)

def _publication_convergence(run_outputs, reference, folder, function_id, dimension):
    """Median/IQR convergence across all independent runs."""
    curves = []
    max_len = 0
    for out in run_outputs:
        h = np.asarray(out['best_history'], dtype=float)
        if np.isfinite(reference):
            err = np.maximum(h - reference, 1e-16)
        else:
            err = np.maximum(np.abs(h), 1e-16)
        curves.append(err)
        max_len = max(max_len, len(err))
    grid = np.arange(1, max_len + 1, dtype=int)
    mat = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        mat[i, :len(c)] = c
        if len(c) < max_len and len(c):
            mat[i, len(c):] = c[-1]
    q1 = np.nanquantile(mat, 0.25, axis=0)
    med = np.nanmedian(mat, axis=0)
    q3 = np.nanquantile(mat, 0.75, axis=0)
    mean = np.nanmean(mat, axis=0)
    cdf = pd.DataFrame({'true_fe': grid, 'error_q1': q1, 'error_median': med, 'error_q3': q3, 'error_mean': mean})
    cdf.to_csv(os.path.join(folder, 'convergence_quantiles_30runs.csv'), index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(grid, np.maximum(med, 1e-16), linewidth=2.0, label='Median error')
    ax.fill_between(grid, np.maximum(q1, 1e-16), np.maximum(q3, 1e-16), alpha=0.2, label='IQR')
    ax.semilogy(grid, np.maximum(mean, 1e-16), linewidth=1.0, linestyle='--', label='Mean error')
    ax.set_xlabel('TRUE function evaluations')
    ax.set_ylabel('Best-so-far error')
    ax.set_title(f'{len(run_outputs)}-Run Convergence | MB-DTASR-PFSA | F{function_id} | D={dimension}')
    ax.grid(True, alpha=0.25)
    ax.legend()
    _publication_save_figure(fig, folder, 'plot_convergence_median_iqr')
    return cdf

def _publication_distribution_plots(run_df, folder, function_id, dimension):
    e = np.maximum(np.abs(run_df['error'].to_numpy(dtype=float)), 1e-16)
    rt = run_df['runtime_s'].to_numpy(dtype=float)
    runs = run_df['run'].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(e, vert=True, showmeans=True)
    ax.set_yscale('log')
    ax.set_xticks([1])
    ax.set_xticklabels(['MB-DTASR-PFSA'])
    ax.set_ylabel('Final error')
    ax.set_title(f'Final Error Distribution | F{function_id} | D={dimension}')
    ax.grid(True, axis='y', alpha=0.25)
    _publication_save_figure(fig, folder, 'plot_final_error_boxplot')
    fig, ax = plt.subplots(figsize=(8, 6))
    vp = ax.violinplot(e, showmeans=True, showmedians=True)
    ax.set_yscale('log')
    ax.set_xticks([1])
    ax.set_xticklabels(['MB-DTASR-PFSA'])
    ax.set_ylabel('Final error')
    ax.set_title(f'Final Error Violin Plot | F{function_id} | D={dimension}')
    ax.grid(True, axis='y', alpha=0.25)
    _publication_save_figure(fig, folder, 'plot_final_error_violin')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(runs, e, marker='o', linewidth=1.2)
    ax.set_xlabel('Independent run')
    ax.set_ylabel('Final error')
    ax.set_title(f'Run-wise Final Error | F{function_id} | D={dimension}')
    ax.grid(True, alpha=0.25)
    _publication_save_figure(fig, folder, 'plot_runwise_final_error')
    es = np.sort(e)
    ecdf = np.arange(1, len(es) + 1) / len(es)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(es, ecdf, marker='.')
    ax.set_xlabel('Final error')
    ax.set_ylabel('Empirical cumulative probability')
    ax.set_title(f'ECDF of Final Error | F{function_id} | D={dimension}')
    ax.grid(True, alpha=0.25)
    _publication_save_figure(fig, folder, 'plot_final_error_ecdf')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(rt, vert=True, showmeans=True)
    ax.set_xticks([1])
    ax.set_xticklabels(['MB-DTASR-PFSA'])
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f'Runtime Distribution | F{function_id} | D={dimension}')
    ax.grid(True, axis='y', alpha=0.25)
    _publication_save_figure(fig, folder, 'plot_runtime_boxplot')

def _load_optional_method_data(path, label, folder):
    if path is None:
        candidate = os.path.join(folder, label)
        path = candidate if os.path.exists(candidate) else None
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {'method', 'run', 'error'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path} must contain columns: method, run, error')
    return df.copy()

def _pairwise_method_statistics(baseline_df, other_df, folder, prefix):
    """Paired Wilcoxon + A12 + Holm when external method/ablation data exist."""
    if other_df is None or len(other_df) == 0:
        return None
    base = baseline_df[['run', 'error']].rename(columns={'error': 'baseline_error'})
    rows = []
    for method, g in other_df.groupby('method'):
        merged = base.merge(g[['run', 'error']], on='run', how='inner')
        a = merged['baseline_error'].to_numpy(dtype=float)
        b = merged['error'].to_numpy(dtype=float)
        if len(a) == 0:
            continue
        try:
            if len(a) >= 2 and (not np.allclose(a, b)):
                stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
                test = 'paired Wilcoxon'
            else:
                stat, p = (np.nan, 1.0)
                test = 'paired Wilcoxon'
        except Exception:
            stat, p = mannwhitneyu(a, b, alternative='two-sided')
            test = 'Mann-Whitney U fallback'
        rows.append({'baseline': 'MB-DTASR-PFSA V19', 'comparison': str(method), 'n_paired': int(len(a)), 'test': test, 'statistic': float(stat) if np.isfinite(stat) else np.nan, 'p_raw': float(p), 'A12_baseline_better': _vargha_delaney_a12(a, b), 'baseline_median_error': float(np.median(a)), 'comparison_median_error': float(np.median(b)), 'wins': int(np.sum(a < b)), 'ties': int(np.sum(a == b)), 'losses': int(np.sum(a > b))})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out['p_holm'] = _holm_adjust(out['p_raw'].to_numpy(dtype=float))
    out['significant_0.05'] = out['p_holm'] < 0.05
    out['A12_magnitude'] = out['A12_baseline_better'].map(_a12_magnitude)
    out.to_csv(os.path.join(folder, prefix + '_pairwise_statistics.csv'), index=False)
    return out

def _write_comparison_templates(folder):
    """Create templates only; no fake comparator or ablation results are invented."""
    template = pd.DataFrame({'method': ['PSO', 'PSO', 'DE', 'DE'], 'run': [1, 2, 1, 2], 'error': [np.nan, np.nan, np.nan, np.nan], 'best_f': [np.nan, np.nan, np.nan, np.nan], 'runtime_s': [np.nan, np.nan, np.nan, np.nan]})
    comp = os.path.join(folder, 'comparator_results_TEMPLATE.csv')
    abl = os.path.join(folder, 'ablation_results_TEMPLATE.csv')
    if not os.path.exists(comp):
        template.to_csv(comp, index=False)
    if not os.path.exists(abl):
        at = template.copy()
        at['method'] = ['-NewtonTR', '-NewtonTR', '-Aniso', '-Aniso']
        at.to_csv(abl, index=False)

def _update_global_manual_function_summary(root, dimension):
    """Scan completed manually-run function folders and update a baseline cross-function table."""
    rows = []
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        path = os.path.join(folder, 'publication_descriptive_statistics.csv')
        if os.path.isdir(folder) and os.path.exists(path):
            try:
                d = pd.read_csv(path)
                if len(d) and int(d['dimension'].iloc[0]) == int(dimension):
                    rows.append(d.iloc[0].to_dict())
            except Exception:
                pass
    if not rows:
        return None
    g = pd.DataFrame(rows).sort_values('function')
    gd = os.path.join(root, f'GLOBAL_D{dimension}')
    os.makedirs(gd, exist_ok=True)
    g.to_csv(os.path.join(gd, 'completed_functions_summary.csv'), index=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    y = np.maximum(np.abs(g['error_mean'].to_numpy(dtype=float)), 1e-16)
    ax.semilogy(g['function'], y, marker='o')
    ax.set_xlabel('CEC-2017 function')
    ax.set_ylabel('Mean final error')
    ax.set_title(f'Completed Manual Functions | Mean Error | D={dimension}')
    ax.set_xticks(g['function'].to_numpy(dtype=int))
    ax.grid(True, alpha=0.25)
    _publication_save_figure(fig, gd, 'plot_completed_functions_mean_error')
    return g

def _update_global_method_statistics(root, dimension, filename, prefix):
    """
    Build cross-function publication statistics as manually completed function
    folders accumulate. Uses one median error per method per function.

    Outputs when data are sufficient:
      - function x method median-error table
      - average-rank table
      - Friedman omnibus test (>=3 methods, >=2 complete functions)
      - baseline-vs-method Wilcoxon across functions + Holm + A12
    """
    records = []
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        base_path = os.path.join(folder, 'publication_run_results.csv')
        other_path = os.path.join(folder, filename)
        if not (os.path.isdir(folder) and os.path.exists(base_path) and os.path.exists(other_path)):
            continue
        try:
            base = pd.read_csv(base_path)
            other = pd.read_csv(other_path)
            if len(base) == 0 or int(base['dimension'].iloc[0]) != int(dimension):
                continue
            if not {'method', 'run', 'error'}.issubset(other.columns):
                continue
            fid = int(base['function'].iloc[0])
            records.append({'function': fid, 'method': 'MB-DTASR-PFSA V19', 'median_error': float(np.nanmedian(base['error'].to_numpy(dtype=float)))})
            for method, g in other.groupby('method'):
                records.append({'function': fid, 'method': str(method), 'median_error': float(np.nanmedian(g['error'].to_numpy(dtype=float)))})
        except Exception:
            continue
    if not records:
        return None
    long = pd.DataFrame(records)
    gd = os.path.join(root, f'GLOBAL_D{dimension}')
    os.makedirs(gd, exist_ok=True)
    long.to_csv(os.path.join(gd, prefix + '_function_method_median_errors.csv'), index=False)
    pivot = long.pivot_table(index='function', columns='method', values='median_error', aggfunc='first')
    complete = pivot.dropna(axis=0, how='any')
    pivot.to_csv(os.path.join(gd, prefix + '_median_error_matrix.csv'))
    if len(complete) == 0 or complete.shape[1] < 2:
        return complete
    rank_matrix = complete.apply(lambda row: pd.Series(rankdata(row.to_numpy(dtype=float), method='average'), index=complete.columns), axis=1)
    rank_matrix.to_csv(os.path.join(gd, prefix + '_rank_matrix.csv'))
    avg_rank = pd.DataFrame({'method': complete.columns, 'average_rank': rank_matrix.mean(axis=0).to_numpy(dtype=float)}).sort_values('average_rank')
    avg_rank.to_csv(os.path.join(gd, prefix + '_average_ranks.csv'), index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(avg_rank['method'], avg_rank['average_rank'])
    ax.set_xlabel('Method')
    ax.set_ylabel('Average rank (lower is better)')
    ax.set_title(f'{prefix.title()} Average Ranks Across Completed CEC Functions | D={dimension}')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.25)
    _publication_save_figure(fig, gd, prefix + '_plot_average_ranks')
    tests = []
    if complete.shape[1] >= 3 and len(complete) >= 2:
        try:
            stat, pval = friedmanchisquare(*[complete[c].to_numpy(dtype=float) for c in complete.columns])
        except Exception:
            stat, pval = (np.nan, np.nan)
        pd.DataFrame([{'n_functions': int(len(complete)), 'n_methods': int(complete.shape[1]), 'friedman_statistic': stat, 'friedman_p': pval, 'significant_0.05': bool(np.isfinite(pval) and pval < 0.05)}]).to_csv(os.path.join(gd, prefix + '_friedman_test.csv'), index=False)
    baseline_name = 'MB-DTASR-PFSA V19'
    if baseline_name in complete.columns:
        a = complete[baseline_name].to_numpy(dtype=float)
        for method in complete.columns:
            if method == baseline_name:
                continue
            b = complete[method].to_numpy(dtype=float)
            try:
                if len(a) >= 2 and (not np.allclose(a, b)):
                    st, pv = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
                else:
                    st, pv = (np.nan, 1.0)
            except Exception:
                st, pv = (np.nan, np.nan)
            tests.append({'baseline': baseline_name, 'comparison': method, 'n_functions': int(len(a)), 'wilcoxon_statistic': st, 'p_raw': pv, 'A12_baseline_better': _vargha_delaney_a12(a, b), 'wins': int(np.sum(a < b)), 'ties': int(np.sum(a == b)), 'losses': int(np.sum(a > b))})
    if tests:
        t = pd.DataFrame(tests)
        t['p_holm'] = _holm_adjust(t['p_raw'].to_numpy(dtype=float))
        t['significant_0.05'] = t['p_holm'] < 0.05
        t['A12_magnitude'] = t['A12_baseline_better'].map(_a12_magnitude)
        t.to_csv(os.path.join(gd, prefix + '_cross_function_wilcoxon_holm_A12.csv'), index=False)
    return complete

def run_publication_function(function_id=FUNCTION_ID, dimension=DIM, n_runs=RUNS_PER_FUNCTION, master_seed=MASTER_SEED, max_true_fes=MAX_TRUE_FES, verbose=True):
    df = run_function_multiple(function_id, dimension, n_runs, master_seed, max_true_fes, verbose)
    folder = os.path.dirname(df.iloc[0]['run_dir'])
    return df, pd.read_csv(os.path.join(folder, '30run_statistical_summary.csv'))


def validate_complete_cec2017_backend(dimension=DIM):
    """Uncounted preflight of all corrected CEC-2017 functions."""
    print("\n" + "=" * 112)
    print("PURE-PYTHON CEC-2017 COMPLETE BACKEND VALIDATION")
    print("=" * 112)
    print(f"Dimension : {dimension}")
    print("Functions : F1 and F3-F30 (F2 deleted)")
    print("Data rule : STRICT CEC-2017 signature; CEC-2020 folders are rejected")
    print("=" * 112)
    rows = []
    for fid in [1] + list(range(3, 31)):
        p = CEC2017Problem(fid, dimension)
        y = float(p.problem(np.zeros(dimension)))
        ok = np.isfinite(y)
        rows.append((fid, y, ok))
        mcount = p.problem.M.shape[0]
        scount = p.problem.OShift.shape[0]
        shuffle = 0 if p.problem.SS is None else int(p.problem.SS.size)
        print(
            f"F{fid:02d}: {'PASS' if ok else 'FAIL'} | "
            f"M={mcount} | Shift={scount} | Shuffle={shuffle} | "
            f"f(0)={y:.12e}"
        )
        if not ok:
            raise RuntimeError(f"CEC-2017 backend validation failed at F{fid}.")
    print("=" * 112)
    print(f"Input data: {CEC2017_INPUT_DATA}")
    print("ALL 29 CORRECTED FUNCTIONS PASSED INTERFACE/FINITE-VALUE PREFLIGHT.")
    print("=" * 112)
    return rows


if __name__ == '__main__':
    mode = str(RUN_MODE).strip().upper()
    if mode in ('SINGLE', 'SUITE'):
        validate_complete_cec2017_backend(DIM)
    if mode == 'SINGLE':
        publication_runs, publication_summary = run_publication_function(function_id=FUNCTION_ID, dimension=DIM, n_runs=RUNS_PER_FUNCTION, master_seed=MASTER_SEED, max_true_fes=MAX_TRUE_FES, verbose=True)
    elif mode == 'SUITE':
        all_results, suite_summary = run_full_suite(function_ids=SUITE_FUNCTION_IDS, dimension=DIM, n_runs=RUNS_PER_FUNCTION, master_seed=MASTER_SEED, max_true_fes=MAX_TRUE_FES, verbose=True, skip_unavailable=SKIP_UNAVAILABLE_FUNCTIONS)
    else:
        raise ValueError('RUN_MODE must be either "SINGLE" or "SUITE".')
