# layout_class.py  -- float32 + speed
#IF YOU WANT TO UNDERSTAND WHAT HAPPENS REAL LAYOUT CLASS SIMPLE
import numpy as np
import scipy as sc
import scipy.interpolate
import scipy.optimize

class layout_class:
    def __init__(self, init):
        layout = heristic_layout(init)
        self.kappa = layout.kappa
        self.generation = layout.generation
        self.misc = layout.misc

class heristic_layout:
    """
    Heuristic layout with float32 everywhere and faster alpha search:
    - precomputes per-block stats
    - computes backup-energy vs alpha using block totals (exact)
    - final pass identical in structure (wind/solar kappa, generation, misc)
    """

    def __init__(self, init):
        # Bind inputs (force float32 to avoid upcasting later)
        self.alpha_n   = int(init.n_alpha)
        self.CF_wind   = init.CF_wind.astype(np.float32, copy=False)     # (T, N)
        self.CF_solar  = init.CF_solar.astype(np.float32, copy=False)    # (T, N)
        self.load      = init.load.astype(np.float32, copy=False)        # (T, N)
        self.mean_load = init.mean_load.astype(np.float32, copy=False)   # (N,)
        self.beta      = np.float32(init.beta)
        self.gamma     = np.float32(init.gamma)

        # Spans + alpha grid (tiny local fix: columns vary)
        self.scheme_combination(init.layout_scheme, init.balancing_scheme, init)

        # Precompute constants for speed
        self._precompute_block_constants()

        # Alpha search
        alpha_opt = self.execute_schemes(self.alpha_span, alpha_opt_search=True)
        self.execute_schemes(alpha_opt)

        self.init = init

    # ------------------------------------------------------------------
    # Setup

    def scheme_combination(self, layout_scheme, balancing_scheme, init):
        n, country_int = init.n, init.country_int

        if layout_scheme == "global":
            self.alpha_span = np.linspace(0, 1, self.alpha_n, dtype=np.float32)[np.newaxis, :]  # (1, A)
            self.layout_span = np.array([0, n], dtype=np.int32)
        else:
            # rows = blocks; columns = alpha grid (varies across columns)
            self.alpha_span = np.tile(
                np.linspace(0, 1, self.alpha_n, dtype=np.float32)[np.newaxis, :],
                (int(init.n_country), 1),
            ).astype(np.float32, copy=False)  # (B, A)
            # assuming init.country_int is already boundaries [0, ..., n] (int)
            self.layout_span = np.asarray(country_int, dtype=np.int32)

        self.balancing_span = (
            np.array([0, n], dtype=np.int32)
            if balancing_scheme == "global"
            else np.asarray(country_int, dtype=np.int32)
        )

    def _precompute_block_constants(self):
        """
        Compute once (all float32):
          - block sizes & total mean loads (layout/balancing)
          - per-bus load share within balancing blocks
          - CF means and per-block denominators for wind/solar
          - per-block time series for fast backup-energy evaluation
        """
        # Boundaries → sizes
        self.layout_block_sizes = np.diff(self.layout_span).astype(np.int32)   # (B,)
        self.bal_block_sizes    = np.diff(self.balancing_span).astype(np.int32)  # (B_bal,)

        # Per-block total mean loads
        self.total_load_blocks_layout = np.add.reduceat(
            self.mean_load, self.layout_span[:-1]
        ).astype(np.float32, copy=False)[:, None]  # (B,1)

        self.total_load_blocks_bal = np.add.reduceat(
            self.mean_load, self.balancing_span[:-1]
        ).astype(np.float32, copy=False)[:, None]  # (B_bal,1)

        # Per-bus share L_n / L_block in balancing blocks
        bal_ids = np.repeat(
            np.arange(len(self.balancing_span) - 1, dtype=np.int32),
            self.bal_block_sizes,
        )
        bal_ids = bal_ids[: self.mean_load.shape[0]]
        self._bal_ids = bal_ids
        self.load_prop_per_bus = (self.mean_load / self.total_load_blocks_bal[self._bal_ids, 0]).astype(
            np.float32, copy=False
        )  # (N,)

        # CF means (per bus)
        self.CF_mean_wind  = self.CF_wind.mean(axis=0, dtype=np.float32)   # (N,)
        self.CF_mean_solar = self.CF_solar.mean(axis=0, dtype=np.float32)  # (N,)

        # Per-block denominators sum CF_mean^(beta+1)
        self.CF_sum_wind = np.add.reduceat(
            (self.CF_mean_wind ** (self.beta + np.float32(1.0))).astype(np.float32, copy=False),
            self.layout_span[:-1],
        ).astype(np.float32, copy=False)[:, None]  # (B,1)

        self.CF_sum_solar = np.add.reduceat(
            (self.CF_mean_solar ** (self.beta + np.float32(1.0))).astype(np.float32, copy=False),
            self.layout_span[:-1],
        ).astype(np.float32, copy=False)[:, None]  # (B,1)

        # Alpha grid for optimization
        self.alpha_grid = self.alpha_span[0]  # (A,) float32

        # ---------- Fast backup-energy precomputation (all float32) ----------
        w_wind  = (self.CF_mean_wind  ** self.beta).astype(np.float32, copy=False)  # (N,)
        w_solar = (self.CF_mean_solar ** self.beta).astype(np.float32, copy=False)  # (N,)

        B = int(self.layout_block_sizes.shape[0])
        T = int(self.CF_wind.shape[0])

        self.s_wind_block  = np.empty((B, T), dtype=np.float32)
        self.s_solar_block = np.empty((B, T), dtype=np.float32)
        self.load_block    = np.empty((B, T), dtype=np.float32)

        for b in range(B):
            i0, i1 = int(self.layout_span[b]), int(self.layout_span[b + 1])
            # BLAS-backed matmul → stays float32 if operands are float32
            self.s_wind_block[b]  = self.CF_wind[:, i0:i1]  @ w_wind[i0:i1]
            self.s_solar_block[b] = self.CF_solar[:, i0:i1] @ w_solar[i0:i1]
            self.load_block[b]    = self.load[:, i0:i1].sum(axis=1, dtype=np.float32)

        # c_wind / c_solar as float32
        self.c_wind  = (self.gamma * self.total_load_blocks_layout[:, 0] / self.CF_sum_wind[:, 0]).astype(
            np.float32, copy=False
        )  # (B,)
        self.c_solar = (self.gamma * self.total_load_blocks_layout[:, 0] / self.CF_sum_solar[:, 0]).astype(
            np.float32, copy=False
        )  # (B,)

    # ------------------------------------------------------------------
    # Main execution

    def execute_schemes(self, alpha, alpha_opt_search=False):
        # CF → generation and kappa
        wind_gen,  wind_kappa  = self.CF_prop(alpha,     self.CF_wind,  self.CF_mean_wind,  self.CF_sum_wind)
        solar_gen, solar_kappa = self.CF_prop(np.float32(1.0) - alpha, self.CF_solar, self.CF_mean_solar, self.CF_sum_solar)

        # balancing (vectorized)
        balancing = self.SynchBalancing(wind_gen, solar_gen)

        if alpha_opt_search:
            # Fast + exact backup-energy curve; returns float32
            alpha_opt = self.alpha_min_backup_energy(balancing).astype(np.float32, copy=False)
            # keep explicit alpha axis so [:,0] / [:,:,0] work later
            if alpha_opt.ndim == 1:
                alpha_opt = alpha_opt[:, None]       # (B,1)
            elif alpha_opt.ndim == 0:
                alpha_opt = alpha_opt[None, None]    # (1,1)
            return alpha_opt
        else:
            balancing2d = balancing[:, :, 0]
            # final outputs (same structure; float32)
            kappa = {
                "wind":  wind_kappa[:, 0],
                "solar": solar_kappa[:, 0],
                'backup': np.quantile(np.abs(balancing2d*(balancing2d<0)), q=0.99, axis=0)
            }


            ren = (wind_gen[:, :, 0] + solar_gen[:, :, 0]).astype(np.float32, copy=False)
            mismatch = (ren - self.load).astype(np.float32, copy=False)

            generation = {
                "wind":        wind_gen[:, :, 0],
                "solar":       solar_gen[:, :, 0],
                "backup":     (-balancing2d * (balancing2d < 0)).astype(np.float32, copy=False),
                "curtailment": ( balancing2d * (balancing2d > 0)).astype(np.float32, copy=False),
                "ren":         ren,
            }
            misc = {
                "mismatch":  mismatch,
                "injection": (mismatch - balancing2d).astype(np.float32, copy=False),
            }

            self.kappa = kappa
            self.generation = generation
            self.misc = misc
    # ------------------------------------------------------------------
    # Core math

    def CF_prop(self, alpha, CF, CF_mean=None, CF_sum=None):
        """
        a_b(α) = γ * α * L_block / Σ CF_mean^(β+1)   (per block)
        κ_n(α) = a_b(α) * CF_mean_n^β               (per bus)
        Gen(t,n,α) = κ_n(α) * CF(t,n)
        Shapes: alpha (B,A) or (B,1) ; CF (T,N) ; returns Gen (T,N,A or 1), kappa (N,A or 1)
        """
        if CF_mean is None:
            CF_mean = CF.mean(axis=0, dtype=np.float32)  # (N,)
        if CF_sum is None:
            CF_sum = np.add.reduceat(
                (CF_mean ** (self.beta + np.float32(1.0))).astype(np.float32, copy=False),
                self.layout_span[:-1],
            ).astype(np.float32, copy=False)[:, None]  # (B,1)

        avg_required_gen = (self.gamma * alpha * self.total_load_blocks_layout).astype(np.float32, copy=False)  # (B,A) or (B,1)
        a = (avg_required_gen / CF_sum).astype(np.float32, copy=False)  # (B,A) or (B,1)

        a_n = np.repeat(a, self.layout_block_sizes, axis=0)  # (N,A) or (N,1)
        kappa = (((CF_mean ** self.beta).astype(np.float32, copy=False)[None, :] * a_n.T).T).astype(
            np.float32, copy=False
        )  # (N,A) or (N,1)

        generation = (kappa[None, :, :] * CF[:, :, None]).astype(np.float32, copy=False)  # (T,N,A) or (T,N,1)
        return generation, kappa

    def SynchBalancing(self, wind_gen, solar_gen):
        """
        Distribute block-total mismatch to buses proportional to mean load.
        All float32.
        """
        mismatch = ((wind_gen + solar_gen) - self.load[:, :, None]).astype(np.float32, copy=False)  # (T,N,A)
        total_mismatch = np.add.reduceat(mismatch, self.balancing_span[:-1], axis=1).astype(
            np.float32, copy=False
        )  # (T,B_bal,A)
        total_mismatch_per_bus = np.repeat(total_mismatch, self.bal_block_sizes, axis=1).astype(
            np.float32, copy=False
        )  # (T,N,A)
        balancing = (self.load_prop_per_bus[None, :, None] * total_mismatch_per_bus).astype(
            np.float32, copy=False
        )
        return balancing

    # ------------------------------------------------------------------
    # SLOW!!!!!!!!!!!!!!!!!!!!!

    def alpha_min_backup_energy(self, balancing):
            backup_gen = -balancing * (balancing < 0)
            backup_energy = np.add.reduceat(backup_gen, self.layout_span[:-1], axis=1).sum(axis=0).T

            alpha = np.linspace(0,1,self.alpha_n)
            alpha_opt = []

            for country in range(backup_energy.shape[1]):
                f = scipy.interpolate.interp1d(alpha,backup_energy[:,country],kind='quadratic')
                alpha_opt.append(sc.optimize.minimize_scalar(f, bounds=(0,1), method='Bounded').x)

            return np.array(alpha_opt)