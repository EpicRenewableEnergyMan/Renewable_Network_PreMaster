import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu


class DensityTopologyTransOptimizer:
    """
    SIMP-style edge topology optimization for minimizing TransValue (TV).

    Key design choices:
    - Physics is honest: ρ only affects conductances W via w_e = w_min + ρ_e^p.
      No post-solve scaling of flows/PTDF by ρ.
    - Real sparsity/scarcity comes from the objective/constraints:
        * Option A (penalty): capex_lambda * sum(length_e * ρ_e)  [capex penalty]
        * Option A' (constraint): hard 'length_budget' enforced via length-weighted OC
        * Option B: soft edge-count penalty using a smooth count S(ρ_proj)
    - Connectivity: always include MST with tiny weight so solves are well-posed, without bias.
      (MST edges are NOT forced active for TV and are kept below w_floor so they never carry TV flow.)
    - Kcap active-set is compute-only (stability/speed). Final prune trims to exactly K.

    Returns: rho_design (pre-projection), rho_proj (post-projection), A_final (nxn adj), info
    """

    def __init__(self, bus_df, dist_matrix, injections,
                 # candidate graph
                 max_km=1000.0,
                 # SIMP penalization (p is scheduled)
                 p_start=1.5, p_end=4.0,
                 w_min=1e-8,
                 # Projection (gated ON only after V ramp)
                 use_projection=True,
                 beta_start=2.0, beta_end=10.0, eta=0.55,
                 proj_delay_frac=None,           # None => start at ramp end; else % of iters
                 # Volume schedule (percent-of-iterations timing)
                 target_edges=912,               # K (final target)
                 V0=0.20,                        # starting volume fraction (for OC if no length_budget)
                 V_min=0.001, V_max=0.20,        # safety bounds for V
                 warmup_frac=0.20,               # first 20% iters: hold V = V0
                 ramp_frac=0.75,                 # until 75%: cosine decay V0 -> V_target
                 cap_edges_factor=1.10,          # K' = 1.1 * K (headroom before prune)
                 move=0.05,                      # OC move limit
                 max_iters=500,
                 # TransValue
                 price=400.0,
                 quantile_q=0.99,
                 # batching
                 batch_scenarios=1024,           # None/0 => all scenarios
                 batch_quantile=0.98,
                 checkpoint_every=25,            # full eval every N iters
                 # connectivity safeguard
                 ensure_connected=True,
                 mst_w=2e-4,                     # kept < w_floor so MST never carries TV flow
                 # thresholds
                 thr_thresh=0.50,                # only for progress counter (edges~)
                 # pruning
                 prune_mode="FAST",              # "FAST" or "EXACT"
                 prune_batch=5,
                 # floors / sparsification
                 rho_floor=1e-2,                 # hard-zero ρ below this
                 w_floor=1e-3,                   # ignore edges with weight below this in physics/TV
                 hard_zero_every=10,             # how often to apply hard zeroing in the loop
                 # scarcity controls
                 capex_lambda=0.0,               # Option A (penalty): λ * sum(length * ρ_proj)
                 length_budget=None,             # Option A' (constraint): sum(length * ρ) = B (uses length-weighted OC)
                 soft_count_tau=None,            # Option B: sigmoid threshold (defaults to eta)
                 soft_count_alpha=0.08,          # Option B: sigmoid sharpness (~1/beta_end)
                 mu_softcount=1e4,               # Option B: μ * max(0, S(ρ_proj) - K_soft)^2
                 K_soft=None,                    # Option B: desired soft count cap (defaults to 1.1 * target_edges)
                 # misc
                 verbose=True):
        self.verbose = verbose

        # Data
        self.bus = bus_df
        self.coords = self.bus[['x','y']].to_numpy().astype(float)
        self.dist = np.asarray(dist_matrix, dtype=float)

        P = np.asarray(injections, dtype=float)
        n = len(bus_df)
        if P.shape[0] == n:
            pass
        elif P.shape[1] == n:
            P = P.T
        else:
            raise ValueError(f"injections must be (n,S) or (S,n); got {P.shape}, n={n}")
        # zero-mean injections per scenario
        self.P = P - P.mean(axis=0, keepdims=True)

        # Candidate edges
        edges, lengths = [], []
        for i in range(n):
            di = self.dist[i]
            for j in range(i+1, n):
                d = float(di[j])
                if d <= max_km:
                    edges.append((i, j)); lengths.append(d)
        if not edges:
            raise ValueError("No candidate edges within max_km.")
        self.edge_list = edges
        self.lengths = np.asarray(lengths, dtype=float)
        self.n, self.m = n, len(edges)
        self._edge_to_idx = {e:k for k,e in enumerate(self.edge_list)}

        # OC
        self.move = float(move)

        # Floors from args
        self.rho_floor = float(rho_floor)
        self.w_floor   = float(w_floor)
        self.hard_zero_every = int(hard_zero_every)

        # SIMP
        self.p_start = float(p_start); self.p_end = float(p_end)
        self.p = float(p_start)
        self.w_min = float(w_min)

        # Projection (gated)
        self.use_projection = bool(use_projection)
        self.beta_start = float(beta_start); self.beta_end = float(beta_end)
        self.eta = float(eta)

        # Scarcity controls
        self.capex_lambda   = float(capex_lambda)
        self.length_budget  = None if length_budget is None else float(length_budget)
        self.soft_count_tau = float(soft_count_tau) if (soft_count_tau is not None) else float(self.eta)
        self.soft_count_alpha = float(soft_count_alpha)
        self.mu_softcount   = float(mu_softcount)
        self.K_soft         = int(K_soft) if (K_soft is not None) else int(round(1.10 * target_edges))

        # Volume control (smooth schedule) for unweighted OC
        self.target_edges = int(target_edges)  # K
        self.cap_edges_factor = float(cap_edges_factor)  # K'/K
        self.Kprime_final = int(max(self.n - 1, int(self.cap_edges_factor * self.target_edges)))
        self.V0 = float(V0)
        self.V_min = float(V_min); self.V_max = float(V_max)
        self.warmup_frac = float(warmup_frac)
        self.ramp_frac   = float(ramp_frac)
        # V_target based on desired K'
        self.V_target = float(np.clip(self.Kprime_final / self.m, self.V_min, self.V_max))
        self.V = float(np.clip(self.V0, self.V_min, self.V_max))
        self.max_iters = int(max_iters)

        # Projection activation time (after V ramp by default)
        if proj_delay_frac is None:
            self.proj_delay_frac = self.ramp_frac
        else:
            self.proj_delay_frac = float(proj_delay_frac)
        Ttot = self.max_iters
        self._proj_on_it = max(int(self.ramp_frac * Ttot), int(self.proj_delay_frac * Ttot))

        # TransValue
        self.price = float(price)
        self.quantile_q = float(quantile_q)
        self.batch_scenarios = None if batch_scenarios in (None,0) else int(batch_scenarios)
        self.batch_quantile = float(batch_quantile)
        self.checkpoint_every = int(checkpoint_every)

        # Connectivity (MST kept below w_floor; not forced active for TV)
        self.ensure_connected = bool(ensure_connected)
        self.mst_w = float(mst_w)
        if self.ensure_connected:
            # ensure MST conductance is below w_floor so it cannot carry TV flow
            self.mst_w = min(self.mst_w, 0.5 * self.w_floor)

        # Precompute MST indices once (by 1/length)
        Gw = nx.Graph()
        for idx, (i, j) in enumerate(self.edge_list):
            Gw.add_edge(i, j, weight=1.0 / max(self.lengths[idx], 1e-6), idx=idx)
        T = nx.maximum_spanning_tree(Gw, weight='weight')
        self._mst_idx = np.fromiter((Gw[u][v]['idx'] for u, v in T.edges()), dtype=int)
        self._mst_set = set(self._mst_idx.tolist())

        # Finalization / logging
        self.thr_thresh = float(thr_thresh)
        self.prune_mode = prune_mode.upper()
        self.prune_batch = int(prune_batch)

        # Design vars
        self.rho = np.full(self.m, self.V, dtype=float)

        # RNG once for batching
        self.rng = np.random.default_rng(12345)

        # History
        self.hist = {"iter": [], "V": [], "p": [], "edges_est": [], "S_soft": [], "Kcap": []}

    # -------- schedules (percent-of-iteration) ----------
    def _V_schedule(self, it):
        T = self.max_iters
        warmup_end = int(self.warmup_frac * T)
        ramp_end   = int(self.ramp_frac   * T)
        V0 = self.V0; Vt = self.V_target

        if it <= warmup_end:
            return V0
        if it <= ramp_end and ramp_end > warmup_end:
            s = (it - warmup_end) / max(1, (ramp_end - warmup_end))  # [0,1]
            return Vt + 0.5 * (V0 - Vt) * (1.0 + np.cos(np.pi * s))
        return Vt

    def _p_schedule(self, it):
        T = self.max_iters
        ramp_end = int(self.ramp_frac * T)
        p0, p1 = self.p_start, self.p_end

        if it <= ramp_end:
            return p0
        # cosine rise over [ramp_end, T]
        s = (it - ramp_end) / max(1, T - ramp_end)
        s = max(0.0, min(1.0, s))
        return p0 + 0.5 * (p1 - p0) * (1.0 - np.cos(np.pi * s))

    def _beta(self, it):
        if it <= self._proj_on_it:
            return self.beta_start
        T = self.max_iters
        s = (it - self._proj_on_it) / max(1, T - self._proj_on_it)  # [0,1]
        return self.beta_start + 0.5 * (self.beta_end - self.beta_start) * (1.0 - np.cos(np.pi * s))

    # -------- physics ----------
    def _incidence(self, n, edge_list):
        rows, cols, vals = [], [], []
        for c,(i,j) in enumerate(edge_list):
            rows += [i,j]; cols += [c,c]; vals += [1.0,-1.0]
        return csr_matrix((vals,(rows,cols)), shape=(n,len(edge_list)))

    def _ensure_connected_w(self, w):
        if not self.ensure_connected:
            return w
        # lift MST weights (but keep below w_floor due to __init__ clamp)
        w[self._mst_idx] = np.maximum(w[self._mst_idx], self.mst_w)
        return w

    def _weighted_transvalue(self, w, full=False, active_idx=None):
        """
        Connected physics:
          - PHYS = ACTIVE ∪ MST  (ensures solvable L)
          - TV aggregates ONLY over ACTIVE edges (MST excluded).
        """
        # ACTIVE set (what we measure)
        if active_idx is None:
            if self.ensure_connected:
                w[self._mst_idx] = np.maximum(w[self._mst_idx], self.mst_w)
            act_mask = (w > self.w_floor)
            active = np.flatnonzero(act_mask)
        else:
            if self.ensure_connected:
                w[self._mst_idx] = np.maximum(w[self._mst_idx], self.mst_w)
            active = np.array(active_idx, dtype=int)

        # PHYSICS set = ACTIVE ∪ MST
        phys = np.unique(np.concatenate([active, self._mst_idx])) if self.ensure_connected else active
        if phys.size == 0:
            return 0.0

        # map ACTIVE indices into PHYS ordering
        phys_map = {int(k): t for t, k in enumerate(phys)}
        active_in_phys = np.array([phys_map[k] for k in active if k in phys_map], dtype=int)
        if active_in_phys.size == 0:
            return 0.0

        # assemble physics on PHYS edges
        el_phys = [self.edge_list[k] for k in phys]
        w_phys = w[phys]
        B = self._incidence(self.n, el_phys)
        W = csc_matrix((w_phys, (np.arange(len(el_phys)), np.arange(len(el_phys)))), shape=(len(el_phys), len(el_phys)))
        L = (B @ W @ B.T).tocsc()

        keep = np.arange(self.n-1)
        try:
            lu = splu(L[keep[:,None], keep])
        except RuntimeError:
            Lr = L[keep[:,None], keep] + 1e-9 * csc_matrix(np.eye(self.n-1))
            lu = splu(Lr)

        # scenarios
        P_full = self.P; S_all = P_full.shape[1]
        if full or (self.batch_scenarios is None) or (self.batch_scenarios >= S_all):
            P = P_full; q = self.quantile_q; use_cvar = False
        else:
            idx = self.rng.choice(S_all, size=self.batch_scenarios, replace=False)
            P = P_full[:, idx]; q = self.batch_quantile; use_cvar = True

        V_r = lu.solve(P[keep,:])
        V = np.vstack([V_r, np.zeros((1, P.shape[1]))])

        F_phys = W @ (B.T @ V)
        F_phys = F_phys.toarray() if hasattr(F_phys, "toarray") else np.asarray(F_phys)
        absF = np.abs(F_phys)

        absF_active = absF[active_in_phys, :]
        if use_cvar:
            nS = absF_active.shape[1]
            kth = int(np.ceil(nS * q))
            part = np.partition(absF_active, kth, axis=1)
            kappa = part[:, kth:].mean(axis=1)
        else:
            kappa = np.quantile(absF_active, q=q, axis=1)

        kappa_full = np.zeros(self.m, dtype=float)
        kappa_full[active] = kappa
        return float(np.sum(kappa_full * self.lengths * self.price))

    # -------- objective & sensitivities ----------
    def _build_weights(self, rho, it):
        rho_bar = rho  # no filtering
        # Projection only after V ramp / activation
        use_proj_now = self.use_projection and (it >= self._proj_on_it)
        if use_proj_now:
            beta = self._beta(it)
            rho_proj = (np.tanh(beta*self.eta) + np.tanh(beta*(rho_bar - self.eta))) / (
                        np.tanh(beta*self.eta) + np.tanh(beta*(1 - self.eta)) + 1e-12)
        else:
            rho_proj = rho_bar
        w = self.w_min + np.power(np.clip(rho_proj, 0, 1), self.p)
        w = self._ensure_connected_w(w)
        return w, rho_bar, rho_proj

    def _soft_count(self, rho_like):
        # Smooth “edge present” count; use rho_proj for crispness
        z = (rho_like - self.soft_count_tau) / max(1e-12, self.soft_count_alpha)
        sig = 1.0 / (1.0 + np.exp(-z))
        return float(sig.sum())

    def _objective(self, rho, it, full=False, active_idx=None):
        w, rho_bar, rho_proj = self._build_weights(rho, it)
        tv = self._weighted_transvalue(w, full=full, active_idx=active_idx)

        # Option A: capex penalty (length-weighted spend)
        if self.capex_lambda > 0.0:
            tv += self.capex_lambda * float(np.dot(self.lengths, rho_proj))

        # Option B: soft edge-count penalty (penalize only the excess)
        if self.mu_softcount > 0.0:
            S = self._soft_count(rho_proj)
            excess = max(0.0, S - self.K_soft)
            tv += self.mu_softcount * (excess * excess)

        return tv, w, rho_bar, rho_proj

    def _sensitivities_SPSA(self, rho, it, c=1e-3, avg=6):
        """
        SPSA gradient with the SAME active set policy used in the main loop:
        - Build Kcap from the *current* iteration and rho (use rho for ranking early).
        - Use that active_idx for both fp and fm evaluations.
        """
        # compute-only Kcap and active indices
        Kcap = self._kcap_schedule(it)
        order = np.argsort(-rho)  # could switch to -rho_proj when projection is on
        top_idx = order[:max(0, Kcap - len(self._mst_idx))]

        g = np.zeros_like(rho)
        for _ in range(avg):
            delta = np.random.choice((-1.0, 1.0), size=rho.size)
            rp = np.clip(rho + c * delta, 0.0, 1.0)
            rm = np.clip(rho - c * delta, 0.0, 1.0)
            fp, *_ = self._objective(rp, it, full=False, active_idx=top_idx)
            fm, *_ = self._objective(rm, it, full=False, active_idx=top_idx)
            g += (fp - fm) / (2.0 * c) * delta
        return g / max(1, avg)


    # -------- OC updates ----------
    def _oc_update(self, rho, grad, vol_frac, move):
        # Standard OC with unweighted "volume" sum(rho) = vol_frac * m
        dC = -grad
        dC = np.maximum(dC, 1e-12)
        l_lo, l_hi = 1e-12, 1e12
        Vsum = vol_frac * rho.size
        rho_new = rho.copy()
        for _ in range(60):
            lam = 0.5 * (l_lo + l_hi)
            x = rho * np.sqrt(dC / lam)
            x = np.minimum(rho + move, np.maximum(rho - move, x))
            x = np.clip(x, 0.0, 1.0)
            s = x.sum()
            if s > Vsum: l_lo = lam
            else:        l_hi = lam
            rho_new = x
            if abs(s - Vsum) < 1e-6 * rho.size:
                break
        return rho_new

    def _oc_update_lenweighted(self, rho, grad, budget, move, lengths):
        # OC with weighted "area" sum(lengths * rho) = budget
        dC = -grad
        dC = np.maximum(dC, 1e-12)
        l_lo, l_hi = 1e-12, 1e12
        rho_new = rho.copy()
        A = np.maximum(lengths, 1e-12)
        for _ in range(60):
            lam = 0.5 * (l_lo + l_hi)
            x = rho * np.sqrt(dC / (lam * A))
            x = np.minimum(rho + move, np.maximum(rho - move, x))
            x = np.clip(x, 0.0, 1.0)
            s = float(np.dot(A, x))
            if s > budget: l_lo = lam
            else:          l_hi = lam
            rho_new = x
            if abs(s - budget) < 1e-6 * budget:
                break
        return rho_new

    # -------- build exactly K' ----------
    def _kcap_schedule(self, it):
        """Cosine schedule for Kcap from K0 -> Kprime_final over [warmup_end, ramp_end]."""
        T = self.max_iters
        warmup_end = int(self.warmup_frac * T)
        ramp_end   = int(self.ramp_frac   * T)
        Kp = self.Kprime_final
        # Start from a generous cap, e.g., 5×K' but not more than all candidates
        K0 = int(min(self.m, max(self.n - 1, 5 * Kp)))
        if it <= warmup_end:
            return K0
        if it <= ramp_end and ramp_end > warmup_end:
            s = (it - warmup_end) / max(1, (ramp_end - warmup_end))  # [0,1]
            Kt = int(round(Kp + 0.5 * (K0 - Kp) * (1.0 + np.cos(np.pi * s))))
            return max(self.n - 1, Kt)
        return Kp

    def _build_kprime_adjacency(self, rho_proj, Kprime):
        n = self.n
        assert Kprime >= n-1, "K' must be at least n-1."
        # MST (precomputed)
        A = np.zeros((n, n), dtype=np.uint8)
        for k in self._mst_idx:
            i, j = self.edge_list[k]; A[i, j] = A[j, i] = 1
        # top-(K'-(n-1)) by rho_proj
        needed = Kprime - (n - 1)
        order = np.argsort(-rho_proj)
        added = 0
        for k in order:
            if k in self._mst_set: continue
            i, j = self.edge_list[k]
            if A[i, j] == 0:
                A[i, j] = A[j, i] = 1
                added += 1
                if added >= needed: break
        return A

    def _trans_from_adj(self, A):
        G = nx.from_numpy_array(A)
        if not nx.is_connected(G):
            raise RuntimeError("Adj not connected")
        edge_list = sorted(G.edges())
        B = self._incidence(A.shape[0], edge_list)
        L = (B @ B.T).tocsc()
        keep = np.arange(A.shape[0]-1)
        lu = splu(L[keep[:,None], keep])
        V_r = lu.solve(self.P[keep,:])
        V = np.vstack([V_r, np.zeros((1,self.P.shape[1]))])
        F = (B.T @ V)
        kappa = np.quantile(np.abs(F), q=self.quantile_q, axis=1)
        lengths = np.array([self.dist[i,j] for (i,j) in edge_list], dtype=float)
        return float(np.sum(kappa * lengths * self.price))

    def _greedy_prune(self, A):
        def non_bridge(adj):
            G = nx.from_numpy_array(adj)
            br = set((min(u,v),max(u,v)) for u,v in nx.bridges(G))
            edges = [(min(u,v),max(u,v)) for u,v in G.edges()]
            return [e for e in edges if e not in br]

        def score_edges(adj):
            G = nx.from_numpy_array(adj)
            edge_list = sorted(G.edges())
            n = adj.shape[0]
            B = self._incidence(n, edge_list)
            L = (B @ B.T).tocsc()
            keep = np.arange(n-1)
            lu = splu(L[keep[:,None], keep])
            V_r = lu.solve(self.P[keep,:])
            V = np.vstack([V_r, np.zeros((1,self.P.shape[1]))])
            F = (B.T @ V)
            kappa = np.quantile(np.abs(F), q=self.batch_quantile, axis=1)
            lengths = np.array([self.dist[i,j] for (i,j) in edge_list], dtype=float)
            return { (min(i,j),max(i,j)): kappa[t]*lengths[t] for t,(i,j) in enumerate(edge_list) }

        target = self.target_edges
        cur = int(A.sum()//2)
        if self.verbose: print(f"[Prune] start {cur} -> {target}")
        if cur <= target: return A

        if self.prune_mode == "EXACT":
            while cur > target:
                cand = non_bridge(A)
                best_e, best_val = None, None
                for (u,v) in cand:
                    A[u,v]=A[v,u]=0
                    if nx.is_connected(nx.from_numpy_array(A)):
                        val = self._trans_from_adj(A)
                        if best_val is None or val < best_val:
                            best_val, best_e = val, (u,v)
                    A[u,v]=A[v,u]=1
                if best_e is None: break
                A[best_e[0],best_e[1]] = A[best_e[1],best_e[0]] = 0
                cur -= 1
                if self.verbose: print(f"[Prune] EXACT removed {best_e}, edges={cur}, TV≈{best_val:.3g}")
            return A

        # FAST
        while cur > target:
            scores = score_edges(A)
            cand = non_bridge(A)
            if not cand: break
            cand.sort(key=lambda e: scores.get(e, np.inf))
            batch = min(self.prune_batch, cur - target)
            removed = 0
            for e in cand:
                if removed >= batch: break
                u,v = e
                A[u,v]=A[v,u]=0
                if nx.is_connected(nx.from_numpy_array(A)):
                    removed += 1
                else:
                    A[u,v]=A[v,u]=1
            cur = int(A.sum()//2)
            if self.verbose: print(f"[Prune] FAST removed={removed}, edges={cur}")
            if removed == 0:
                if self.verbose: print("[Prune] FAST stuck; switching to EXACT.")
                self.prune_mode = "EXACT"
        return A

    # -------- main ----------
    def optimize(self):
        rho = self.rho.copy()
        best = None

        tv, *_ = self._objective(rho, it=0, full=False)
        best = (rho.copy(), tv)
        if self.verbose:
            print(f"[Init] TV≈{tv:.3g}  m={self.m}  V={self.V:.3f}  p={self.p:.2f}")

        ramp_end = int(self.ramp_frac * self.max_iters)

        for it in range(1, self.max_iters+1):
            # smooth schedules
            self.V = float(np.clip(self._V_schedule(it), self.V_min, self.V_max))
            self.p = float(self._p_schedule(it))

            # compute-only active set cap
            Kcap = self._kcap_schedule(it)
            order = np.argsort(-rho)  # could switch to -rho_proj late for stability
            top_idx = order[:max(0, Kcap - len(self._mst_idx))]

            full = (it % self.checkpoint_every == 0)
            tv, w, rho_bar, rho_proj = self._objective(rho, it=it, full=full, active_idx=top_idx)

            if tv < best[1]:
                best = (rho.copy(), tv)

            # tiny late controller on V (only after V-ramp completes) — cosmetic
            if it >= ramp_end:
                edges_now = int((rho_proj > self.thr_thresh).sum())
                err = (edges_now - self.Kprime_final) / max(1, self.Kprime_final)
                V_lo, V_hi = 0.8*self.V_target, 1.2*self.V_target
                self.V = float(np.clip(self.V * (1.0 - 0.02 * err),
                                       max(self.V_min, V_lo),
                                       min(self.V_max, V_hi)))

            # sensitivities (batch) + OC update (exact "volume" or length budget)
            g = self._sensitivities_SPSA(rho, it, c=1e-3, avg=6)
            if self.length_budget is not None:
                rho = self._oc_update_lenweighted(rho, g, budget=self.length_budget,
                                                  move=self.move, lengths=self.lengths)
            else:
                rho = self._oc_update(rho, g, vol_frac=self.V, move=self.move)

            # periodic hard sparsification (prevents tiny values from lingering)
            if (it % self.hard_zero_every) == 0:
                rho[rho < self.rho_floor] = 0.0

            # log
            S_soft = self._soft_count(rho_proj)
            approx_edges = int((rho_proj > self.thr_thresh).sum())
            self.hist["iter"].append(it)
            self.hist["V"].append(self.V)
            self.hist["p"].append(self.p)
            self.hist["edges_est"].append(approx_edges)
            self.hist["S_soft"].append(S_soft)
            self.hist["Kcap"].append(Kcap)

            if self.verbose and (it % 10 == 0):
                mode = "FULL" if full else "BATCH"
                print(f"[Iter {it:04d}] TV≈{tv:.3g}  edges~{approx_edges}  S_soft={S_soft:.1f}  "
                      f"V={self.V:.4f}  p={self.p:.2f}  Kcap={Kcap}  mode={mode}")

        # Use best rho (design) and final projection (now ON with β_end if enabled)
        rho_design = best[0]
        if self.use_projection:
            beta_final = self.beta_end
            rho_proj = (np.tanh(beta_final*self.eta) + np.tanh(beta_final*(rho_design - self.eta))) / (
                       np.tanh(beta_final*self.eta) + np.tanh(beta_final*(1 - self.eta)) + 1e-12)
        else:
            rho_proj = rho_design

        # Build exactly K' (~1.1*K) to guarantee prune headroom
        A0 = self._build_kprime_adjacency(rho_proj, self.Kprime_final)
        if self.verbose:
            print(f"[Build] edges={int(A0.sum()//2)} (K'={self.Kprime_final}) before prune")

        # Prune to K
        A_final = self._greedy_prune(A0)
        tv_final = self._trans_from_adj(A_final)
        if self.verbose:
            print(f"[Final] edges={int(A_final.sum()//2)}  TransValue={tv_final:.3g}")

        info = {
            "TransValue": tv_final,
            "Kprime_final": self.Kprime_final,
            "V_target": self.V_target,
            "hist": self.hist
        }
        return rho_design, rho_proj, A_final, info
