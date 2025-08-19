import numpy as np
import scipy.sparse as sparse
from types import SimpleNamespace as StepState
from flowtracing.source_to_sink import source_to_sink_class
import time


class flowtracing_class:
    def __init__(self, init, layout, link_network, n_bins=50, verbose=True):
        flowtrace_model = q_partition_method(init, layout, link_network, n_bins=n_bins, verbose=verbose)
        self.avg_export, self.link_prop_theory = flowtrace_model.run()


class q_partition_method:
    def __init__(self, init, layout, link_network, n_bins=50, verbose=False):
        # ---------- core data ----------
        self._math_dtype = np.float32

        self.Injections = layout.misc['injection'].astype(self._math_dtype, copy=False)  # (T, n)
        self.incidence = sparse.csr_matrix(link_network.layout['incidence'])
        # keep a float32 copy to avoid cast-per-step
        self.incidence32 = self.incidence.astype(self._math_dtype)

        self.adjacency = link_network.layout['adjacency']  # used only for .nonzero() in source_to_sink_class

        self.FlowPattern = link_network.flows['FlowPattern'].astype(self._math_dtype, copy=False)  # (T, n_l)
        self.t_total = int(init.t)
        self.kappa_trans = link_network.kappa.astype(self._math_dtype, copy=False)                 # (n_l,)
        self.n = int(init.n)
        self.n_l = int(link_network.n_l)

        # ---------- binning ----------
        self.n_bins = int(n_bins)
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1, dtype=self._math_dtype)

        self.abs_flow = np.abs(self.FlowPattern)  # (T, n_l) float32
        denom = np.where(self.kappa_trans == 0, 1, self.kappa_trans).astype(self._math_dtype, copy=False)
        self.norm_flow = np.clip(self.abs_flow / denom, 0, 1 - np.float32(1e-8))                  # (T, n_l) float32
        self.bin_indices = (self.norm_flow * (self.n_bins - 1)).astype(np.int32)                  # (T, n_l)

        # ---------- extras ----------
        self.load = layout.generation.get('load', init.load).astype(self._math_dtype, copy=False)
        self.generation = {
            k: v.astype(self._math_dtype, copy=False) for k, v in layout.generation.items()
        }
        self.verbose = bool(verbose)

    # --------- per-step state ---------

    def Initialize_time_step(self, t):
        flows_t = self.FlowPattern[t, :]  # (n_l,) float32

        # weighted incidence in float32
        F = self.incidence32.T.multiply(flows_t[:, None]).tocsr()

        # split by sign (still sparse float32)
        F_out = F.multiply(F > 0)
        F_in  = (-F).multiply(F < 0)

        # injections float32
        P = self.Injections[t, :]
        P_plus  = np.clip(P, 0, None)
        P_minus = np.clip(-P, 0, None)

        # totals float32
        F_out_total = np.asarray(F_out.sum(axis=0), dtype=self._math_dtype).ravel()
        F_in_total  = np.asarray(F_in.sum(axis=0),  dtype=self._math_dtype).ravel()

        return StepState(
            t=t, F=F, F_out=F_out, F_in=F_in,
            F_out_total=F_out_total, F_in_total=F_in_total,
            P=P, P_plus=P_plus, P_minus=P_minus
        )

    # --------- forward (source → sink) ---------

    def run_source_to_sink(self, s):
        t0 = time.perf_counter()
        model = source_to_sink_class(self.adjacency, s.F_in, s.F_out, s.F_out_total, s.P_minus, s.P_plus)

        # walk
        tw0 = time.perf_counter()
        walk = model.walk()
        t_walk = time.perf_counter() - tw0

        # q_nn (may be float64 internally; cast once to float32 for accumulations)
        tq0 = time.perf_counter()
        q_nn = model.q_nn(walk).astype(self._math_dtype, copy=False)
        t_qnn = time.perf_counter() - tq0

        # partition_q (cast outputs once)
        tp0 = time.perf_counter()
        q_wind, q_solar, q_backup = model.partition_q(s.t, q_nn, self.generation, self.load[s.t, :])
        q_wind  = q_wind.astype(self._math_dtype, copy=False)
        q_solar = q_solar.astype(self._math_dtype, copy=False)
        q_backup= q_backup.astype(self._math_dtype, copy=False)
        t_partition = time.perf_counter() - tp0

        t_total = time.perf_counter() - t0
        return q_nn, q_wind, q_solar, q_backup, (t_total, t_walk, t_qnn, t_partition)

    # --------- reverse (sink → source), return valid (links, sources) ---------

    def run_sink_to_source(self, s):
        """
        Reverse pass that returns:
          - q_nn_rev: (N,N) float32 for fast accumulation
          - links_valid: (K,) valid link indices
          - sources_valid: (K,) corresponding source nodes
          - timings: (total, walk, q_nn_only)
        """
        t0 = time.perf_counter()
        #Swap P_plus and P_minus along with F_in and F_out for sink to source walk
        model = source_to_sink_class(self.adjacency, s.F_out, s.F_in, s.F_in_total, s.P_plus, s.P_minus)

        # walk
        tw0 = time.perf_counter()
        walk = model.walk()
        t_walk = time.perf_counter() - tw0

        # q_nn (reverse responsibility) as float32
        tq0 = time.perf_counter()
        q_nn_rev = model.q_nn(walk).astype(self._math_dtype, copy=False)
        t_qnn_only = time.perf_counter() - tq0

        # valid (link, source) pairs
        link_walk = walk['link_wise']
        valid = link_walk[:, 0] != -1
        links_valid   = link_walk[valid, 0].astype(np.int32, copy=False)
        sources_valid = link_walk[valid, 1].astype(np.int32, copy=False)

        t_total = time.perf_counter() - t0
        return q_nn_rev, links_valid, sources_valid, (t_total, t_walk, t_qnn_only)

    # --------- accumulate using valid (link, source) pairs ---------

    def CondAvg_accumulate_pairs(self, q_nn32, links, sources, sum_q_bins, count_q_link, bin_ind):
        """
        Accumulate using only valid (links, sources) pairs:
          - Group links by bin with argsort (one pass).
          - For each group, add q_nn32[:, sources_group] into the bin buffer.

        q_nn32      : (N, N) float32
        links       : (K,)   int32
        sources     : (K,)   int32
        sum_q_bins  : list of B arrays, each (N, L) float32
        count_q_link: (L, B) int32
        bin_ind     : (L,)   int32
        """
        if links.size == 0:
            return sum_q_bins, count_q_link

        bins = bin_ind[links]
        order = np.argsort(bins, kind='stable')
        bins_sorted = bins[order]
        links_sorted = links[order]
        sources_sorted = sources[order]

        boundaries = np.flatnonzero(np.diff(bins_sorted)) + 1
        starts = np.r_[0, boundaries]
        ends   = np.r_[boundaries, links_sorted.size]

        for start, end in zip(starts, ends):
            b = int(bins_sorted[start])
            idx_seg = links_sorted[start:end]
            src_seg = sources_sorted[start:end]
            # both sides (N, len(seg)), float32
            sum_q_bins[b][:, idx_seg] += q_nn32[:, src_seg]
            count_q_link[idx_seg, b]  += 1
        return sum_q_bins, count_q_link

    # --------- main loop ---------

    def run(self):
        n, n_l, T = self.n, self.n_l, self.t_total

        # forward accumulators (float32)
        total        = np.zeros((n, n), dtype=self._math_dtype)
        wind_total   = np.zeros_like(total)
        solar_total  = np.zeros_like(total)
        backup_total = np.zeros_like(total)

        # per-bin buffers (float32) and counts (int32)
        sum_q_bins   = [np.zeros((n, n_l), dtype=self._math_dtype) for _ in range(self.n_bins)]
        count_q_link = np.zeros((n_l, self.n_bins), dtype=np.int32)

        # timers (float64 timers are fine)
        t_init, t_srcsink, t_sinks, t_condavg = 0.0, 0.0, 0.0, 0.0
        t_walk_src, t_qnn_src, t_partition = 0.0, 0.0, 0.0
        t_walk_sink, t_qnn_sink = 0.0, 0.0

        for t in range(T):
            # progress print every 1000 iterations
            if (t % 100 == 0):
                print(f"[FlowTracing] t={t}/{T}")

            # initialize step
            t0 = time.perf_counter()
            s = self.Initialize_time_step(t)
            t_init += time.perf_counter() - t0

            # forward pass
            q_nn_fwd, q_wind, q_solar, q_backup, (ts_total, ts_walk, ts_qnn, ts_part) = self.run_source_to_sink(s)
            total        += q_nn_fwd * s.P_minus
            wind_total   += q_wind   * s.P_minus
            solar_total  += q_solar  * s.P_minus
            backup_total += q_backup * s.P_minus
            t_srcsink   += ts_total
            t_walk_src  += ts_walk
            t_qnn_src   += ts_qnn
            t_partition += ts_part

            # reverse pass (no q_ln build)
            q_nn_rev32, links_valid, sources_valid, (tk_total, tk_walk, tk_qnn_only) = self.run_sink_to_source(s)
            t_sinks     += tk_total
            t_walk_sink += tk_walk
            t_qnn_sink  += tk_qnn_only

            # accumulate (group by bin once)
            t0 = time.perf_counter()
            sum_q_bins, count_q_link = self.CondAvg_accumulate_pairs(
                q_nn_rev32, links_valid, sources_valid, sum_q_bins, count_q_link, self.bin_indices[t, :]
            )
            t_condavg += time.perf_counter() - t0

        # pack per-bin buffers into the expected (N, L, B) tensor (still float32)
        sum_q_link = np.stack(sum_q_bins, axis=2)

        # final timing breakdown
        if self.verbose:
            print("Timing results (seconds):")
            print(f"  Initialize_time_step: {t_init:.3f}")
            print(f"  run_source_to_sink:   {t_srcsink:.3f}")
            print(f"    - walk:             {t_walk_src:.3f}")
            print(f"    - q_nn:             {t_qnn_src:.3f}")
            print(f"    - partition_q:      {t_partition:.3f}")
            print(f"  run_sink_to_source:   {t_sinks:.3f}")
            print(f"    - walk:             {t_walk_sink:.3f}")
            print(f"    - q_nn (rev):       {t_qnn_sink:.3f}")
            print(f"  CondAvg_accumulate:   {t_condavg:.3f}")
            total_time = t_init + t_srcsink + t_sinks + t_condavg
            print(f"  Total loop:           {total_time:.3f} "
                  f"(avg {total_time/T*1000:.2f} ms/step)")

        # outputs (stay float32/int32)
        link_prop_theory = {
            'q_link_sum':   sum_q_link,         # float32
            'q_link_count': count_q_link,       # int32
            'n_bins':       self.n_bins,
            'abs_flow':     self.abs_flow,      # float32
            'norm_flow':    self.norm_flow,     # float32
            'bin_edges':    self.bin_edges,     # float32
        }
        avg_export = {
            'total':  total / T,                # float32
            'wind':   wind_total / T,           # float32
            'solar':  solar_total / T,          # float32
            'backup': backup_total / T,         # float32
        }
        return avg_export, link_prop_theory