import numpy as np
from collections import namedtuple

class source_to_sink_class:
    def __init__(self, adjacency, F_in, F_out, F_out_total, P_minus, P_plus):
        self.adjacency   = adjacency
        self.F_in_csc    = F_in.tocsc()    # cols = sinks → (links, weights)
        self.F_out_csr   = F_out.tocsr()   # rows = links → (source node)
        self.F_out_total = P_minus * 0 + F_out_total   # keep shape/dtype
        self.P_minus     = P_minus
        self.P_plus      = P_plus

        self.l, self.n = F_in.shape

        # ---- Precompute link → source node (fast path assumes 1 nnz per link row) ----
        r0 = self.F_out_csr.indptr[:-1]
        r1 = self.F_out_csr.indptr[1:]
        row_nnz = (r1 - r0)

        # If any link row has ≠1 nnz, fall back per-row (rare). Keeps correctness.
        if np.any(row_nnz != 1):
            src_idx = np.empty(self.l, dtype=np.int32)
            for link in range(self.l):
                s0, s1 = self.F_out_csr.indptr[link], self.F_out_csr.indptr[link + 1]
                idx = self.F_out_csr.indices[s0:s1]
                # choose the first as “source”; if multiple
                src_idx[link] = idx[0] if idx.size else -1
        else:
            src_idx = self.F_out_csr.indices[r0].astype(np.int32)

        self._link_src = src_idx  # shape (L,), int32

        # cache csc column pointers/indices/data for F_in (avoid attribute lookups)
        self._c_indptr = self.F_in_csc.indptr
        self._c_index  = self.F_in_csc.indices
        self._c_data   = self.F_in_csc.data

        # dtype strategy: do the math in float32 to cut bandwidth, return float32
        self._math_dtype = np.float32
        self._ret_dtype  = np.result_type(np.float32, P_minus.dtype, P_plus.dtype)

    # ----------------------------------------------------------------------------------

    def walk(self):  # source → sink
        WalkNode = namedtuple('WalkNode', ['links', 'sources', 'sinks'])

        n_link = 0
        sub_step_intervals = []

        # int32 for all index structures
        walk_link = np.full((self.l, 3), -1, dtype=np.int32)
        link_n, source_node_n, sink_node_n = [], [], []

        # Unknowns: number of incoming positive links per node (column nnz)
        Unknowns = self.F_in_csc.getnnz(axis=0).astype(np.int32)  # (n,)
        initial_source_nodes = np.where(Unknowns == 0)[0].astype(np.int32)

        # decrement neighbors of initial sources
        neighbors = self.adjacency[:, initial_source_nodes].nonzero()[0]
        for nb in neighbors:
            Unknowns[nb] -= 1
        Unknowns[initial_source_nodes] = -1

        max_iter = 1000
        iter_count = 0
        while Unknowns.sum(dtype=np.int64) != -self.n and iter_count < max_iter:
            sink_nodes = np.where(Unknowns == 0)[0].astype(np.int32)
            if sink_nodes.size == 0:
                break

            for sink in sink_nodes:
                c0, c1 = self._c_indptr[sink], self._c_indptr[sink + 1]
                links_from_sources = self._c_index[c0:c1]          # (k,)
                if links_from_sources.size:
                    sources = self._link_src[links_from_sources]   # (k,)
                else:
                    sources = np.empty(0, dtype=np.int32)

                link_n.append(links_from_sources)
                source_node_n.append(sources)
                sink_node_n.append(sink)

                # write consecutive block (preserve order/pairing)
                k = links_from_sources.size
                if k:
                    sl = slice(n_link, n_link + k)
                    walk_link[sl, 0] = links_from_sources
                    walk_link[sl, 1] = sources
                    walk_link[sl, 2] = sink
                    n_link += k

            sub_step_intervals.append(0)

            neighbors = self.adjacency[:, sink_nodes].nonzero()[0]
            for nb in neighbors:
                if Unknowns[nb] != -1:
                    Unknowns[nb] -= 1
            Unknowns[sink_nodes] -= 1
            iter_count += 1

        if iter_count >= max_iter:
            return 'NoFlow'

        walk_node = [WalkNode(link_n, source_node_n, sink_node_n)]
        return {
            'link_wise': walk_link,            # keep full shape with -1 padding
            'node_wise': walk_node,
            'substep_intervals': sub_step_intervals,
            'absolute_source_nodes': initial_source_nodes
        }

    # ----------------------------------------------------------------------------------

    def q_nn(self, walk):
        sinks = walk['node_wise'][0].sinks

        # compute in float32, return float64
        q = np.zeros((self.n, self.n), dtype=self._math_dtype)

        # absolute sources: unit responsibility for themselves
        abs_src = walk['absolute_source_nodes']
        q[abs_src, abs_src] = 1.0

        # constant denom per sink
        P_out = (self.P_minus + self.F_out_total).astype(self._math_dtype, copy=False)
        P_plus32 = self.P_plus.astype(self._math_dtype, copy=False)

        for sink in sinks:
            c0, c1 = self._c_indptr[sink], self._c_indptr[sink + 1]
            links   = self._c_index[c0:c1]
            weights = self._c_data[c0:c1].astype(self._math_dtype, copy=False)

            # incoming responsibility from upstream sources via links
            if links.size:
                sources = self._link_src[links]
                P_in_j = q[:, sources] @ weights   # (n,)
            else:
                P_in_j = np.zeros(self.n, dtype=self._math_dtype)

            # add self-provision at the sink itself
            P_in_j[sink] += P_plus32[sink]

            denom = P_out[sink] if P_out[sink] != 0 else self._math_dtype(1.0)
            q[:, sink] = P_in_j / denom

        # return dtype matches your previous (float64 in practice)
        return q.astype(self._ret_dtype, copy=False)

    # ----------------------------------------------------------------------------------

    def partition_q(self, t, q, generation, load):
        # operate in-place-style on float32 views to cut bandwidth, then upcast on return
        q32 = q.astype(self._math_dtype, copy=False)

        wind   = generation['wind'][t, :].astype(self._math_dtype, copy=False)
        solar  = generation['solar'][t, :].astype(self._math_dtype, copy=False)
        backup = generation['backup'][t, :].astype(self._math_dtype, copy=False)
        curt   = generation['curtailment'][t, :].astype(self._math_dtype, copy=False)
        ren    = generation['ren'][t, :].astype(self._math_dtype, copy=False)
        load32 = load.astype(self._math_dtype, copy=False)

        case_A = (ren > load32) & (curt > 0)
        case_B = (ren < load32) & (backup > 0)
        case_C = (ren > load32) & (backup > 0)

        q_wind  = np.zeros_like(q32)
        q_solar = np.zeros_like(q32)
        q_back  = np.zeros_like(q32)

        # avoid repeated division by building safe denominators
        ren_A   = ren[case_A, None]
        wind_A  = wind[case_A, None]
        solar_A = solar[case_A, None]

        ren_C   = ren[case_C, None]
        load_C  = load32[case_C, None]
        back_C  = backup[case_C, None]
        safe_C  = back_C + ren_C - load_C
        safe_C[safe_C == 0] = 1

        q_wind[case_A, :]  = q32[case_A, :] * (wind_A / ren_A)
        q_solar[case_A, :] = q32[case_A, :] * (solar_A / ren_A)

        scale_C = (ren_C - load_C) / safe_C
        q_wind[case_C, :]  = q32[case_C, :] * (wind[case_C, None]  / ren_C) * scale_C
        q_solar[case_C, :] = q32[case_C, :] * (solar[case_C, None] / ren_C) * scale_C

        q_back[case_B, :]  = q32[case_B, :]
        q_back[case_C, :]  = q32[case_C, :] * (back_C / safe_C)

        # return in original dtype
        return (q_wind.astype(self._ret_dtype, copy=False),
                q_solar.astype(self._ret_dtype, copy=False),
                q_back.astype(self._ret_dtype, copy=False))

    # ----------------------------------------------------------------------------------

    def q_ln(self, walk, q_nn):
        """
        Build q_ln (L x N): for each valid link l, row l = q_nn[:, source(l)]^T.
        - Reuses an internal float32 buffer to avoid per-call allocations.
        - Uses q_nn.T as a view (no extra transpose copy).
        """
        link_walk = walk['link_wise']
        valid = link_walk[:, 0] != -1
        links = link_walk[valid, 0]
        # robust precomputed mapping link -> source node (built in __init__)
        sources = self._link_src[links]

        # lazily allocate reusable buffer (float32 for bandwidth)
        buf = getattr(self, "_q_ln_buf", None)
        if (buf is None) or (buf.shape != (self.l, self.n)) or (buf.dtype != self._math_dtype):
            self._q_ln_buf = np.empty((self.l, self.n), dtype=self._math_dtype)
            buf = self._q_ln_buf

        # zero only once per call (cheaper than fresh allocation)
        buf.fill(0)

        # use a transpose view of q_nn to avoid constructing an intermediate (n,k) copy
        qT = q_nn.T  # view; no data copy
        # write selected rows in one shot
        buf[links, :] = qT[sources, :].astype(self._math_dtype, copy=False)

        # upcast for return to match previous external dtype contract
        return buf.astype(self._ret_dtype, copy=False)
