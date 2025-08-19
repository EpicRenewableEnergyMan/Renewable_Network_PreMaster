import numpy as np
class used_kappa_class:
    def __init__(self,init,layout,link_layout,flowtrace):
        kappa_model = use_own_ren_first(init, layout, link_layout, flowtrace)
        self.kappa = kappa_model.run()

class use_own_ren_first:
    def __init__(self,init,layout,link_layout,flowtrace):
        self.generation = layout.generation
        self.kappa = layout.kappa
        self.avg_export = flowtrace.avg_export
        self.n = self.kappa['wind'].shape[0]
        self.injection = layout.misc['injection']
        self.length = link_layout.layout['length']

        link = flowtrace.link_prop_theory
        self.q_link_sum = link['q_link_sum']
        self.q_link_count = link['q_link_count']
        self.n_bins = link['n_bins']
        self.abs_flow = link['abs_flow']

    def kappa_wind_solar_backup(self):
        wind_gen = self.generation['wind']
        solar_gen = self.generation['solar']
        backup_gen = self.generation['backup']
        curtailment = self.generation['curtailment']

        wind_kappa = self.kappa['wind']
        solar_kappa = self.kappa['solar']
        backup_kappa = self.kappa['backup']

        avg_export_wind = self.avg_export['wind']
        avg_export_solar = self.avg_export['solar']
        avg_export_backup = self.avg_export['backup']

        ####################
        ren_gen = wind_gen + solar_gen

        wind_fraction = wind_gen / ren_gen
        wind_fraction[np.isnan(wind_fraction)] = 0

        solar_fraction = solar_gen / ren_gen
        solar_fraction[np.isnan(solar_fraction)] = 0

        curtailment_local = curtailment.copy()
        curtailment_local[self.injection<0] = 0

        avg_wind_curtailment_local = np.multiply(curtailment_local,wind_fraction).mean(axis=0)
        avg_solar_curtailment_local = np.multiply(curtailment_local,solar_fraction).mean(axis=0)

        def get_kappa_used(kappa_installed,avg_export,avg_generation,avg_curtailment,N):
            kappa_used = np.zeros((N,N))
            for n in range(N):
                for m in range(N):
                    if m!=n:
                        kappa_used[m,n] = (avg_export[m,n] / (avg_generation[m]-avg_curtailment[m])) * kappa_installed[m]
            np.fill_diagonal(kappa_used,(kappa_installed - kappa_used.sum(axis=1)))
            return kappa_used

        kappa_used_wind = get_kappa_used(wind_kappa,avg_export_wind,wind_gen.mean(axis=0),avg_wind_curtailment_local,self.n)
        kappa_used_solar = get_kappa_used(solar_kappa,avg_export_solar,solar_gen.mean(axis=0),avg_solar_curtailment_local,self.n)
        kappa_used_backup = get_kappa_used(backup_kappa,avg_export_backup,backup_gen.mean(axis=0),np.zeros(self.n),self.n)
        return kappa_used_wind.sum(axis=0),kappa_used_solar.sum(axis=0),kappa_used_backup.sum(axis=0)



    def kappa_trans(self, q=0.99):
        import numpy as np

        sum_q    = self.q_link_sum
        count_q  = self.q_link_count
        n_bins   = self.n_bins
        abs_flow = self.abs_flow

        N, L, B = sum_q.shape
        assert count_q.shape == (L, B) and B == n_bins


        kappa_l = np.quantile(abs_flow, q=q, axis=0).astype(np.float32)   # (L,)

        dkappa = kappa_l / n_bins                                             # (L,)

        # 2) empirical pdf from the counts we actually accumulated
        count = count_q.astype(np.float32)                                    # (L, B)
        T_l   = np.maximum(count.sum(axis=1, keepdims=True), 1.0)             # (L, 1)
        P     = (count / T_l).T                                               # (B, L)
        P_c   = np.vstack([P[:k, :].sum(axis=0) for k in range(B)]).astype(np.float32)  # (B, L)

        # 3) conditional averages  ⟨c_ln | bin⟩  with correct per-(l,k) renorm
        denom = np.where(count_q > 0, count_q, 1.0).astype(np.float32)        # (L, B)
        cond_avg = (sum_q / denom[None, :, :]).astype(np.float32)             # (N, L, B)
        cond_avg = np.nan_to_num(cond_avg, nan=0.0)

        # # --- FIX: renormalize per (l,k) so shares sum to 1 where count>0, else 0 ---
        row_sum = cond_avg.sum(axis=0)                                        # (L, B)
        row_sum_safe = np.where(count_q > 0, np.maximum(row_sum, 1e-12), 1.0) # (L, B)
        cond_avg = cond_avg / row_sum_safe[None, :, :]                        # (N, L, B)
        cond_avg = cond_avg * (count_q > 0)[None, :, :]                       # zero empty bins
        # ---------------------------------------------------------------------------


        # 4) integrate per link (discrete Hoersch/Schäfer)
        K_ln_T = np.zeros((N, L), dtype=np.float32)
        for l in range(L):
            denom_tail = np.clip(1.0 - P_c[:, l], 1e-8, None)                # (B,)
            integrand  = cond_avg[:, l, :] * P[:, l][None, :]                # (N, B)
            rev_cumsum = np.cumsum(integrand[:, ::-1], axis=1)[:, ::-1]      # Σ_{b≥k}
            weights    = (dkappa[l] / denom_tail)[None, :]                   # (1, B)
            K_ln_T[:, l] = (rev_cumsum * weights).sum(axis=1)

        # Identity must hold now
        #assert np.allclose(K_ln_T.sum(axis=0), kappa_l, rtol=1e-6, atol=1e-8), "Per-link κ sum mismatch."

        #Now the full "volume"
        K_ln_T_Volume = K_ln_T * self.length

        return K_ln_T.sum(axis=0), K_ln_T_Volume.sum(axis=1)


    def run(self):
        wind,solar,backup = self.kappa_wind_solar_backup()
        trans,trans_volume = self.kappa_trans()

        kappa = {
            'wind':wind,
            'solar':solar,
            'backup': backup,
            'trans':trans,
            'trans_volume':trans_volume,
            }
        return kappa