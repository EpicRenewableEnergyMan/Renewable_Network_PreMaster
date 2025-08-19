import numpy as np

class LCOE_PostProcessor:

    def __init__(self,rn,installed_trans_volume_method='load_based',prices_override=None):
        self.P = self.prices(prices_override)
        self.n = rn.init.n
        self.load = rn.init.load
        self.backup_generation = rn.layout.generation['backup']
        self.installed_kappa = rn.layout.kappa
        self.used_kappa = rn.used_kappa.kappa
        self.network = rn.network

        self.installed_kappa_volume = self.installed_trans_volume(installed_trans_volume_method)
        rn.layout.kappa.update({"trans_volume": self.installed_kappa_volume})


    # ---------- Prep work ----------
    def prices(self,prices_override=None):
        factor = 1000
        base = {
            # Capital expenditure
            "CapexTrans": 400,  # [EURO/(MW km)]
            "CapexWind": 1623 * factor, # [EURO/(MW)]
            "CapexSolar": 762.5 * factor, # [EURO/(MW)]
            "CapexBackup": 880 * factor, # [EURO/(MW)]
            # Operational expenditure
            "OpexWind": 25.355 * factor,  # [EURO/(MW year)]
            "OpexSolar": 17.24 * factor, # [EURO/(MW year)]
            "OpexBackup": 29.04 * factor, # [EURO/(MW year)]
            "OpexBackupVar": 20.1, # Variable expenditure per unit fuel [EURO/(MWh)]
            "OpexTrans": 8, # [EURO/(MW km year)]
            "etabackup": None, # Thermal effeciency of fuel for backup [MW_out/MW_th]
            # Lifetime [years]
            "LifeWind": 27,
            "LifeSolar": 30,
            "LifeBackup": 25,
            "LifeTrans": 40,
            # Rate of return
            "RateOfReturn": 0.04
        }  # Calayouts prices dict from function
        if prices_override:
            base.update(prices_override)
        return base

    def installed_trans_volume(self, method):
        if method == 'load_based':
            return self.load_based()
        if method == 'half_link':
            return self.half_link()

    def load_based(self):
        length = np.asarray(self.network.layout['length'], dtype=float)
        return self.load.sum(axis=0)*self.load.sum()*(self.installed_kappa['trans']*length).sum()

    def half_link(self):
        edges  = np.asarray(self.network.layout['edge_list'], dtype=int)  # shape (n_links, 2)
        length = np.asarray(self.network.layout['length'], dtype=float)
        # MW·km per link
        link_volume = length * self.installed_kappa['trans']
        nodal_volume = np.zeros(self.n, dtype=float)
        np.add.at(nodal_volume, edges[:, 0], link_volume * 0.5)
        np.add.at(nodal_volume, edges[:, 1], link_volume * 0.5)

        return nodal_volume





    ##############################################################

    def _lcoe_component(self, capex, opex, r, life, load_mwh_year):

        n_nodes = self.n

        # ensure arrays
        capex = np.asarray(capex, dtype=float)
        opex  = np.asarray(opex,  dtype=float)

        # broadcast scalars to nodal arrays if needed
        if capex.ndim == 0:
            capex = np.full(n_nodes, capex, dtype=float)
        if opex.ndim == 0:
            opex = np.full(n_nodes, opex, dtype=float)

        # ---- Denominator: discounted load (build Intermediate like old script) ----
        Intermediate = np.zeros((n_nodes, life), dtype=float)
        for i in range(1, life + 1):
            Intermediate[:, i - 1] = load_mwh_year / ((1.0 + r) ** i)
        load_denom = Intermediate.sum(axis=1)  # (n_nodes,)

        # ---- Numerator: CapEx + discounted OpEx (loop exactly like before) ----
        opex_disc = np.zeros(n_nodes, dtype=float)
        for i in range(1, life + 1):
            opex_disc += opex / ((1.0 + r) ** i)

        value = capex + opex_disc  # (n_nodes,)

        # safe divide
        out = np.full(n_nodes, np.nan, dtype=float)
        mask = load_denom > 0
        out[mask] = value[mask] / load_denom[mask]
        return out

    ###############################################################

    def lcoe_eu(self,kappa):
        p = self.P
        r = p['RateOfReturn']
        load = self.load.sum()
        backup_gen = self.backup_generation.sum()

        kappa_wind = kappa['wind'].sum()
        kappa_solar = kappa['solar'].sum()
        kappa_backup = kappa['backup'].sum()
        kappa_trans = kappa['trans_volume'].sum()

        lcoe_wind  = self._lcoe_component(p['CapexWind']  *kappa_wind,   p['OpexWind']  *kappa_wind,                                   r, p['LifeWind'],   load)
        lcoe_solar = self._lcoe_component(p['CapexSolar'] *kappa_solar,  p['OpexSolar'] *kappa_solar,                                  r, p['LifeSolar'],  load)
        lcoe_backup= self._lcoe_component(p['CapexBackup']*kappa_backup, p['OpexBackup']*kappa_backup + p['OpexBackupVar']*backup_gen, r, p['LifeBackup'], load)
        lcoe_trans = self._lcoe_component(p['CapexTrans'] *kappa_trans,  p['OpexTrans'] *kappa_trans,                                  r, p['LifeTrans'],  load)

        result = {
            "total":  lcoe_wind + lcoe_solar + lcoe_backup + lcoe_trans,
            "wind":   lcoe_wind,
            "solar":  lcoe_solar,
            "backup": lcoe_backup,
            "trans":  lcoe_trans,
        }
        return result

    def lcoe_nodal(self,kappa):
        p = self.P
        r = p['RateOfReturn']
        load = self.load.sum(axis=0)
        backup_gen = self.backup_generation.sum(axis=0)

        kappa_wind = kappa['wind']
        kappa_solar = kappa['solar']
        kappa_backup = kappa['backup']
        kappa_trans = kappa['trans_volume']

        lcoe_wind  = self._lcoe_component(p['CapexWind']  *kappa_wind,   p['OpexWind']  *kappa_wind,                                   r, p['LifeWind'],   load)
        lcoe_solar = self._lcoe_component(p['CapexSolar'] *kappa_solar,  p['OpexSolar'] *kappa_solar,                                  r, p['LifeSolar'],  load)
        lcoe_backup= self._lcoe_component(p['CapexBackup']*kappa_backup, p['OpexBackup']*kappa_backup + p['OpexBackupVar']*backup_gen, r, p['LifeBackup'], load)
        lcoe_trans = self._lcoe_component(p['CapexTrans'] *kappa_trans,  p['OpexTrans'] *kappa_trans,                                  r, p['LifeTrans'],  load)

        result = {
            "total":  lcoe_wind + lcoe_solar + lcoe_backup + lcoe_trans,
            "wind":   lcoe_wind,
            "solar":  lcoe_solar,
            "backup": lcoe_backup,
            "trans":  lcoe_trans,
        }
        return result



