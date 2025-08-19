from post_processing.LCOE_postprocessor import LCOE_PostProcessor
import initialize.data_class as dc
from layout.layout_class import layout_class
from link_layout.LinkLayout_class import link_network_class
from flowtracing.flowtracing_class import flowtracing_class
from flowtracing.used_kappa_class import used_kappa_class


class RenewableNetwork:
    def __init__(self, data_dir="Data", defaults=None):
        self.raw = dc.get_data(data_dir)
        self.defaults = (defaults or {
            "gamma": 1, "beta": 1, "n_t": 8760,
            "Layout_Scheme": "global", "Balancing_Scheme": "global",
        }).copy()

        # Public stage objects
        self.init = None
        self.layout = None
        self.network = None
        self.flowtrace = None
        self.used_kappa = None

    def _merge(self, overrides):
        p = self.defaults.copy()
        if overrides: p.update(overrides)
        return p

    def init_stage(self, **params):
        if params or self.init is None:
            self.init = dc.data_class(self.raw, **self._merge(params))
            # reset downstream
            self.layout = self.network = self.flowtrace = self.used_kappa = None

    def layout_stage(self, **params):
        self.init_stage(**params)
        if self.layout is None:
            self.layout = layout_class(self.init)
            self.network = self.flowtrace = self.used_kappa = None

    def network_stage(self, **params):
        self.layout_stage(**params)
        if self.network is None:
            self.network = link_network_class(self.init, self.layout)
            self.flowtrace = self.used_kappa = None

    def flowtracing_stage(self, **params):
        self.network_stage(**params)
        if self.flowtrace is None:
            self.flowtrace = flowtracing_class(self.init, self.layout, self.network)
            self.used_kappa = None

    def used_kappa_stage(self, **params):
        self.flowtracing_stage(**params)
        if self.used_kappa is None:
            self.used_kappa = used_kappa_class(self.init, self.layout, self.network, self.flowtrace)

    # --- make stages callable as rn.layout(...) ---
    def __getattr__(self, name):
        if name in ["init", "layout", "network", "flowtrace", "kappa"]:
            return lambda **params: getattr(self, f"{name}_stage")(**params)
        raise AttributeError(f"'RenewableNetwork' object has no attribute '{name}'")




rn = RenewableNetwork()
rn.used_kappa_stage(n_t= 8760,gamma=1,beta=1)
#%% post processing (Work in progress)
m_lcoe = LCOE_PostProcessor(rn,installed_trans_volume_method='half_link')
b = m_lcoe.lcoe_nodal(rn.layout.kappa)