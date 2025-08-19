import numpy as np
import pandas as pd

# ---------- loader (independent of params) ----------
_DATA_CACHE = None

def get_data(data_dir="Data", force=False):
    """
    Read CSVs once and cache them. Set force=True to re-read from disk.
    """
    global _DATA_CACHE
    if _DATA_CACHE is None or force:
        BusData  = pd.read_csv(f"{data_dir}/bus.csv")
        LoadData = np.genfromtxt(f"{data_dir}/Load.csv", delimiter=",", dtype=np.float32)
        CF_solar = pd.read_csv(f"{data_dir}/CF_solar.csv").to_numpy(dtype=np.float32)
        CF_wind  = pd.read_csv(f"{data_dir}/CF_wind.csv").to_numpy(dtype=np.float32)
        _DATA_CACHE = {
            "BusData":  BusData,
            "Load":     LoadData,
            "CF_solar": CF_solar,
            "CF_wind":  CF_wind,
        }
    return _DATA_CACHE

# ---------- (params only) ----------
class data_class:
    def __init__(self, data_dict, **params):
        self.data = data_dict  # already-loaded raw data
        self.BusData  = self.data["BusData"]
        self.load     = self.data["Load"]
        self.CF_solar = self.data["CF_solar"]
        self.CF_wind  = self.data["CF_wind"]

        self._init_country_data()

        # now set params (no loading)
        self.input_parameters = {}
        self.set_params(**params)

    def set_params(self,n_alpha=11, **kwargs):
        # update only what was provided
        self.input_parameters.update(kwargs)

        # pull with defaults
        self.layout_scheme    = self.input_parameters.get("Layout_Scheme")
        self.balancing_scheme = self.input_parameters.get("Balancing_Scheme")
        self.gamma            = np.float32(self.input_parameters.get("gamma", 0.0))
        self.beta             = np.float32(self.input_parameters.get("beta", 0.0))
        self.n_alpha          = n_alpha
        self.t                = self.input_parameters.get("n_t")

        # any aggregates that should be on FULL data (t_cut handled outside)
        self.mean_load  = self.load.mean(axis=0)
        self.n          = self.load.shape[1]

    def _init_country_data(self):
        self.country_of_n = self.BusData["country"].to_numpy()
        self.country_list = np.unique(self.country_of_n)

        starts = []
        for country in self.country_list:
            idx = np.where(self.country_of_n == country)[0]
            if idx.size:
                starts.append(idx[0])

        starts = np.sort(np.array(starts, dtype=int))
        if starts.size == 0 or starts[0] != 0:
            starts = np.insert(starts, 0, 0)

        n_buses = self.BusData.shape[0]
        if starts[-1] != n_buses:
            starts = np.append(starts, n_buses)

        self.country_int = starts.astype(int)
        self.n_country   = len(self.country_list)
