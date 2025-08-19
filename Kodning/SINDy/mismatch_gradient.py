
import pysindy as ps
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

#WORK IN PROGRESS

class FeatureBuilder:
    def __init__(self):
        self.scalers = {}

    def process_features(self, raw_features, feature_names, config):
        """
        Apply smoothing, scaling, and clipping to each feature based on config.
        Supports both 2D [n_hours, n_features] and 3D [n_hours, n_nodes, n_features].
        """
        n_hours = raw_features.shape[0]
        n_nodes = raw_features.shape[1] if raw_features.ndim == 3 else None
        n_features = raw_features.shape[-1]

        processed = np.zeros_like(raw_features)

        for i, name in enumerate(feature_names):
            settings = config.get(name, {})
            method = settings.get('smooth', 'rolling')
            window = settings.get('window', 24)
            clip = settings.get('clip', None)
            scale = settings.get('scale', False)

            if raw_features.ndim == 3:
                for node in range(n_nodes):
                    series = raw_features[:, node, i]
                    processed_series = self._smooth_single(series, method, window)

                    if clip:
                        processed_series = np.clip(processed_series, *clip)

                    if scale:
                        mean, std = np.mean(processed_series), np.std(processed_series)
                        std = std if std > 1e-6 else 1.0
                        processed_series = (processed_series - mean) / std
                        self.scalers[name] = (mean, std)

                    processed[:, node, i] = processed_series

            elif raw_features.ndim == 2:
                series = raw_features[:, i]
                processed_series = self._smooth_single(series, method, window)

                if clip:
                    processed_series = np.clip(processed_series, *clip)

                if scale:
                    mean, std = np.mean(processed_series), np.std(processed_series)
                    std = std if std > 1e-6 else 1.0
                    processed_series = (processed_series - mean) / std
                    self.scalers[name] = (mean, std)

                processed[:, i] = processed_series

        return processed

    def _smooth_single(self, series, method, window):
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import savgol_filter

        if method == 'rolling':
            return pd.Series(series).rolling(window=window, min_periods=1, center=True).mean().bfill().ffill().to_numpy()
        elif method == 'gaussian':
            return gaussian_filter1d(series, sigma=window / 6.0)
        elif method == 'savgol':
            if window % 2 == 0:
                window += 1
            return savgol_filter(series, window_length=window, polyorder=2, mode='nearest')
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    # def add_cyclic_time_features(self, n_hours):
    #     hours = np.arange(n_hours)
    #     sin_hour = np.sin(2 * np.pi * (hours % 24) / 24)
    #     cos_hour = np.cos(2 * np.pi * (hours % 24) / 24)
    #     sin_day = np.sin(2 * np.pi * (hours % 8760) / 8760)
    #     cos_day = np.cos(2 * np.pi * (hours % 8760) / 8760)
    #     return np.stack([sin_hour, cos_hour, sin_day, cos_day], axis=1), ['sin_hour', 'cos_hour', 'sin_day', 'cos_day']

    def add_cyclic_time_features(self, n_hours):
        hours = np.arange(n_hours)
        sin_hour = np.sin(2 * np.pi * (hours % 24) / 24)
        cos_hour = np.cos(2 * np.pi * (hours % 24) / 24)
        return np.stack([sin_hour, cos_hour], axis=1), ['sin_hour', 'cos_hour']


    def plot_smoothed_feature(self, raw_data, smoothed_data, feature_names, node=0, feature='solar', hours=1000):
        if feature not in feature_names:
            raise ValueError(f"Feature '{feature}' not found in provided feature names.")
        feature_idx = feature_names.index(feature)

        if raw_data.ndim == 2:
            raw_series = raw_data[:hours, feature_idx]
            smoothed_series = smoothed_data[:hours, feature_idx]
            title = f"{feature} (Global)"
        elif raw_data.ndim == 3:
            raw_series = raw_data[:hours, node, feature_idx]
            smoothed_series = smoothed_data[:hours, node, feature_idx]
            title = f"{feature} (Node {node})"
        else:
            raise ValueError("Expected 2D or 3D input data.")

        plt.figure(figsize=(10, 4))
        plt.plot(raw_series, label='Raw', alpha=0.6)
        plt.plot(smoothed_series, label='Processed', linewidth=2)
        plt.title(f"{title} — Raw vs Processed")
        plt.xlabel("Time [hours]")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()





from pysindy.feature_library import CustomLibrary


from pysindy.utils.axes import AxesArray, comprehend_axes



class FilteredPolynomialLibrary(CustomLibrary):
    def __init__(self, base_library, exclude_terms=None):
        self.base_library = base_library
        self.exclude_terms = exclude_terms if exclude_terms is not None else []

        # Pass-through attributes needed for PySINDy compatibility
        self.degree = getattr(base_library, "degree", None)
        self.include_bias = getattr(base_library, "include_bias", True)
        self.interaction_only = getattr(base_library, "interaction_only", False)

        self.safe_indices = None
        self.input_features = None
        self.function_names = []

        super().__init__(library_functions=None, function_names=[])


    # def fit(self, X, y=None):
    #     self.base_library.fit(X, y)

    #     self.n_features_in_ = self.base_library.n_features_in_
    #     self.n_output_features_ = self.base_library.n_output_features_

    #     all_names = self.base_library.get_feature_names()
    #     self.input_features = all_names

    #     self.safe_indices = [
    #         i for i, name in enumerate(all_names)
    #         if not any(excl in name for excl in self.exclude_terms)
    #     ]
    #     self.function_names = [all_names[i] for i in self.safe_indices]

    #     # Debug output
    #     print("\n🔍 All terms in polynomial library:")
    #     print(all_names)

    #     print("\n❌ Excluded terms (matched):")
    #     print([name for i, name in enumerate(all_names) if i not in self.safe_indices])

    #     print("\n✅ Remaining terms used:")
    #     print(self.function_names)

    #     from functools import partial
    #     self.functions = [
    #         partial(lambda x, j: self.base_library.transform(x)[:, j], j=j)
    #         for j in self.safe_indices
    #     ]

    #     return self

    def fit(self, X, y=None, **fit_params):
        self.base_library.fit(X, y, **fit_params)

        self.n_features_in_ = self.base_library.n_features_in_
        self.n_output_features_ = self.base_library.n_output_features_

        all_names = self.base_library.get_feature_names()
        self.input_features = all_names

        # Build map from user-friendly names to x-index (e.g., 'sin_hour' -> x5)
        if hasattr(self.base_library, "input_features") and self.base_library.input_features is not None:
            name_to_x = {v: f"x{i}" for i, v in enumerate(self.base_library.input_features)}
        else:
            name_to_x = {}

        # Translate exclude_terms like 'sin_hour^2' -> 'x5^2'
        translated_exclude_terms = []
        for term in self.exclude_terms:
            for name, x in name_to_x.items():
                term = term.replace(name, x)
            translated_exclude_terms.append(term)

        # Apply exclusion
        excluded_matched = []
        self.safe_indices = []
        for i, name in enumerate(all_names):
            if any(excl in name for excl in translated_exclude_terms):
                excluded_matched.append(name)
            else:
                self.safe_indices.append(i)

        self.function_names = [all_names[i] for i in self.safe_indices]

        # Debug output
        print("\n🔍 All terms in polynomial library:")
        print(all_names)

        print("\n❌ Excluded terms (matched):")
        print(excluded_matched)

        print("\n✅ Remaining terms used:")
        print(self.function_names)

        from functools import partial
        self.functions = [
            partial(lambda x, j: self.base_library.transform(x)[:, j], j=j)
            for j in self.safe_indices
        ]

        return self




    def transform(self, X):
        X_trans = self.base_library.transform(X)

        # If called with a list (multiple trajectories), apply filtering per item
        if isinstance(X_trans, list):
            return [
                AxesArray(x[:, self.safe_indices], comprehend_axes(x))
                for x in X_trans
            ]

        # If called with a single array
        return AxesArray(X_trans[:, self.safe_indices], comprehend_axes(X_trans))


    def get_feature_names(self, input_features=None):
        if input_features is None:
            input_features = self.base_library.input_features
        return [self.base_library.get_feature_names(input_features)[i] for i in self.safe_indices]








class NodeDynamicsModel:
    def __init__(self, poly_degree=2, threshold=1e-4):
        self.poly_degree = poly_degree
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['M_i', 'ΔM_i']

    def prepare_training_data(self, mismatch_data, adjacency_matrix, external_features=None, external_feature_names=None):
        n_time, n_nodes = mismatch_data.shape
        laplacian = adjacency_matrix - np.diag(np.sum(adjacency_matrix, axis=1))
        delta_mismatch = mismatch_data @ laplacian.T
        mismatch_deriv = np.gradient(mismatch_data, axis=0)

        X = []
        y = []

        for t in range(n_time):
            for i in range(n_nodes):
                row = [mismatch_data[t, i], delta_mismatch[t, i]]
                if external_features is not None:
                    if external_features.ndim == 3:
                        row += list(external_features[t, i])
                    elif external_features.ndim == 2:
                        row += list(external_features[t])
                X.append(row)
                y.append(mismatch_deriv[t, i])

        X = np.array(X)
        self.X_scaled = self.scaler.fit_transform(X)
        #self.y = np.array(y)
        self.y = np.array(y).reshape(-1, 1)  # shape (N, 1)


        if external_feature_names:
            self.feature_names = ['M_i', 'ΔM_i'] + external_feature_names

    def fit(self):

        #excluded = ['sin_hour sin_hour', 'cos_hour cos_hour', 'sin_hour^2', 'cos_hour^2']
        excluded = [
    'sin_hour^2', 'cos_hour^2','sin_hour cos_hour']
#     'sin_hour cos_hour',
#     'CF_solar sin_hour', 'CF_solar cos_hour',
#     'CF_wind sin_hour', 'CF_wind cos_hour',
#     'Load sin_hour', 'Load cos_hour',
#     'ΔM_i sin_hour', 'ΔM_i cos_hour',
#     'M_i sin_hour', 'M_i cos_hour',
# ]

 # customize here

        #base_lib = ps.PolynomialLibrary(degree=self.poly_degree, include_bias=True)

        base_lib = ps.PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=True
        )
        base_lib.fit(self.X_scaled)  # Fit first so features get registered
        base_lib.input_features = self.feature_names  # ✅ Set the correct names manually


        library = FilteredPolynomialLibrary(base_lib, exclude_terms=excluded)

        self.model = ps.SINDy(
            feature_library=library,
            optimizer=ps.STLSQ(threshold=self.threshold),
            feature_names=self.feature_names
        )
        self.model.fit([self.X_scaled], x_dot=[self.y],multiple_trajectories=True)



    def print_equation(self):
        if self.model:
            self.model.print()
        else:
            print("Model not yet fitted.")

    def predict_derivative(self, M_i, delta_M_i, external_features=None):
        row = [M_i, delta_M_i]
        if external_features is not None:
            row += list(external_features)
        row_scaled = self.scaler.transform([row])
        return self.model.predict(row_scaled)[0]


class SpatialNetworkModel:
    def __init__(self, node_model, adjacency_matrix):
        self.node_model = node_model
        self.adjacency = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.laplacian = adjacency_matrix - np.diag(adjacency_matrix.sum(axis=1))

    def simulate(self, initial_mismatch, n_hours, external_features=None):
        M = np.zeros((n_hours, self.n_nodes))
        M[0] = initial_mismatch

        for t in range(1, n_hours):
            delta_M = M[t - 1] @ self.laplacian.T
            features = np.stack([M[t - 1], delta_M], axis=1)

            if external_features is not None:
                if external_features.ndim == 3:
                    ext = external_features[t - 1]
                elif external_features.ndim == 2:
                    ext = np.repeat(external_features[t - 1][None, :], self.n_nodes, axis=0)
                else:
                    raise ValueError("Invalid shape for external features")

                features = np.hstack([features, ext])

            features_scaled = self.node_model.scaler.transform(features)
            dM_dt = self.node_model.model.predict(features_scaled).flatten()
            M[t] = M[t - 1] + dM_dt

        return M