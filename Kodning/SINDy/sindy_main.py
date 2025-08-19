import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import SINDy.mismatch_gradient as SINDy_model

class sindy_caller:
    def __init__(self, init, layout, feature_config,
                 external_feature_getter=None,
                 test_size=0.2, random_state=42):

        self.adjacency = init.link_data['adjacency']
        self.n_nodes = self.adjacency.shape[0]
        self.mismatch = layout['generation']['mismatch']
        self.n_hours = self.mismatch.shape[0]

        self.builder = SINDy_model.FeatureBuilder()
        all_features = []
        all_names = []

        # === External features ===
        if external_feature_getter is not None:
            self.raw_data, self.feature_names = external_feature_getter(init, layout)
            self.smoothed = self.builder.process_features(
                self.raw_data, self.feature_names, feature_config
            )
            all_features.append(self.smoothed)
            all_names += self.feature_names
        else:
            self.raw_data = self.smoothed = None
            self.feature_names = []

        # === Add cyclic time features ===
        cyclic, cyclic_names = self.builder.add_cyclic_time_features(self.n_hours)
        cyclic_broadcast = np.repeat(cyclic[:, None, :], self.n_nodes, axis=1)
        all_features.append(cyclic_broadcast)
        all_names += cyclic_names

        # === Final feature matrix ===
        self.feature_names = all_names
        full_features = np.concatenate(all_features, axis=2)

        self.train_mismatch, self.test_mismatch, self.train_features, self.test_features = train_test_split(
            self.mismatch, full_features, test_size=test_size,
            random_state=random_state, shuffle=False
        )

    def plot_smoothed(self, feature_name, node=0, hours=1000):
        if self.smoothed is None or self.feature_names is None:
            raise ValueError("No processed features found.")
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found. Available: {self.feature_names}")

        feature_idx = self.feature_names.index(feature_name)
        raw_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else None

        raw_data = self.raw_data[:, node, raw_idx] if self.raw_data.ndim == 3 else self.raw_data[:, raw_idx]
        smoothed_data = self.smoothed[:, node, feature_idx] if self.smoothed.ndim == 3 else self.smoothed[:, feature_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(raw_data[:hours], label='Unprocessed', alpha=0.5)
        plt.plot(smoothed_data[:hours], label='Processed', linewidth=2)
        plt.title(f"{feature_name} (Node {node}) — Raw vs Processed")
        plt.xlabel("Time [hours]")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def model_builder(self, poly_degree=2, threshold=0.01, print_eq=True):
        self.model = SINDy_model.NodeDynamicsModel(poly_degree=poly_degree, threshold=threshold)
        self.model.prepare_training_data(
            self.train_mismatch,
            self.adjacency,
            external_features=self.train_features,
            external_feature_names=self.feature_names
        )
        self.model.fit()
        if print_eq:
            self.model.print_equation()

    def predict(self, n_hours, test=True):
        features = self.test_features if test else self.train_features
        mismatch = self.test_mismatch if test else self.train_mismatch

        if n_hours > features.shape[0]:
            raise ValueError(f"Requested n_hours={n_hours}, but only {features.shape[0]} available")

        init_mismatch = mismatch[0]
        simulator = SINDy_model.SpatialNetworkModel(self.model, self.adjacency)

        return simulator.simulate(init_mismatch, n_hours, external_features=features[:n_hours])

    def evaluate(self, n_hours=48, node_idx=None):
        predicted = self.predict(n_hours, test=True)

        true = self.test_mismatch[:n_hours]

        if node_idx is not None:
            y_true = true[:, node_idx]
            y_pred = predicted[:, node_idx]
            print(f"Node {node_idx} — R²: {r2_score(y_true, y_pred):.3f}, MAE: {mean_absolute_error(y_true, y_pred):.3f}")
            plt.figure()
            plt.plot(y_true, label="Actual")
            plt.plot(y_pred, label="Predicted")
            plt.title(f"Node {node_idx} Prediction")
            plt.xlabel("Time [hours]")
            plt.ylabel("Mismatch")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            r2s = [r2_score(true[:, i], predicted[:, i]) for i in range(self.n_nodes)]
            maes = [mean_absolute_error(true[:, i], predicted[:, i]) for i in range(self.n_nodes)]
            print(f"All nodes — Mean R²: {np.mean(r2s):.3f}, Mean MAE: {np.mean(maes):.3f}")