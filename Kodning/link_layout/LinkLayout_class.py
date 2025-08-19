import numpy as np
import networkx as nx
import geopy.distance
import hashlib
from pathlib import Path
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent  # <-- link_layout folder

class link_network_class:
    def __init__(self,init,layout,cache_file=None):
        link_network = classic_link_network(init, layout, cache_file)
        adjacency = np.loadtxt("Data/adjacency.csv", delimiter=",", dtype=int)
        self.layout,self.flows,self.kappa = link_network.final_network_layout(adjacency)
        self.n_l = self.kappa.shape[0]


        #ADD TRANS KAPPA TO INSTALLED KAPPA CLASS
        layout.kappa.update({"trans": self.kappa})

class classic_link_network:
    def __init__(self, init, layout, cache_file):
        self.Bus = init.BusData
        self.Injections = layout.misc['injection']

        # default cache lives next to this file (link_layout/)
        self.cache_file = Path(cache_file) if cache_file else (MODULE_DIR / "dist_matrix_cache.npy")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
        self.meta_file = self.cache_file.with_name(self.cache_file.name + ".meta")

        self.dist_matrix = self._load_or_compute_dist_matrix()

        self._lap_factor = None
        self._lap_reduced_shape = None

    def topology_optimized_network_layout(self):
        from link_layout.topology_opt_layout import DensityTopologyTransOptimizer
        optimizer = DensityTopologyTransOptimizer(self.Bus, self.dist_matrix, self.Injections.T)
        rho_design, rho_proj, A_final, info = optimizer.optimize()
        return rho_design, rho_proj, A_final, info

    def final_network_layout(self, adjacency_data):
        return self.evaluate_adjacency(adjacency_data)

    def _hash_bus_coordinates(self):
        coords = self.Bus[['x', 'y']].to_numpy().astype(float)
        return hashlib.sha256(coords.tobytes()).hexdigest()

    def _load_or_compute_dist_matrix(self):
        coords_hash = self._hash_bus_coordinates()
        if self.cache_file.exists() and self.meta_file.exists():
            saved_hash = self.meta_file.read_text().strip()
            if saved_hash == coords_hash:
                return np.load(self.cache_file)
        # compute fresh
        coords = self.Bus[['x', 'y']].to_numpy()
        n = len(coords)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                dist = geopy.distance.distance(coords[i], coords[j]).km
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        np.save(self.cache_file, dist_matrix)
        self.meta_file.write_text(coords_hash)
        return dist_matrix

    def evaluate_adjacency(self, adjacency_data):
        PTDF, incidence, edge_list = self.PTDF(adjacency_data)
        FlowPattern = self.Injections @ PTDF.T
        kappa, length = self.Trans_capacity(edge_list, FlowPattern)

        layout= {
            'edge_list':edge_list,
            'incidence':incidence,
            'adjacency':adjacency_data,
            'length':length
            }
        flows = {
            'PTDF':PTDF,
            'FlowPattern':FlowPattern
            }
        return layout,flows,kappa

    def PTDF(self,adjacency):
        # Build graph
        G = nx.from_numpy_array(np.asarray(adjacency, float), create_using=nx.Graph())
        for e in G.edges:
            G[e[0]][e[1]]['weight'] = 1

        incidence  = pd.DataFrame(nx.incidence_matrix(G, oriented=True).todense(), index=G.nodes, columns=G.edges)
        #Laplacian and pseudoinverse
        laplacian  = pd.DataFrame(nx.laplacian_matrix(G).todense(), index=G.nodes, columns=G.nodes)
        Lp         = pd.DataFrame(np.linalg.pinv(laplacian, hermitian=True), index=G.nodes, columns=G.nodes)

        PTDFMatrix = incidence.T @ Lp

        return PTDFMatrix.values, incidence.values, list(G.edges)


    def Trans_capacity(self, edge_list, FlowPattern):
        length = np.array([self.dist_matrix[i, j] for i, j in edge_list], dtype=np.float32)
        kappaTrans = np.quantile(np.abs(FlowPattern), q=0.99, axis=0)
        return kappaTrans, length

