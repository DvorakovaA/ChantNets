import pickle
import graph_tool.all as gt
from collections import defaultdict
import os
import networkx as nx

class SBModel:
    """
    A wrapper for graph-tool's SBM fitting functions that allows us to easily
    fit an SBM to a graph and keep the best result.
    """
    def __init__(self):
        self.graph = None
        self.best_states = {}
        self.n_init = 10
        self.deg_corrected = True
        self.states = defaultdict(list)

    def load_graph(self, path):
        """
        Load a graph from a file.
        """
        self.graph = gt.load_graph(path)
        print(f"Loaded graph with {self.graph.num_vertices()} vertices, {self.graph.num_edges()} edges")
    
    def load_states(self, path):
        """
        Load fitted states from a file.
        """
        self.best_states = pickle.load(open(path, 'rb'))
    
    def load_graph_nx(self, G):
        """
        Load a graph from a networkx graph.
        """
        g = gt.Graph(directed=G.is_directed())
        vprop_name = g.new_vertex_property("string")
        eprop_count = g.new_edge_property("int")
        eprop_weight = g.new_edge_property("double")
        for v in G.nodes():
            

        
        self.graph = g
        print(f"Loaded graph with {self.graph.num_vertices()} vertices, {self.graph.num_edges()} edges")

    def get_states(self):
        """
        Return states from all fitting attempts.
        """
        return self.states
    
    def fit_sbm(self, n_init=10):
        """
        Try fitting SBM to the graph using graph-tool's built-in functions
        multiple times and keep the best result.
        """
        self.states = defaultdict(list)
        print(f"Fitting SBM to graph with {self.graph.num_vertices()} vertices and {self.graph.num_edges()} edges...")
        for i in range(n_init):
            print(f"Fitting SBM (init {i+1}/{n_init})...")
            args = dict(deg_corr=self.deg_corrected, clabel=self.graph.vp["type"])
            state = gt.minimize_blockmodel_dl(self.graph, state_args=args)
            
            self.states['DC_SBM'].append({"state": state, "model": state.entropy()})
            print(f"[{i+1}/{n_init}] entropy = {state.entropy():.2f}")

        best = min(self.states['DC_SBM'], key=lambda s: s["model"])
        self.best_states['DC_SBM'] = best["state"]


    def fit_nested_sbm(self, n_init=10):
        """
        Try fitting nested SBM to the graph using graph-tool's built-in functions
        multiple times and keep the best result.
        """
        self.states = defaultdict(list)
        print(f"Fitting nested SBM to graph with {self.graph.num_vertices()} vertices and {self.graph.num_edges()} edges...")
        for i in range(n_init):
            print(f"Fitting nested SBM (init {i+1}/{n_init})...")
            state = gt.minimize_nested_blockmodel_dl(self.graph,
                        state_args=dict(deg_corr=self.deg_corrected, clabel=self.graph.vp["type"]))
            self.states['Nested_DC_SBM'].append({"state": state, "model": state.entropy()})
            print(f"[{i+1}/{n_init}] entropy = {state.entropy():.2f}")

        best = min(self.states['Nested_DC_SBM'], key=lambda s: s["model"])
        self.best_states['Nested_DC_SBM'] = best["state"]
        #print(self.states)

    def fit_sbm_weighted(self, weight_label, n_init=10):
        """
        weighted SBM to the graph using graph-tool's built-in functions
        """
        self.states = defaultdict(list)
        print(f"Fitting weighted SBM to graph with {self.graph.num_vertices()} vertices and {self.graph.num_edges()} edges...")

        weight_prop = self.graph.ep[weight_label]

        for i in range(n_init):
            print(f"Fitting weighted SBM (init {i+1}/{n_init})...")

            args = dict(
                deg_corr = self.deg_corrected,
                clabel = self.graph.vp["type"],
                recs = [weight_prop],
                rec_types = ["real-normal"])

            state = gt.minimize_blockmodel_dl(self.graph, state_args = args)

            self.states['Weighted_DC_SBM'].append({"state": state, "model": state.entropy()})
            print(f"[{i+1}/{n_init}] entropy = {state.entropy():.2f}")

        best = min(self.states['Weighted_DC_SBM'], key = lambda s: s["model"])
        self.best_states['Weighted_DC_SBM'] = best["state"]

    def fit_nested_sbm_weighted(self, weight_label="weight", n_init=10):
        """
        weighted nested SBM to the graph using graph-tool's built-in functions
        """
        self.states = defaultdict(list)
        print(f"Fitting weighted nested SBM to graph with {self.graph.num_vertices()} vertices and {self.graph.num_edges()} edges...")

        weight_prop = self.graph.ep[weight_label]

        for i in range(n_init):
            print(f"Fitting weighted nested SBM (init {i+1}/{n_init})...")

            state = gt.minimize_nested_blockmodel_dl(
                self.graph,
                state_args = dict(
                    deg_corr = self.deg_corrected,
                    clabel = self.graph.vp["type"],
                    recs = [weight_prop],
                    rec_types = ["real-normal"]))

            self.states['Weighted_Nested_DC_SBM'].append({"state": state, "model": state.entropy()})
            print(f"[{i+1}/{n_init}] entropy = {state.entropy():.2f}")

        best = min(self.states['Weighted_Nested_DC_SBM'], key = lambda s: s["model"])
        self.best_states['Weighted_Nested_DC_SBM'] = best['state']
    

    def save_states(self, path):
        """
        Save the fitted best state to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(self.best_states, open(path, 'wb'))
    