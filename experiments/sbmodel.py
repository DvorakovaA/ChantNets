"""

"""
import pickle
import graph_tool.all as gt


class SBModel:
    """
    A wrapper for graph-tool's SBM fitting functions that allows us to easily
    fit an SBM to a graph and keep the best result.
    """
    def __init__(self, graph):
        self.graph = graph
        self.model = None
        self.n_init = 10


    def fit_sbm(self, n_init=10):
        """
        Try fitting SBM to the graph using graph-tool's built-in functions
        multiple times and keep the best result.
        """
        # gt.minimize_blockmodel_dl
        pass


    def fit_nested_sbm(self, n_init=10):
        """
        Try fitting nested SBM to the graph using graph-tool's built-in functions
        multiple times and keep the best result.
        """
        # gt.minimize_nested_blockmodel_dl
        pass

    def fit_sbm_weighted(self, weight_label, n_init=10):
        """
        Try fitting SBM to the graph using graph-tool's built-in functions
        multiple times and keep the best result.
        """
        # gt.minimize_blockmodel_dl
        pass


    def fit_nested_sbm_weighted(self, weight_label, n_init=10):
        """
        Try fitting nested SBM to the graph using graph-tool's built-in functions
        multiple times and keep the best result.
        """
        # gt.minimize_nested_blockmodel_dl
        pass

    def load_model(self, model):
        """
        Load a previously fitted model.
        """
        self.model = model
    
    def save_model(self, path):
        """
        Save the fitted model to a file.
        """
        pass