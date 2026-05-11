"""
File containing utility functions for the experiments in this NA project.
"""

import pycantus
import graph_tool.all as gt


def construct_bipart_source_feast_graph(corpus):
    """
    """
    pass


def fit_sbm(graph, n_init=10):
    """
    Try fitting SBM to the graph using graph-tool's built-in functions
    multiple times and keep the best result.
    """
    # gt.minimize_blockmodel_dl
    pass


def fit_nested_sbm(graph, n_init=10):
    """
    Try fitting nested SBM to the graph using graph-tool's built-in functions
    multiple times and keep the best result.
    """
    # gt.minimize_nested_blockmodel_dl
    pass

def fit_sbm_weighted(graph, weight_label, n_init=10):
    """
    Try fitting SBM to the graph using graph-tool's built-in functions
    multiple times and keep the best result.
    """
    # gt.minimize_blockmodel_dl
    pass


def fit_nested_sbm_weighted(graph, weight_label, n_init=10):
    """
    Try fitting nested SBM to the graph using graph-tool's built-in functions
    multiple times and keep the best result.
    """
    # gt.minimize_nested_blockmodel_dl
    pass

