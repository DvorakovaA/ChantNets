"""
File containing utility functions for the experiments in this NA project.
"""

import pycantus
import graph_tool.all as gt
from collections import Counter
import numpy as np


def construct_bipart_source_feast_graph(corpus):
    """
    """
    g = gt.Graph(directed=False)
    source_map, feast_map = {}, {}
    vprop_name = g.new_vertex_property("string")
    vprop_type = g.new_vertex_property("string")
    print('Constructing bipartite graph between sources and feasts...')

    for chant in corpus.chants:
        if chant.srclink not in source_map:
            source_vertex = g.add_vertex()
            source_map[chant.srclink] = source_vertex
            vprop_name[source_vertex] = chant.srclink
            vprop_type[source_vertex] = 0 #'source'

        if chant.feast not in feast_map:
            feast_vertex = g.add_vertex()
            feast_map[chant.feast] = feast_vertex
            vprop_name[feast_vertex] = chant.feast
            vprop_type[feast_vertex] = 1 #'feast'

        g.add_edge(source_map[chant.srclink], feast_map[chant.feast])

    g.vp["name"] = vprop_name
    g.vp["type"] = vprop_type
    return g


def save_graph(g, path):
    """Save graph to file."""
    g.save(path)


