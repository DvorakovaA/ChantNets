"""
File containing utility functions for the experiments in this NA project.
"""

import pycantus
import graph_tool.all as gt
from collections import defaultdict
import numpy as np
import os
import networkx as nx
import itertools


def construct_bipart_source_feast_graph(corpus):
    """
    Construct a bipartite weighted graph where source nodes are connected to feast nodes
    if at least one chant from that source belongs to that feast
    """
    g = gt.Graph(directed=False)
    source_map, feast_map = {}, {}

    vprop_name = g.new_vertex_property("string")
    vprop_type = g.new_vertex_property("int") # i changed this since we store 0 and 1
    eprop_count = g.new_edge_property("int")
    eprop_weight = g.new_edge_property("double")

    # unique Cantus IDs for each sourse-feast pair
    source_feast_chant_ids = defaultdict(set)

    for chant in corpus.chants:
        source = chant.srclink
        feast = chant.feast
        cantus_id = chant.cantus_id

        if source is None or feast is None or cantus_id is None:
            continue

        source_feast_chant_ids[(source, feast)].add(cantus_id)

    print("Constructing bipartite graph between sources and feasts...")

    for (source, feast), cantus_ids in source_feast_chant_ids.items():

        if source not in source_map:
            source_vertex = g.add_vertex()
            source_map[source] = source_vertex
            vprop_name[source_vertex] = source
            vprop_type[source_vertex] = 0

        if feast not in feast_map:
            feast_vertex = g.add_vertex()
            feast_map[feast] = feast_vertex
            vprop_name[feast_vertex] = feast
            vprop_type[feast_vertex] = 1

        edge = g.add_edge(source_map[source], feast_map[feast])
        count = len(cantus_ids)
        eprop_count[edge] = count
        eprop_weight[edge] = 0.5 if count == 1 else np.log2(count)

    g.vp["name"] = vprop_name
    g.vp["type"] = vprop_type

    g.ep["count"] = eprop_count
    g.ep["weight"] = eprop_weight

    print(f"Number of source nodes: {len(source_map)}")
    print(f"Number of feast nodes: {len(feast_map)}")
    print(f"Number of source-feast edges: {g.num_edges()}")

    return g

def save_graph(g, path):
    """Save graph to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.save(path)


def build_feast_network(corpus, feast):
    # Build the network
    G = nx.Graph(name=feast)
    for source in corpus.sources:
        G.add_node(source.srclink, name=source.siglum)
    source_chants = defaultdict(set)
    for chant in corpus.chants:
        source_chants[chant.srclink].add(chant.cantus_id)
    for source1, source2 in itertools.combinations(corpus.sources, 2):
        shared_chants = source_chants[source1.srclink] & source_chants[source2.srclink]
        #print(f'{source1.siglum} and {source2.siglum} share {len(shared_chants)} chants')
        if shared_chants:
            jaccard = len(shared_chants) / len(source_chants[source1.srclink] | source_chants[source2.srclink])
            G.add_edge(source1.srclink, source2.srclink, weight=jaccard)
    return G


def graph_info_nx(G, fast = False):
    """
    Print basic info about a NetworkX graph.
    """
    print("{:>12s} | '{:s}'".format('Graph', G.name))

    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    print("{:>12s} | {:,d} ({:,d})".format('Nodes', n, nx.number_of_isolates(G)))
    print("{:>12s} | {:,d} ({:,d})".format('Edges', m, nx.number_of_selfloops(G)))
    print("{:>12s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, max([k for _, k in G.degree()])))

    if not fast:
        C = sorted(nx.connected_components(nx.MultiGraph(G)), key = len, reverse = True)

        print("{:>12s} | {:.1f}% ({:,d})".format('Components', 100 * len(C[0]) / n, len(C)))

        print("{:>12s} | {:.4f}".format('Clustering', nx.average_clustering(G if type(G) == nx.Graph else nx.Graph(G))))
        
        C = nx.community.louvain_communities(G, weight = 'weight')
        Q = nx.community.modularity(G, C, weight = 'weight')
        
        print("{:>12s} | {:.4f} ({:,d})".format('Modularity from Louvain', Q, len(C)))
    print()


def get_partitions_from_state(state):
    """
    Get the partition of nodes from a graph-tool state object.
    """
    node_map = state.get_blocks()
    graph = node_map.get_graph()
    partitions = defaultdict(list)
    for v in graph.vertices():
        partition = node_map[v]
        partitions[partition].append(int(v))
    return partitions