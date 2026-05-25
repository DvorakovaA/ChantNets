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
import pandas as pd
import json
from pathlib import Path

# ~ GRAPH CONSTRUCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        eprop_weight[edge] = 0.5 if count == 1 else np.log2(count) # setting weight to 0.5 when only one chant

    g.vp["name"] = vprop_name
    g.vp["type"] = vprop_type

    g.ep["count"] = eprop_count
    g.ep["weight"] = eprop_weight

    print(f"Number of source nodes: {len(source_map)}")
    print(f"Number of feast nodes: {len(feast_map)}")
    print(f"Number of source-feast edges: {g.num_edges()}")

    return g


def construct_bipart_source_feast_reducted_graph(corpus, min_cid_per_feast):
    """
    Construct a bipartite weighted graph where source nodes are connected to feast nodes
    while we consider only feasts that have at least min_cid_per_feast unique cantus IDs 
    across all sources
    """
    g = gt.Graph(directed=False)
    source_map, feast_map = {}, {}

    vprop_name = g.new_vertex_property("string")
    vprop_type = g.new_vertex_property("int")
    eprop_count = g.new_edge_property("int")
    eprop_weight = g.new_edge_property("double")

    # deetct bog enough feasts
    feast_sizes = defaultdict(set)
    for chant in corpus.chants:
        feast = chant.feast
        cantus_id = chant.cantus_id

        if feast is None or cantus_id is None:
            continue

        feast_sizes[feast].add(cantus_id)

    big_enough_feasts = {feast for feast, cantus_ids in feast_sizes.items() if len(cantus_ids) >= min_cid_per_feast}

    # unique Cantus IDs for each source-feast pair
    source_feast_cid = defaultdict(set)

    for chant in corpus.chants:
        source = chant.srclink
        feast = chant.feast
        cantus_id = chant.cantus_id

        if source is None or feast is None or cantus_id is None:
            continue
        if feast not in big_enough_feasts:
            continue

        source_feast_cid[(source, feast)].add(cantus_id)

    print("Constructing bipartite graph between sources and feasts...")

    for (source, feast), cantus_ids in source_feast_cid.items():
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
        eprop_weight[edge] = 0.5 if count == 1 else np.log2(count) # setting weight to 0.5 when only one chant

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


# ~ SBM PARTITION EXTRACTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_partitions_from_state(state, sigla_dict):
    """
    Get the partition of nodes from a graph-tool state object.
    """
    node_map = state.get_blocks()
    graph = node_map.get_graph()
    partitions = defaultdict(list)
    sigla_partitions = defaultdict(list)
    feast_partitions = defaultdict(list)
    for v in graph.vertices():
        partition = node_map[v]
        partitions[partition].append(int(v))
        if graph.vp["type"][v] == 0: # only add sigla for source nodes
            sigla_partitions[partition].append(sigla_dict[graph.vp["name"][v]])
        else:
            feast_partitions[partition].append(graph.vp["name"][v])
    return partitions, sigla_partitions, feast_partitions


def get_nested_partitions_from_state(state, sigla_dict):
    """
    Get the nested partition of nodes from a graph-tool state object.
    """
    node_map = state.get_levels()[0].get_blocks() # get node map from first level of nested SBM
    graph = node_map.get_graph()
    partitions = {}
    sigla_partitions = {}
    feast_partitions = {}
    """
    for level in state.get_levels():
        partitions[level] = defaultdict(list)
        sigla_partitions[level] = defaultdict(list)
        feast_partitions[level] = defaultdict(list)
        node_map = level.get_blocks()
        for v in graph.vertices():
            partition = node_map[v]
            partitions[level][partition].append(int(v))
            if graph.vp["type"][v] == 0: # only add sigla for source nodes
                sigla_partitions[level][partition].append(sigla_dict[graph.vp["name"][v]])
            else:
                feast_partitions[level][partition].append(graph.vp["name"][v])

    return partitions, sigla_partitions, feast_partitions
    """
    for i in range(len(state.get_levels())):
        level = state.project_level(i)
        partitions[i] = defaultdict(list)
        sigla_partitions[i] = defaultdict(list)
        feast_partitions[i] = defaultdict(list)
        node_map = level.get_blocks()
        for v in graph.vertices():
            partition = node_map[v]
            partitions[i][partition].append(int(v))
            if graph.vp["type"][v] == 0: # only add sigla for source nodes
                sigla_partitions[i][partition].append(sigla_dict[graph.vp["name"][v]])
            else:
                feast_partitions[i][partition].append(graph.vp["name"][v])

    return partitions, sigla_partitions, feast_partitions


def save_nested_partitions(sigla_partitions, path):
    """
    Save the nested partitions to a file.
    """
    sigla_rows = defaultdict(dict)

    for level, sigla_partition in sigla_partitions.items():
        for partition, sigla_list in sigla_partition.items():
            for siglum in sigla_list:
                sigla_rows[siglum][level] = partition

    max_level = max(
        max(levels.keys())
        for levels in sigla_rows.values()
        if levels)

    df = pd.DataFrame({"siglum": list(sigla_rows.keys())})

    for level in range(max_level + 1):
        df[f"level_{level}"] = [
            sigla_rows[siglum].get(level, None)
            for siglum in df["siglum"]]

    for level in range(max_level + 1):
        old_col = f"level_{level}"
        new_col = f"level{level}_new"

        values = sorted(df[old_col].dropna().unique())
        mapping = {old_value: new_value for new_value, old_value in enumerate(values)}

        df[new_col] = df[old_col].map(mapping)

    os.makedirs(os.path.dirname(path), exist_ok = True)
    df.to_csv(path, index = False)

    print(f"Saved nested partitions to {path}")
    return df

# ~ DENDROGRAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def insert_child(node, levels, leaf_name):
    """
    Recursively insert node into hierarchy.
    """
    if not levels:
        node.setdefault("children", []).append({"name": leaf_name})
        return

    level = str(levels[0])
    children = node.setdefault("children", [])
    for child in children:
        if child["name"] == level:
            insert_child(child, levels[1:], leaf_name)
            return

    new_child = {"name": level, "children": []}
    children.append(new_child)
    insert_child(new_child, levels[1:], leaf_name)


def build_hierarchy(df):
    """
    Build nested dict from pandas Dataframe.
    """
    root = {"name": "root", "children": []}
    for _, row in df.iterrows():
        levels = [row["level2_new"], row["level1_new"], row["level0_new"]]
        insert_child(root, levels, row["siglum"])
    return root

def get_dendro_json(input_csv, columns, output_path):
    """
    Get the dendrogram JSON from CSV file and save it to a file.
    """
    df = pd.read_csv(input_csv, usecols=columns)
    tree = build_hierarchy(df)

    with open(output_path, "w") as f:
        json.dump(tree, f, indent=2)



# ~ GRAPH INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


# ~ For comparing graphs using pairs of edges

def thresholded_edges(G, threshold = 0.0):
    """
    Return edges whose weight is at least threshold
    """
    edges = set()

    for u, v, data in G.edges(data = True):
        weight = data.get("weight", 1)

        if weight >= threshold:
            edges.add(tuple(sorted((u, v))))

    return edges


def compare_edge_overlap(G1, G2, threshold = 0.0):
    """
    Compare two feast networks by edge overlap.
    """
    edges1 = thresholded_edges(G1, threshold)
    edges2 = thresholded_edges(G2, threshold)

    shared_edges = edges1 & edges2
    all_edges = edges1|edges2

    overlap = len(shared_edges)/len(all_edges) if len(all_edges) > 0 else 0

    return {
        "network_1": G1.name,
        "network_2": G2.name,
        "threshold": threshold,
        "edges_1": len(edges1),
        "edges_2": len(edges2),
        "shared_edges": len(shared_edges),
        "all_edges": len(all_edges),
        f"edge_overlap_{threshold}": overlap
    }

# ~ Comparing SBM partitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def networks_reduction_on_shared_nodes(G1, G2):
    """
    Reduce two graphs to their shared nodes.
    """
    shared_nodes = set(G1.nodes()) & set(G2.nodes())
    G1_reduced = G1.subgraph(shared_nodes).copy()
    G2_reduced = G2.subgraph(shared_nodes).copy()
    return G1_reduced, G2_reduced