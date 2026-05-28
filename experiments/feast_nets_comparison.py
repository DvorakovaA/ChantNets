"""

"""
import pycantus
import pycantus.data as data
from pycantus.filtration import Filter
import copy
import utils
import importlib
import networkx as nx
import os
from contextlib import redirect_stdout
import re
import pandas as pd
import itertools 
import sbmodel
import pickle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse

N_INIT = 20
MIN_CHANTS_PER_SOURCE = 600
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

FEAST_NAMES = [[f'Dom. {i} Quadragesimae', f"Quadragesima, dom. {i}", f"Quadragesimae, dom. {i}"] for i in range(1, 5)] \
             + [['Dom. in Palmis'], ['Dom. Resurrectionis']]

WHOLE_DAY_LABEL = "whole_day"

OFFICE_LABELS = {
    WHOLE_DAY_LABEL: "Whole day",
    "L": "Lauds",
    "V": "Vespers",
    "V2": "Second Vespers",
    "M": "Matins",
}

def load_dataset(min_chants_per_source):

    corpus = data.load_dataset('cantuscorpus_v1.0')

    n_chants = len(corpus.chants)
    print(f'Number of chants in CantusCorpus v1.0: {n_chants}')
    n_sources = len(corpus.sources)
    print(f'Number of sources in CantusCorpus v1.0: {n_sources}')

    # Clean the corpus
    # Drop doxology
    doxo_filter = pycantus.filtration.Filter('doxo_filter')
    doxo_filter.add_value_exclude('cantus_id', '909000')
    corpus.apply_filter(doxo_filter)

    # Drop fragments => sources with less than MIN_CHANTS_PER_SOURCE chants
    corpus.drop_small_sources_data(min_chants=min_chants_per_source)

    n_chants = len(corpus.chants)
    print(f'Number of chants in CantusCorpus v1.0 after cleaning: {n_chants}')
    n_sources = len(corpus.sources)
    print(f'Number of sources in CantusCorpus v1.0 after cleaning: {n_sources}')

    sigla_dict = {source.srclink: source.siglum for source in corpus.sources}

    return corpus, sigla_dict


def build_corpora(corpus, feast_names):

    corpora = {
        office_code: {}
        for office_code in OFFICE_LABELS
    }

    for feast in feast_names:

        feast_name = feast[0]

        whole_day_corpus = copy.deepcopy(corpus)
        feast_filter = Filter(f'{feast_name}_whole_day_filter')
        feast_filter.add_value_include('feast', list(feast))
        whole_day_corpus.apply_filter(feast_filter)
        whole_day_corpus.drop_empty_sources()

        corpora[WHOLE_DAY_LABEL][feast_name] = whole_day_corpus

        for office_code, office_label in OFFICE_LABELS.items():

            if office_code == WHOLE_DAY_LABEL:
                continue

            feast_corpus = copy.deepcopy(whole_day_corpus)
            office_filter = Filter(f'{feast_name}_{office_code}_filter')
            office_filter.add_value_include('office', office_code)
            feast_corpus.apply_filter(office_filter)
            feast_corpus.drop_empty_sources()

            corpora[office_code][feast_name] = feast_corpus

    return corpora
    
def create_graphs(corpora, path):

    nets = {}

    for office_code, feast_corpora in corpora.items():
        office_networks = {}

        for feast_name, feast_corpus in feast_corpora.items():

            graph = utils.build_feast_network(feast_corpus, feast_name)
            office_networks[feast_name] = graph
            # clean the feast name, every character not a letter, digit, underscore or hyphen is replaced by underscore
            filename = f"{re.sub(r'[^A-Za-z0-9_-]+', '_', feast_name)}_{OFFICE_LABELS[office_code].lower()}_network.edgelist"

            output_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), path, "nets", filename)
            nx.write_edgelist(graph, output_fpath)

        nets[office_code] = office_networks

    return nets

def print_nets_info(nets):

    sizes_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nets", "feast_office_net_sizes.txt")
    info_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nets", "feast_office_net_info.txt")

    with open(sizes_fpath, 'w') as f_sizes:
        with open(info_fpath, 'w') as f_info:
            for office_code, office_nets in nets.items():
                for feast_name, graph in office_nets.items():

                    with redirect_stdout(f_sizes):
                        print(
                            f"{feast_name} {OFFICE_LABELS[office_code]}: "
                            f"{graph.number_of_nodes()} nodes, "
                            f"{graph.number_of_edges()} edges"
                        )

                    with redirect_stdout(f_info):
                        print("{:>12s} | '{:s}'".format('Office', OFFICE_LABELS[office_code]))
                        utils.graph_info_nx(graph, fast=False)
                        print("--------------------------------")

def compare_edgewise_networks(nets):

    overlap_dfs = {}

    for office_code, networks in nets.items():
        rows = []
        for feast1, feast2 in itertools.combinations(networks.keys(), 2):
            row = {'network_1': feast1, 'network_2': feast2}
            for threshold in THRESHOLDS:
                result = utils.compare_edge_overlap(networks[feast1], networks[feast2], threshold=threshold)
                row['edges_1'] = result['edges_1']
                row['edges_2'] = result['edges_2']
                row['shared_edges'] = result['shared_edges']
                row['edge_overlap_' + str(threshold)] = result[f'edge_overlap_{threshold}']
            rows.append(row)
        overlap_dfs[office_code] = pd.DataFrame(rows)
    return overlap_dfs

def perform_weighted_sbm_analysis(network, sigla_dict):
    model = sbmodel.SBModel()
    model.load_graph_nx(network)
    model.fit_sbm_weighted(weight_label='weight', n_init=N_INIT)
    best_state = model.best_states['Weighted_DC_SBM']
    partitions = utils.get_partitions_from_state(best_state, sigla_dict=sigla_dict)
    return partitions

def prepare_sbm_by_pairs(networks, sigla_dict):
    sbm_results_by_pairs = {}
    for feast1, feast2 in itertools.combinations(networks.keys(), 2):
        network1, network2 = utils.networks_reduction_on_shared_nodes(networks[feast1], networks[feast2])
        partition1 = perform_weighted_sbm_analysis(network1, sigla_dict)[1]
        partition2 = perform_weighted_sbm_analysis(network2, sigla_dict)[1]
        sbm_results_by_pairs[(feast1, feast2)] = (partition1, partition2)
    
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual", "sbm_results.pkl")
    with open(fpath, 'wb') as f:
        pickle.dump(sbm_results_by_pairs, f)

    return sbm_results_by_pairs

def transform_partition_to_labels(partition):
    node_to_block = {}
    for block, nodes in partition.items():
        for node in nodes:
            node_to_block[node] = block
    return node_to_block

def compare_weighted_sbm_partitions(sbm_on_nets):
    rows = []
    for (feast1, feast2), (partition1, partition2) in sbm_on_nets.items():
        row = {'network_1': feast1, 'network_2': feast2}
        row['num_partitions_1'] = len(partition1)
        row['num_partitions_2'] = len(partition2)

        partition1 = transform_partition_to_labels(partition1)
        partition2 = transform_partition_to_labels(partition2)

        shared_nodes = sorted(set(partition1) & set(partition2))
        labels1 = [partition1[node] for node in shared_nodes]
        labels2 = [partition2[node] for node in shared_nodes]

        row['n_shared_nodes'] = len(shared_nodes)
        row['adjusted_rand_index'] = adjusted_rand_score(labels1, labels2)
        row['normalized_mutual_info'] = normalized_mutual_info_score(labels1, labels2)

        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    return comparison_df

def summary_accross_feasts(nets, overlap_dfs, sigla_dict, path):
    edge_comparison_df = overlap_dfs[WHOLE_DAY_LABEL]
    sbm_results_by_pairs = prepare_sbm_by_pairs(nets[WHOLE_DAY_LABEL], sigla_dict)
    sbm_comparison_df = compare_weighted_sbm_partitions(sbm_results_by_pairs)
    comparison_df = edge_comparison_df.merge(sbm_comparison_df, on=["network_1", "network_2"])

    csv_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), path, "comparison.csv")
    comparison_df.to_csv(csv_fpath, index=False)
    print(f"Feast comparison saved to {csv_fpath}")

def summary_accross_office(overlap_dfs, path="visual"):
    office_comparison_df = None

    for office_code, df in overlap_dfs.items():
        df = df.rename(columns={
            col: f"{col}_{office_code}"
            for col in df.columns
            if col not in ["network_1", "network_2"]
        })

        if office_comparison_df is None:
            office_comparison_df = df
        else:
            office_comparison_df = office_comparison_df.merge(df, on=["network_1", "network_2"], how="outer")

    csv_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), path, "office_edge_overlap_comparison.csv")
    office_comparison_df.to_csv(csv_fpath, index = False)
    print(f"Office comparison saved to {csv_fpath}")



if __name__ == '__main__':

    # The file accepts single (optional) argument --feast-names-fpath with a name of the feast in each line
    parser = argparse.ArgumentParser()
    parser.add_argument("--feast-names-fpath", help="File path of feast names, variants on one line separated by ';'.", type=str, default="")
    parser.add_argument('--min_chant_per_source', default=MIN_CHANTS_PER_SOURCE, type=int, help="Minimum number of chants a source have to be included in the analysis")
    args = parser.parse_args()

    if args.feast_names_fpath == "":
        print("No feast names file provided, using default feast names.")
        feast_names = FEAST_NAMES
        results_path = "visual"
        nets_path = ""
    else:
        print(f"Loading feast names from {args.feast_names_fpath}")
        with open(args.feast_names_fpath, 'r') as f_feast_names:
            feast_lines = [line.strip() for line in f_feast_names if line.strip()]
            feast_names = [line.split(';') for line in feast_lines]
        # save results beside the file with feast names
        results_path = os.path.dirname(os.path.abspath(args.feast_names_fpath))
        nets_path = os.path.dirname(os.path.abspath(args.feast_names_fpath))
    
    # Load data
    print("Loading and cleaning the dataset... with minimum chants per source:", args.min_chant_per_source)
    corpus, sigla_dict = load_dataset(minimum_chants_per_source=args.min_chant_per_source)

    # Data manipulation
    corpora = build_corpora(corpus, feast_names)
    nets = create_graphs(corpora, nets_path)
    print_nets_info(nets)
    overlap_dfs = compare_edgewise_networks(nets)

    # Comparisons
    summary_accross_feasts(nets, overlap_dfs, sigla_dict, results_path)
    summary_accross_office(overlap_dfs, results_path)
    
