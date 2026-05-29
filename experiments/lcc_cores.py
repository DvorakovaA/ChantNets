"""
This script compares the largest connected cores of the source-feast networks 
across different feasts and thresholds. It computes the Jaccard similarity between 
the largest cores of each pair of feasts for various thresholds 
and saves the results in a CSV file for further analysis.
"""
import networkx as nx
import itertools
import pandas as pd
import importlib
import utils
import os
import argparse

CORE_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]

def load_nets(nets_dir):
    nets = {}
    for fname in os.listdir(nets_dir):
        if fname.endswith("whole day_network.edgelist"):
            feast_name = fname.replace(".edgelist", "")
            net_path = os.path.join(nets_dir, fname)
            g = nx.read_edgelist(net_path)
            name = feast_name.replace("_whole day_network", "")
            g.name = name
            nets[name] = g
            print(f"Loaded network for feast: {name} with {len(nets[name].nodes())} nodes and {len(nets[name].edges())} edges from {net_path}")
    return nets

def compute_lcc_cores_comparison(nets, dir): 
    rows = []
    for feast1, feast2 in itertools.combinations(nets.keys(), 2):
        for threshold in CORE_THRESHOLDS:
            result = utils.compare_largest_cores(
                nets[feast1],
                nets[feast2],
                threshold
            )
            rows.append(result)

    core_comparison_df = pd.DataFrame(rows)
    core_comparison_df.to_csv(os.path.join(dir, "lcc_core_comparison.csv"), index=False)
    print(f"Saved core comparison results to {os.path.join(dir, 'lcc_core_comparison.csv')}")



def main(args):
    print(f"Loading networks from {os.path.join(args.dir, 'nets')}")
    nets = load_nets(os.path.join(args.dir, "nets"))
    print(f"Loaded {len(nets)} networks: {list(nets.keys())}")
    compute_lcc_cores_comparison(nets, args.dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Analyse SBM modeling results.")
    parser.add_argument("--dir", type=str, required=True, help="Directory to save the comparison results.")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)