"""

"""
import pycantus
import pycantus.data as data
from pycantus.filtration import Filter
import copy
import utils
import sbmodel
import argparse
import os
import pandas as pd



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

    return corpus



def build_arg_parser():
    parser = argparse.ArgumentParser(description="Feast-based blockmodeling experiment")
    parser.add_argument('--min_')
    parser.add_argument('--min_chant_per_source', default=100, type=int, help="Minimum number of feasts a source must be associated with to be included in the analysis")
    parser.add_argument('--feast_reduction_threshold', default=0, type=int, help="Minimum number of CIDs a feast must be associated with to be included in the analysis")
    parser.add_argument('--min_cid_per_source_feast', default=0, type=int, help="Minimum number of CIDs a source-feast pair must be associated with to be included in the analysis")
    parser.add_argument('--number_of_iterations', default=20, type=int, help="Number of iterations for the blockmodeling algorithm")
    parser.add_argument('--output_dir', default='results', type=str, help="Directory to save the results")
    return parser.parse_args()


# ~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main(args):
    # Load and clean the dataset
    print("Loading and cleaning the dataset... with minimum chants per source:", args.min_chant_per_source)
    corpus = load_dataset(args.min_chant_per_source)
    sigla_dict = {source.srclink: source.siglum for source in corpus.sources}

    if args.feast_reduction_threshold > 0:
        print(f"Constructing bipartite source-feast graph with feast reduction threshold: {args.feast_reduction_threshold} and minimum CIDs per source-feast pair: {args.min_cid_per_source_feast}")
        g = utils.construct_bipart_source_feast_reducted_graph(corpus, args.feast_reduction_threshold, args.min_cid_per_source_feast)
    else:
        print(f"Constructing bipartite source-feast graph with minimum CIDs per source-feast pair: {args.min_cid_per_source_feast}")
        g = utils.construct_bipart_source_feast_graph(corpus, args.min_cid_per_source_feast)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    utils.save_graph(g, os.path.join(output_dir, f"source_feast_bi_graph_min_{args.min_chant_per_source}.gt"))
    
    model = sbmodel.SBModel()
    model.graph = g
    print()

    # DC
    model.fit_sbm(n_init=args.number_of_iterations)

    # DC-nested
    model.fit_nested_sbm(n_init=args.number_of_iterations)

    # DC-weighted
    #model.fit_sbm_weighted(n_init=args.number_of_iterations)

    # DC-nested-weighted
    #model.fit_nested_sbm_weighted(n_init=args.number_of_iterations)

    # Print best states entropies
    print("Best DC_SBM entropy:", model.best_states['DC_SBM'].entropy())
    print("Best Nested_DC_SBM entropy:", model.best_states['Nested_DC_SBM'].entropy())
    #print("Best DC_SBM_weighted entropy:", model.best_states['Weighted_DC_SBM'].entropy())
    #print("Best Nested_DC_SBM_weighted entropy:", model.best_states['Weighted_Nested_DC_SBM'].entropy())
    print()

    # Save best states
    model.save_states(os.path.join(output_dir, "feast_blockmodeling_results.pkl"))
    print(f"Saved best states to {output_dir}/feast_blockmodeling_results.pkl")
    print()


    print('Starting infering...')
    best_state = model.best_states['DC_SBM']
    index_partitions, sigla_partitions, feasts_partitions = utils.get_partitions_from_state(best_state, sigla_dict)
    print('DC_SBM partitions:')
    print('  Number of partitions:', len(index_partitions))
    print('  Number of sigla partitions:', len(sigla_partitions))
    print('  Number of feast partitions:', len(feasts_partitions))
    print()
    print('Saving DC_SBM partitions...')
    utils.save_partitions_txt(sigla_partitions, feasts_partitions, output_dir)
    source_vs_feast_df = utils.get_sigla_vs_feast_partitions()
    source_vs_feast_df.to_csv(os.path.join(args.output_dir, "dc_source_vs_feast_partitions.csv"), index=False)
    
    best_state = model.best_states['Nested_DC_SBM']
    index_partitions, sigla_partitions, feasts_partitions = utils.get_nested_partitions_from_state(best_state, sigla_dict)
    print('Saving Nested_DC_SBM partitions...')
    utils.save_partitions_txt(sigla_partitions, feasts_partitions, output_dir)
    print('Preparing dendrogram data...')
    utils.save_nested_partitions(sigla_partitions, os.path.join(output_dir, "nested_dc_sigla_partitions.json"))
    utils.get_dendro_json()




if __name__ == '__main__':
    args = build_arg_parser()
    main(args)