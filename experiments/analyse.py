"""
Script to run analysis of different SBM modeling results.
Produces visualisations from collected results.
"""
import utils
import sbmodel
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import graph_tool as gt

MODELS = [
    "DC_SBM",
    "Nested_DC_SBM",
    "Weighted_DC_SBM",
    "Weighted_Nested_DC_SBM"
]

def plot_model_comparison(log_odds, out_path):
    """
    Plot log-likelihood differences across models.
    """
    cmap = plt.colormaps['Dark2']
    colours = {m: cmap(i / len(log_odds)) for i, m in enumerate(log_odds)}

    fig, ax = plt.subplots(figsize = (10, 2), constrained_layout = True)
    ax.axhline(0, color='black', linewidth=1)

    for model, xs in log_odds.items():
        ax.scatter(xs, np.zeros_like(xs), s=60, zorder=3, color=colours[model], alpha=0.3)

    label_positions = []
    for model, xs in log_odds.items():

        x = np.median(xs)
        y = 0.14

        for prev_x, prev_y in label_positions:
            if abs(x - prev_x) < 5000: 
                y = max(y, prev_y + 0.03)

        label_positions.append((x, y))

        ax.text(x, y, model, ha='center', va='bottom', fontsize=12, color=colours[model])

    all_vals = np.concatenate(list(log_odds.values()))

    tick_step = 5000

    ticks = np.arange(
        np.floor(all_vals.min() / tick_step) * tick_step,
        np.ceil(all_vals.max() / tick_step) * tick_step + tick_step,
        tick_step
    )

    for t in ticks:
        ax.plot([t, t], [-0.06, 0.06], color='black', linewidth=0.5)

    ax.set_yticks([])
    ax.set_xlim(left=min(0, all_vals.min() - 100), right=all_vals.max() + 100)
    ax.set_xlabel("Log10 odds relative to best model")

    fig.savefig(out_path, dpi = 300, bbox_inches = "tight")

def create_entropy_plot(models, blockmodeling_fpath="feast_blockmodeling_results.pkl", plot_fpath="model_comparison_plot.png"):
    """
    Create plot for comparing etropies of different models.
    """
    entropies = utils.load_models_entropies(blockmodeling_fpath, models)
    log_odds = utils.compute_log_odds(entropies)
    plot_model_comparison(log_odds, plot_fpath)
    print(f"Created plot at: {plot_fpath}")



def main(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Read models from input directory
    best_states = pickle.load(open(os.path.join(base_dir, args.input_dir, "best_states.pkl"), "rb"))
    print(f"Loaded best states for models: {list(best_states.keys())} from {os.path.join(base_dir, args.input_dir, 'best_states.pkl')}")
    # Creating visualisation comparing models
    create_entropy_plot(best_states.keys(), os.path.join(base_dir, args.input_dir, "best_states.pkl"), os.path.join(base_dir, args.input_dir, "model_comparison_plot.png"))
    print(f"Saved model comparison plot to {os.path.join(base_dir, args.input_dir, 'model_comparison_plot.png')}")
    print()

    # 



def build_arg_parser():
    parser = argparse.ArgumentParser(description="Analyse SBM modeling results.")
    parser.add_argument("--input_dir", type=str, default="sbm_results", help="Directory containing SBM results.")
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    main(args)
    


