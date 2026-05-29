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
from pathlib import Path
import json

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

def create_html_dendrogram(json_fpath, html_template_fpath="dendrogram_template.html", html_out_fpath="dendrogram.html"):
    """
    Creates html dendrogram based on existing template. Embeds JSON data inline to avoid CORS issues when opened as a local file.
    """
    json_fpath = Path(json_fpath)
    json_data = json_fpath.read_text(encoding="utf-8")

    svg_fname = json_fpath.with_suffix(".svg").name 
    pdf_fname = json_fpath.with_suffix(".pdf").name

    html = Path(html_template_fpath).read_text(encoding="utf-8")

    html = html.replace(
        'const data = await d3.json("{{JSON_FPATH}}")',
        f'const data = {json_data}'
    )

    html = html.replace("{{SVG_FPATH}}", svg_fname)

    Path(html_out_fpath).write_text(html, encoding="utf-8")


def main(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sigla_dict = pickle.load(open(os.path.join(base_dir, "extra_data/sigla_dict.pkl"), "rb"))

    # Read models from input directory
    best_states = pickle.load(open(os.path.join(base_dir, args.input_dir, "best_states.pkl"), "rb"))
    print(f"Loaded best states for models: {list(best_states.keys())} from {os.path.join(base_dir, args.input_dir, 'best_states.pkl')}")
    # Creating visualisation comparing models
    create_entropy_plot(best_states.keys(), os.path.join(base_dir, args.input_dir, "best_states.pkl"), os.path.join(base_dir, args.input_dir, "model_comparison_plot.png"))
    print(f"Saved model comparison plot to {os.path.join(base_dir, args.input_dir, 'model_comparison_plot.png')}")
    print()

    for model_name, state in best_states.items():
        print(f"Analyzing model: {model_name}")
        if 'nested' in model_name.lower():
            index_partitions, sigla_partitions, feast_partitions = utils.get_nested_partitions_from_state(state['best_state'], sigla_dict)
            # Temporale vs Sanctorale comparison
            utils.get_sanctorale_feasts_in_partitions(feast_partitions[0], os.path.join(base_dir, args.input_dir), prefix=model_name.lower())
            utils.plot_sanctorale_partition_histogram(feast_partitions[0], os.path.join(base_dir, args.input_dir), prefix=model_name.lower())
            utils.plot_sanctorale_partition_histogram_vertical(feast_partitions[0], os.path.join(base_dir, args.input_dir), prefix=model_name.lower())
        else:
            index_partitions, sigla_partitions, feast_partitions = utils.get_partitions_from_state(state['best_state'], sigla_dict)
            # Temporale vs Sanctorale comparison
            utils.get_sanctorale_feasts_in_partitions(feast_partitions, os.path.join(base_dir, args.input_dir), prefix=model_name.lower())
            utils.plot_sanctorale_partition_histogram(feast_partitions, os.path.join(base_dir, args.input_dir), prefix=model_name.lower())
            utils.plot_sanctorale_partition_histogram_vertical(feast_partitions, os.path.join(base_dir, args.input_dir), prefix=model_name.lower())

    create_html_dendrogram(
        os.path.join(base_dir, args.input_dir, "nested_dc_sigla_dendro.json"), 
        os.path.join(base_dir, "dendrogram_template.html"), 
        os.path.join(base_dir, args.input_dir, "nested_dc_sigla_dendro.html")
    )

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Analyse SBM modeling results.")
    parser.add_argument("--input_dir", type=str, default="sbm_results", help="Directory containing SBM results.")
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    main(args)
    


