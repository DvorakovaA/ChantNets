"""
Script to run analysis of different SBM modeling results.
Produces visualisations from collected results.
"""
import utils
import sbmodel
import os
import matplotlib.pyplot as plt
import numpy as np

N_INIT = 20
MIN_CHANTS_PER_SOURCE = 100
DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))

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

def create_entropy_plot(models, blockmodeling_fname="feast_blockmodeling_results.pkl"):
    """
    Create plot for comparing etropies of different models.
    """
    blockmodeling_fpath = os.path.join(DIR_ABS_PATH, "results", blockmodeling_fname)
    entropies = utils.load_models_entropies(blockmodeling_fpath, models)
    log_odds = utils.compute_log_odds(entropies)
    plot_fpath = os.path.join(DIR_ABS_PATH, "visual", f"model_comparison_{N_INIT}_min_{MIN_CHANTS_PER_SOURCE}.pdf")
    plot_model_comparison(log_odds, plot_fpath)
    print(f"Created plot at: {plot_fpath}")

if __name__ == '__main__':

    print("Creating entropy comparison plot.")
    create_entropy_plot(MODELS)
    


