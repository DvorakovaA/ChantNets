"""
Creates a nested JSON hierarchy for dendrogram visualization from manuscript partition data.
"""

import pandas as pd
import json
from pathlib import Path

INPUT_CSV = Path("partitions/nested_dc_sbm_partitions_20_min_100.csv")
OUTPUT_JSON = Path("visualisations_dendrogram/manuscripts_dendro.json")
COLS = ["siglum", "level0_new", "level1_new", "level2_new"]

def insert_child(node, levels, leaf_name):
    """Recursively insert node into hierarchy."""
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
    """Build nested dict from dataframe."""
    root = {"name": "root", "children": []}
    for _, row in df.iterrows():
        levels = [row["level2_new"], row["level1_new"], row["level0_new"]]
        insert_child(root, levels, row["siglum"])
    return root


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, usecols=COLS)
    tree = build_hierarchy(df)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(tree, f, indent=2)
