# ChantNets

Repository for INA course project taken at FRI UL: **Exploring Gregorian Chant Repertoire Stability with Networks**.  
  
By Anna Dvořaková, Ana Nikolić and Luka Pantar.

## Prerequisites & Installation

This project runs on Python (version 3.10-3.12), so prepare Python virtual envirtonment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

The project uses **`graph-tool`** library, which is not installable via standard `pip`. Refer to the [graph-tool installation](https://graph-tool.skewed.de/installation.html) for detailed instructions.


## Reproducing the Results

All scripts are located and should be run from the `experiments` directory.

### A. Bipartite Source-Feast Blockmodeling (SBM)

This constructs a bipartite network linking manuscript sources to feasts and fits four different SBM variants:
 - Degree-Corrected
 - Nested Degree-Corrected
 - Weighted Degree-Corrected
 - Weighted Nested Degree-Corrected

**Fit the SBM Models**:
   ```bash
   python source_feast_blockmodeling.py
   ```
   - `--min_chant_per_source`: Minimum number of feasts a source must be associated with to be included in the analysis (default: `100`).
   - `--feast_reduction_threshold`: Minimum number of CIDs a feast must be associated with to be included in the analysis (default: `0`).
   - `--min_cid_per_source_feast`: Minimum number of CIDs a source-feast pair must be associated with to be included in the analysis (default: `0`).
   - `--number_of_iterations`: Number of iterations for the blockmodeling algorithm (default: `20`).
   - `--output_dir`: Directory to save the results (default: `results`)
   - `--do_vs_partitions`: Whether to compute source vs feast partitions dataframe (default: `False`).
   
**Analyze and Plot SBM Results**:
   Run `analyse.py` to read the best states from SBM fitting, compute weight distributions, plot weight descriptions, output Temporale vs Sanctorale partitions, and generate dendrograms for nested models.
   ```bash
   python analyse.py
   ```
   - `--output_dir`: Directory containing SBM results. (default: `results`)

### B. Feast Network Comparison

This script compares chant networks generated for specific feasts (e.g., Lent and Advent Sundays) across whole day and four different offices (Lauds, Matins, Vespers, Second Vespers).

**Compare Feast Networks**:
   Run `feast_nets_comparison.py` to construct co-occurrence networks based on shared chants, write network edgelists, compare edgewise overlap under different thresholds, and check community structure alignment via ARI/NMI.
   ```bash
   python feast_nets_comparison.py
   ```
   - `--feast-names-fpath`: File path of feast names, variants on one line separated by ';'. If left empty, default Sundays of Lent plus Palm and Resurrection Sundays are used.
   - `--min_chant_per_source`: Minimum number of chants a source have to be included in the analysis (default: `200`).


#### Largest Connected Component (LCC) Core Analysis

To evaluate Jaccard similarity between the largest connected components of the feast networks across various edge weight thresholds:
```bash
python lcc_cores.py --dir results
```
- `--dir`: Directory to read the nets from `\nets` subdirectory and to save the comparison results into.
