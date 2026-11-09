# Layer-wise Cross-Modal Fusion in Graph Neural Networks for Multimodal Node Classification

**Tayyab Ur Rehman Rathore**
Course project, **ESOF-5052 Big Data Machine Learning**, Lakehead University, Fall 2026

Full documentation as PDF: [README.pdf](README.pdf)

## 1. What this project is about

In a **multimodal graph**, every node has a text description *and* an image (for example a product's title and
photo), and edges connect related nodes (products bought together). The task is **node classification**:
predicting each node's class (for example a product category or a book genre) from its own text and image and
from its neighbors.

Existing graph neural networks combine the two modalities either **once at the input** (early fusion:
concatenating text and image features, as in GraphSAGE) or **once at the output** (late fusion: separate networks
per modality, as in MMGCN and MGAT). The MM-Graph benchmark (Zhu et al., CVPR 2025) found that late-fusion models
do not reliably beat simple early fusion.

This project proposes a **two-stream GNN with cross-modal fusion at every layer**: each modality keeps its own
stream, and before every message-passing layer the two streams are fused and the fused signal is added back to
both. The goal is to let text and image inform each other while neighborhood information is being gathered, at a
cost comparable to GraphSAGE.

## 2. Method in brief

For each node, the text and image features are L2-normalized separately and projected to *d* dimensions. Then, at
every layer *ℓ*:

1. **fuse:** *f* = φ(*h*<sub>text</sub>, *h*<sub>image</sub>) (average of the two streams, or a learned gate);
2. **inject:** add *f* to both streams;
3. **aggregate:** each stream runs its own GraphSAGE layer over its sampled neighbors, followed by LayerNorm,
   ReLU and Dropout.

After the last layer the two streams are concatenated and a linear classifier predicts the class. See Section IV
of the report for the equations, the architecture figure and the data/operation flow.

## 3. Repository structure

```
layerwise-fusion-gnn/
├── README.md               this file
├── README.pdf              this documentation as PDF
├── requirements.txt        Python dependencies
├── run_experiment.py       command-line entry point: train, evaluate, summarize, plot
├── run_analysis.py         significance tests, per-class and efficiency tables, published comparison
├── src/
│   ├── config.py           all hyperparameters (the training-setup table of the report)
│   ├── data.py             loading, validation, cleaning, per-modality normalization
│   ├── augment.py          modality dropout and edge dropout (training only)
│   ├── models.py           proposed model (fusion, fusion-gate, late) and baselines
│   ├── train.py            training loop, cosine schedule, best-validation selection
│   ├── evaluate.py         test metrics, inference time, result storage
│   ├── analysis.py         statistics and report tables
│   └── plots.py            figures for the report (600 dpi PNG + PDF)
├── notebooks/
│   ├── 01_EDA.ipynb        exploratory data analysis, validation, bias analysis
│   └── 02_training_demo.ipynb   short documented end-to-end training run
├── results/                results.json (all runs) and summary tables
└── figures/                figures (created when running)
```

## 4. Requirements

**Hardware:** a CUDA GPU is recommended. Development used an NVIDIA RTX 4060 (8 GB) on Windows 11 with WSL2
(Ubuntu). For Goodreads-NC, use `--fp16` and, if needed, fewer sampled neighbors to fit in 8 GB.

**Software:** Python 3.10 to 3.12, PyTorch 2.x and the matching DGL build, plus the packages in
`requirements.txt`:

```bash
# 1) install PyTorch and DGL for your CUDA version (see pytorch.org and dgl.ai)
# 2) then:
pip install -r requirements.txt
```

## 5. Data

Two node-classification datasets from the MM-Graph benchmark:

| Dataset | Nodes | Edges\* | Classes | Node text | Node image | Edge |
|---|---|---|---|---|---|---|
| Ele-Fashion | 97,766 | 199,586 | 11 used (ids 0–11) | product title | product photo | bought together |
| Goodreads-NC | 685,294 | 6,841,179 | 10 | book description | book cover | liked by the same readers |

\* Undirected, after removing self-loops and duplicate edges.

Download into `data/` (about 2.3 GB for Ele-Fashion, 17 GB for Goodreads-NC with all four feature files):

```bash
pip install -U huggingface_hub
hf download mm-graph-org/mm-graph --repo-type dataset --include "ele-fashion/*" --local-dir ./data
hf download mm-graph-org/mm-graph --repo-type dataset --include "books-nc/*" --local-dir ./data
```

Each dataset folder contains `nc_edges-nodeid.pt`, `labels-w-missing.pt`, `split.pt` and one feature file per
encoder pair: `clip_feat.pt`, `imagebind_feat.pt`, `t5vit_feat.pt`, `t5dino_feat.pt` (text half first, image half
second). Source: Zhu et al., *Mosaic of Modalities*, CVPR 2025,
https://huggingface.co/datasets/mm-graph-org/mm-graph.

## 6. Usage

**Smoke test** (1 seed, 2 epochs, proposed model and GraphSAGE; about a minute on Ele-Fashion):

```bash
python run_experiment.py --quick
```

**Full experiment** (30 epochs, seeds 0, 1, 2):

```bash
python run_experiment.py --dataset ele-fashion --feature clip --models fusion late sage gcn gat
python run_experiment.py --dataset books-nc --feature clip --models fusion late sage --fp16
```

**Useful options:** `--feature {clip,imagebind,t5vit,t5dino}`, `--models fusion fusion-gate late sage gcn gat`,
`--epochs`, `--seeds`, `--num-neighbors`, `--modality-dropout`, `--edge-dropout`, `--class-weighted-loss`,
`--no-layer-norm`, `--fp16`, `--data-dir`. Run `python run_experiment.py --help` for the full list.

**Analysis** of the saved runs (summary, significance tests, per-class F1, efficiency, published comparison):

```bash
python run_analysis.py --dataset ele-fashion --feature clip --reference fusion
```

**Outputs:**

- `results/results.json`: every run (metrics, per-class F1, confusion matrix, training history, settings);
- `results/summary_<dataset>_<feature>.csv`: mean and standard deviation over seeds per model;
- `results/analysis_<dataset>_<feature>/`: analysis tables (CSV) and the results table in LaTeX;
- `figures/curves_*`, `figures/accuracy_*`, `figures/macro_f1_*`, `figures/per_class_f1_*`: training curves,
  model comparison and per-class results.

Finished runs are skipped when a command is repeated, so an interrupted experiment can be restarted safely.

## 7. Evaluation protocol

- The benchmark's official random 6/1/3 train/validation/test split; the splits are checked to be disjoint.
- Every model uses the same settings (`src/config.py`).
- After every epoch, accuracy is measured on the **validation** set only; the best-validation weights are kept.
- The **test** set is used once per run, with those weights.
- Reported: accuracy, macro-F1 (all classes weighted equally), per-class F1, parameters, peak GPU memory, training
  time per epoch and inference time per node; mean and standard deviation over three seeds.
