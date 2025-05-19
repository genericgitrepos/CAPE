# CAPE: Convergence-Aware Prediction Engine

**Zero-shot prediction of training steps for deep neural networks — without running full training.**

CAPE is a lightweight system that estimates the number of steps needed for a deep learning model to converge, using only initialization-time signals. This helps researchers and practitioners quickly estimate training time and optimize resource allocation across unseen architectures and datasets.

<p align="center">
  <img src="./docs/cape.png" alt="CAPE logo" width="40%" height="40%">
</p>
<p align="center" style="font-size: 11px;">
  [This logo is generated using DALL.E 3 by OpenAI]
</p>

---

## 📌 Highlights

- **Architecture-Agnostic**: Generalizes across MLPs, CNNs, RNNs, and Transformers.
- **Fast Probing**: Extracts features in <1 second using a single batch.
- **Meta-Learning**: Predicts convergence using an XGBoost meta-regressor trained on a diverse meta-dataset.
- **Robust Evaluation**: Outperforms curve-extrapolation baselines, even on unseen datasets (e.g., KMNIST, EMNIST).
- **No Full Training Required**: Save hours of GPU compute for each configuration.

---

## 🛠️ How It Works

CAPE performs the following:

1. **Probe the Model**: At initialization, CAPE extracts computational features:
   - Parameter count (`P`)
   - Gradient norm (`g²`)
   - NTK trace proxy (`τ`)
   - Learning rate (`η`), batch size (`B`), dataset size (`N`)
2. **Build Feature Vector**: All features are log-transformed.
3. **Predict Steps**: A trained XGBoost model maps features to predicted convergence steps (`T*`).

---

## 📁 Repository Structure

```
CAPE/
├── docs/                      # Diagrams and paper figures
├── evaluation/                # Contains all the jupyter notebooks needed for reproducing the results
├── meta_dataset_generators/   # Scripts to generate meta-datasets via probing + convergence
├── meta_datasets/             # Our gathered data for meta-datasets
├── .gitignore                        
├── environment.yml            # Environment file for conda
└── README.md                  # This file
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/genericgitrepos/CAPE
cd CAPE
```

### 2. Set up the environment
We recommend using the provided `conda` environment:
```bash
conda env create -f environment.yml
conda activate cape
```

### 3. Run meta-dataset generation
```bash
python meta_dataset_generators/generate.py --model {model-name} --save_dir {dir-to-save}
```
- Replace `{model-name}` with one of the following:
  - `mlp`
  - `cnn`
  - `rnn`
  - `transformer`
- Replace `{dir-to-save}` with the directory where you want to save the generated meta-dataset.

---

## 📈 Example: Meta-Dataset Entry

| Feature     | Description                          |
|-------------|--------------------------------------|
| `logP`      | log(Number of trainable parameters)  |
| `logG2`     | log(Average squared gradient norm)   |
| `logTau`    | log(NTK trace proxy)                 |
| `logB`      | log(Batch size)                      |
| `logLR`     | log(Learning rate)                   |
| `logN`      | log(Dataset size)                    |
| `T_star`    | Actual steps to convergence          |

---

## 🔁 Reproducing Results

To reproduce the experiments and plots from the paper:

1. After activating the conda environment, navigate to the `evaluation/` directory.
2. Open the relevant Jupyter notebooks (e.g., `mlp_eval.ipynb`, `transformer_eval.ipynb`).
3. Make sure the `reuse` flag is set to `True` in the notebook cells.
4. Run the notebook to see results and visualizations directly.

> These notebooks will load the pre-generated meta-datasets and replay evaluations without repeating the heavy computations.

---

## 🧪 Citation

> 📄 **CAPE: Generalized Convergence Prediction Across Architectures Without Full Training**  
> *Submitted to NeurIPS 2025*

---
