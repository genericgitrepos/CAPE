# CAPE: Convergence-Aware Prediction Engine

**Zero-shot prediction of training convergence epochs — without running full training.**

CAPE (Convergence-Aware Prediction Engine) is a **probing-based meta-learning framework** that predicts how many epochs a deep neural network (DNN) will take to converge, using **only initialization-time signals**.  
By combining analytical descriptors and lightweight probing features, CAPE enables **zero-shot convergence estimation** across architectures and datasets — without relying on any training curves.

<p align="center">
  <img src="./Docs/cape.png" alt="CAPE logo" width="40%" height="40%">
</p>
<p align="center" style="font-size: 11px;">
  [Logo generated using DALL·E 3 by OpenAI]
</p>

---

## 📘 Overview

Training modern deep networks to convergence is **expensive and time-consuming**.  
Existing methods estimate per-iteration or per-epoch costs but fail to predict *how long* a model will take to converge.  
CAPE addresses this by using a **brief probe at initialization** to extract structural and dynamical features that reflect the model’s optimization landscape.

The system then applies a **Random Forest meta-regressor** trained on a diverse meta-dataset to forecast convergence epochs under a **validation-based early-stopping rule** — achieving **high correspondence (r = 0.89)** with true convergence epochs across architectures.

---

## ✨ Highlights

- **Zero-Shot Prediction:** Estimates convergence epochs before any training occurs.  
- **Architecture-General:** Trained across MLPs, CNNs, RNNs, and Transformers.  
- **Probing-Based Features:** Uses lightweight, initialization-only statistics (<1 s per model).  
- **Meta-Learning Design:** Regression-based meta-model trained on a broad meta-dataset.  
- **Robust Generalization:** Accurate even under unseen datasets, architectures, or optimizers.  
- **Validation-Based Stopping:** Predicts epochs to convergence under standard early-stopping.  

---

## ⚙️ Methodology

CAPE operates in three stages:

### 1️⃣ Probing-Based Feature Extraction  
At initialization, CAPE computes a small set of analytical and dynamical features from a single batch:
- **Parameter count** (`P`)
- **Initial loss** (`L₀`)
- **Gradient norm** (`g²`)
- **NTK trace proxy** (`τ`)
- **Learning rate** (`η`)
- **Batch size** (`B`)
- **Dataset size** (`N`)
- **Architecture ID** (`a`)

All features are log-transformed to ensure scale stability.

### 2️⃣ Meta-Dataset Construction  
Each feature vector `z` is paired with the ground-truth convergence epoch `T_conv` measured using a **validation-based early-stopping rule**.  
The resulting meta-dataset spans **diverse architectures and datasets** (CIFAR, TinyImageNet, IMDB, SST2, etc.).

### 3️⃣ Meta-Regressor Training  
A **Random Forest regressor** (200 estimators, depth 8) is trained on log-transformed inputs and targets to minimize:
\[
\frac{1}{M}\sum_j (\log \hat{T}_\text{conv}^{(j)} - \log T_\text{conv}^{(j)})^2
\]
Predicted values are exponentiated to recover the epoch count.

---

## 📊 Experimental Results

CAPE was evaluated on 11 representative models:

- **MLPs:** AS-MLP, MLP-Mixer, ResMLP  
- **CNNs:** ResNet-50, DenseNet-121, MobileNetV2  
- **RNNs:** LSTM, GRU, BiLSTM  
- **Transformers:** DeiT-Tiny, DistilBERT  

Each trained under controlled hyperparameters (LR ∈ {5e-4, 1e-3, 2e-3}, B ∈ {8–256}, optimizers ∈ {Adam, AdamW, SGD, Adafactor}).  

**Performance summary:**

| Evaluation Protocol | MAE ↓ | RMSE ↓ | Pearson r ↑ |
|----------------------|-------|--------|--------------|
| Cross-Fold (5×CV)    | 4.63  | 8.10   | 0.89         |
| Leave-One-Dataset-Out (LODO) | 6.85 | 10.57 | 0.81 |
| Leave-One-Model-Out (LOMO)   | 7.27 | 11.04 | 0.79 |

CAPE outperforms:
- **Learning-Curve Extrapolation (LCE)** — requires partial training.  
- **Scaling-Law Models** — rely only on {log P, log N}.  
- **Probe-Only Variants** — lacking contextual features.

---

## 🚀 Reproducing Experiments

1. **Clone and activate environment**
   ```bash
   git clone https://github.com/genericgitrepos/CAPE
   cd CAPE
   conda env create -f environment.yml
   conda activate cape
   ```

2. **Generate meta-datasets**
   ```bash
   cd "Training"
   python MLP_Train.py
   python CNN_Train.py
   python RNN_Train.py
   python Transformer_Train.py
   ```
   CSVs will be saved locally in the same directory.

3. **Reproduce results**
   All the results from the paper can be reproduced by running the evaluation scripts:
   ```bash
   cd "../Evaluation"
   python CAPE_Evaluation.py
   ```
---

## 🧩 Example Meta-Dataset Entry

| Feature | Description |
|----------|-------------|
| `logP` | Log(Number of trainable parameters) |
| `logL0` | Log(Initial loss at initialization) |
| `logG2` | Log(Average squared gradient norm) |
| `logTau` | Log(NTK trace proxy) |
| `logB` | Log(Batch size) |
| `logLR` | Log(Learning rate) |
| `logN` | Log(Dataset size) |
| `T_conv` | Actual convergence epochs (validation-based) |

---

## 🧪 Citation

If you find this work useful, please cite:

> **CAPE: Generalized Convergence Prediction Across Architectures Without Full Training**  
> *Under review at TMLR, 2025.*

---

## 🔗 Links
- [💻 Project Repository](https://github.com/genericgitrepos/CAPE)
