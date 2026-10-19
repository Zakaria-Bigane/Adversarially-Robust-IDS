# 🛡️ Adversarially Robust Intrusion Detection System (IDS)

**Bridging Lab Accuracy & Adversarial Reality: Domain-Constrained FGSM Evasion Attacks and Adversarial Training on CIC-IDS2017.**

> *Deep Learning in Cybersecurity (ICCN - INE2) project conducted at the Institut National des Postes et Télécommunications (INPT).*

---

## 🎯 Overview & Motivation

Modern Machine Learning-based Intrusion Detection Systems (IDS) achieve near-perfect accuracy on clean benchmark traffic but remain highly fragile against **adversarial evasion attacks**. In real-world network environments, an attacker cannot arbitrarily modify all tabular features of a network flow (such as destination ports or retroactive TCP flags) without breaking protocol semantics. However, an adversary **can** manipulate physical packet characteristics—such as inserting idle delays (time-stretching) or padding payloads (NOP sleds)—to shift malicious flows across the classifier's decision boundary into the "Benign" region.

This project implements an end-to-end adversarial robustness pipeline on the **CIC-IDS2017 (Wednesday-WorkingHours)** capture (**691,406 flows**, 77 features):
1. **Behavioral Feature Engineering:** Stripping host/session identifiers (IPs, timestamps) to force behavioral learning over instance memorization.
2. **Domain-Constrained Threat Modeling:** Restricting gradient-based perturbations via a binary mask $m \in \{0,1\}^{77}$ to **4 physically mutable features**.
3. **Constrained FGSM Attack:** Demonstrating how bounded perturbations ($\epsilon = 0.05$) trigger a **175× increase in False Negatives** ($40 \rightarrow 7,019$ missed attacks) while remaining statistically stealthy.
4. **PGD-Style Adversarial Training:** Hardening the `RobustIDS` Multi-Layer Perceptron (MLP) with dynamically generated adversarial mini-batches to fully restore detection resilience (**99.8% recall under attack**).

---

## 🔬 Threat Model: Mutable vs. Immutable Features

In a white-box evasion scenario, perturbations are bounded by $\Vert{}\delta\Vert{}_\infty \le \epsilon$ in the MinMax-normalized feature space and restricted strictly to dimensions controllable by the attacker's packet generation:

| Mutable Feature | Physical Manipulation Mechanism | Correlation with Attack ($r$) |
| :--- | :--- | :---: |
| **`Flow Duration`** | Insert idle gaps / time-stretch session | $r = 0.49$ |
| **`Flow IAT Mean`** | Adjust inter-arrival timing jitter | $r = 0.39$ |
| **`Fwd Packet Length Max`** | Pad forward payload bytes (e.g., NOP sleds) | $r = 0.00$ |
| **`Bwd Packet Length Max`** | Trigger/pad backward response payload bytes | $r = 0.63$ |

*The remaining **73 features** (protocol flags, header lengths, destination ports, etc.) are locked via a binary mask tensor $m$ during gradient computation:*

$$x^{\text{adv}} = \text{clip}_{[0,1]}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(f_\theta(x), y)\right) \odot m\right)$$

---

## 🧠 Model Architecture (`RobustIDS`)

A lightweight, regularized Multi-Layer Perceptron built in PyTorch for fast tabular inference and training reproducibility:

```python
class RobustIDS(nn.Module):
    def __init__(self, input_dim):
        super(RobustIDS, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Raw logits for BCEWithLogitsLoss
        )
```

* **Training Setup:** `AdamW` ($\text{lr} = 10^{-3}$, weight decay $\lambda = 10^{-4}$), `BCEWithLogitsLoss`, batch size `128` (train) / `512` (eval), 80/20 stratified train-test split (`seed = 42`), `MinMaxScaler` fitted strictly on training data to prevent data leakage.
* **Adversarial Training Defense:** Mini-batches are doubled on-the-fly by concatenating clean samples $(x, y)$ with constrained FGSM samples $(x^{\text{adv}}, y)$ generated with BatchNorm stats frozen (`model.eval()`).

---

## 📊 Experimental Results

### 1. Baseline Model Fragility under Constrained FGSM ($\epsilon = 0.05$)
On the held-out test set (**138,282 flows**: 87,937 Benign / 50,345 Malicious):

| Condition | Accuracy | Malicious Recall | PR-AUC | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Traffic** | 99.8% | 99.9% | 0.999 | 1,175 | **40** |
| **Constrained FGSM ($\epsilon = 0.05$)** | 93.9% | 86.1% | 0.987 | 2,073 | **7,019** |
| **Impact ($\Delta$)** | **-5.9 pp** | **-13.8 pp** | **-0.012** | +76.4% | **+17,447% (175×)** |

*Across the perturbation sweep $\epsilon \in [0.00, 0.15]$, baseline accuracy degrades monotonically from **99.8%** down to **84.3%**.*

### 2. Hardened Model (Adversarial Training Enabled, $\epsilon = 0.05$)

| Condition | Accuracy | Malicious Recall | F1-Score | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Traffic** | 99.7% | 99.8% | 99.1% | 1,230 | **125** |
| **Constrained FGSM ($\epsilon = 0.05$)** | **99.8%** | **99.8%** | **99.1%** | 1,231 | **126** |
| **Impact ($\Delta$)** | **< 0.1 pp** | **< 0.1 pp** | **0.0 pp** | +1 | **+1** |

> **Key Takeaway:** Adversarial training neutralizes the constrained FGSM evasion attack, reducing missed attacks under adversarial pressure from **7,019** down to **126** while maintaining **99.7%+** accuracy on clean traffic.

---

## 📂 Repository Structure

```text
Adversarially-Robust-IDS/
├── projet_deep_learning.ipynb                 # Full pipeline: EDA, Baseline, FGSM Attack, Defense & Plots
├── requirements.txt                           # Python dependencies
├── docs/
│   ├── Report_Adversarially_Robust_IDS.pdf    # Full Technical & Academic Report
│   └── Presentation_Adversarially_Robust.pdf  # Project Presentation Slides
├── LICENSE                                    # MIT License
└── README.md
```

---

## 🚀 Reproducibility & Usage

All experiments enforce global determinism (`seed = 42` across NumPy, PyTorch CPU/CUDA, and cuDNN).

1. **Clone the repository and install dependencies:**
   ```bash
   git clone [https://github.com/Zakaria-Bigane/Adversarially-Robust-IDS.git](https://github.com/Zakaria-Bigane/Adversarially-Robust-IDS.git)
   cd Adversarially-Robust-IDS
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Execute the Jupyter Notebook:**
   ```bash
   jupyter nbconvert --to notebook --execute projet_deep_learning.ipynb
   ```
   *(Or open `projet_deep_learning.ipynb` interactively in JupyterLab / VS Code / Google Colab. The dataset is fetched automatically via `kagglehub`).*

---

## 📄 Documentation

* 📘 **[Full Technical Report (PDF)](docs/Report_Adversarially_Robust_IDS.pdf)**
* 📊 **[Presentation Slides (PDF)](docs/Presentation_Adversarially_Robust.pdf)**

---

## 👥 Authors

* **Zakaria Bigane** & **Marwan Moujane** — Cybersecurity Engineering Students at **INPT** (Rabat, Morocco).

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.