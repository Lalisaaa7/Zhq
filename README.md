# # Protein-DNA Binding Site Prediction with ESM, GNN, and Diffusion Models

This project implements a deep learning pipeline for predicting protein-DNA binding sites. It integrates pre-trained protein language models (ESM-2), graph neural networks (GNNs), and diffusion models for positive sample augmentation.

## 🔍 Key Features

- Utilizes ESM-2 to extract amino acid sequence features
- Converts protein sequences to graphs using sliding window connectivity
- Employs GCN/GAT-based models for node classification (binding site prediction)
- Trains a diffusion generator to synthesize positive samples and address class imbalance
- Supports enhanced graph construction (G*) using edge prediction
- Includes batch testing and ROC curve visualization

## 📁 Project Structure

```
.
├── main.py                      # Main training + testing pipeline
├── batch_test.py                # Evaluate on all test sets and draw ROC
├── test.py                      # Inference on a single test file
├── modules/
│   ├── model.py                 # GCN/GAT/GINE definitions
│   ├── diffusion.py             # Diffusion generator logic
│   ├── edge_predictor.py        # Predicts connections for generated nodes
│   └── data_loader.py           # Data reading and graph building
├── diffusion_gnn_mask.py       # GNN with time-dependent diffusion mask
├── diffusion_model.py          # Lightweight generator for positive synthesis
├── data_loader_from_raw.py     # ESM-powered feature extraction & graphing
├── Raw_data/                    # Protein .txt datasets (sequence + label)
├── Weights/                     # Model weight files
└── ROC_Curves/                  # Output folder for ROC plots
```

## 📦 Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn matplotlib
pip install fair-esm
```

## 📄 Data Format

Each sample consists of 3 lines in a `.txt` file:

```
>Sequence_ID
AMINOACIDSEQUENCE
0010100000000000000000 (binding site labels)
```

## 🚀 Quick Start

### 1. Train and test model

```bash
python main.py
```

This performs:
- Dataset loading and splitting
- Diffusion model training on positive samples
- Graph augmentation and edge generation
- GNN model training and evaluation

### 2. Batch test on multiple test sets

```bash
python batch_test.py
```

Runs tests over all `Test` files in `Raw_data/` and saves ROC curves.

### 3. Single test file evaluation

```bash
python test.py
```

(Default test set: `Raw_data/DNA-129_Test.txt`)

## 🧬 Model Architecture

- **Feature Extractor:** ESM-2 (1280-dim per amino acid)
- **Graph Construction:** Sliding window (e.g., ±5)
- **GNN Classifiers:** GCN / GAT / GINE + MLP
- **Loss:** Focal Loss (for imbalanced data)

## 💡 Diffusion Strategy

- Trains `DiffusionModel` on positive class only
- Generates new positive node features
- Uses `EdgePredictor` to connect synthetic nodes into enhanced graph (G*)
- Combines original + synthetic data for final GNN training

## 📊 Evaluation Metrics

- Accuracy, F1 Score, MCC, AUC
- ROC Curve with AUC score
- Recall, Precision, Specificity (optional)

## ⚠ Notes

- ESM-2 downloads weights on first run (requires internet)
- Make sure label length = sequence length in all datasets
- Save weights to `Weights/best_model.pt`
- Keep feature dimensions consistent across scripts

## 🙋 Contact

Please use GitHub Issues or contact the author for any questions or contributions.
