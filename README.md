# BMR – Multi-Label Retinal Disease Classification

This repository contains the implementation used for **multi-label retinal disease classification** (Diabetic Retinopathy, Glaucoma, AMD) using deep learning models such as **ResNet18**, **EfficientNet** and several advanced variants (SE blocks, Multi-Head Attention, ensemble learning and CVAE).

The main script is:

- `BMR.py`

---

## Execution Environment

### Recommended: Google Colab (Easiest)

This code is **designed to run on Google Colab** and uses **Google Drive paths** by default.

The paths in `BMR.py` assume:

```
/content/drive/MyDrive/final_project_resources/
```

To run in Colab:

1. Open a new notebook in Colab
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Upload `BMR.py` to Colab in a new cell inside the notebook
4. Ensure your dataset, CSV files and pretrained checkpoints follow the same directory structure as defined in the script.
5. Run the notebook

### Local Execution

If you want to run this outside Colab, you must:

1. Install required libraries (preferably in a virtual environment)

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn pillow
```
2. Update file paths
All dataset and checkpoint paths are hardcoded for Colab at the top of BMR.py.

You must change them to match your local file system, for example:

```python
train_csv = "/local/path/train.csv"
train_image_dir = "/local/path/images/train"
saved_models_path = "/local/path/saved_models"
```

## Data Structure

```
final_project_resources/
│
├── train.csv
├── val.csv
├── offsite_test.csv
├── onsite_test_submission.csv
├── code_template.py
├── README.md
│
├── images/
│   ├── train/
│   ├── val/
│   ├── offsite_test/
│   └── onsite_test/
│
├── pretrained_backbone/
│   ├── ckpt_resnet18_ep50.pt
│   └── ckpt_efficientnet_ep50.pt
│
├── saved_models/
├── submissions/
└── checkpoints/
```

## Implemented Tasks

### Task 1 – Transfer Learning
- **Task 1.1:** No training (evaluation using a pretrained backbone only)
- **Task 1.2:** Frozen backbone with training applied only to the classifier head
- **Task 1.3:** Full fine-tuning of the entire network with class-specific decision thresholds

### Task 2 – Handling Class Imbalance
- **Task 2.1:** Training using **Focal Loss**
- **Task 2.2:** Training using **Class-Balanced Binary Cross-Entropy (BCE) Loss**

### Task 3 – Attention-Based Models
- **Task 3.1:** Integration of **Squeeze-and-Excitation (SE) blocks**
- **Task 3.2:** Integration of **Multi-Head Self-Attention (MHA)** mechanisms

### Task 4 – Ensemble Learning and Generative Modeling
- **Weighted Average Ensemble:** Combines predictions from multiple trained models using predefined weights
- **Conditional Variational Autoencoder (CVAE):** Used for conditional image generation based on disease labels
