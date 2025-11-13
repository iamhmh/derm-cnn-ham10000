---
library_name: pytorch
tags:
  - image-classification
  - cnn
  - dermatology
  - ham10000
  - computer-vision
pipeline_tag: image-classification
license: cc-by-nc-4.0
---

# derm-cnn-ham10000

**A convolutional neural network trained on the HAM10000 dataset for multi-class skin lesion classification.**

This model predicts **7 skin lesion categories** from dermatoscopic images.  
It is lightweight, easy to deploy, and comes with an inference script for quick testing.

---

## 🧠 Model Details

**Architecture:** Custom CNN (4 conv blocks + 5 fully-connected layers)  
**Input:** RGB image resized to **28×28**  
**Output:** 7-class logits  
**Framework:** PyTorch  
**Weights:** `model.pth`  

### Classes
| Index | Label |
|-------|-------------------------------|
| 0 | Actinic keratoses (akiec) |
| 1 | Basal cell carcinoma (bcc) |
| 2 | Benign keratosis (bkl) |
| 3 | Dermatofibroma (df) |
| 4 | Melanoma (mel) |
| 5 | Melanocytic nevi (nv) |
| 6 | Vascular lesions (vasc) |

`labels.json` contains this mapping.

---

## 📊 Performance

Metrics computed on the official HAM10000 split:

- **Accuracy:** 0.99  
- **Macro F1-score:** 0.99  
- **Weighted F1-score:** 0.99  

Class-level summary:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| akiec | 1.00 | 1.00 | 1.00 |
| bcc | 0.99 | 1.00 | 0.99 |
| bkl | 0.98 | 1.00 | 0.99 |
| df | 1.00 | 1.00 | 1.00 |
| mel | 0.99 | 0.93 | 0.96 |
| nv | 1.00 | 1.00 | 1.00 |
| vasc | 0.96 | 0.99 | 0.98 |

Full report available in `classification_report.txt`.

---

## 🚀 How to Use

### Install dependencies
```bash
pip install torch torchvision numpy pillow
```

### Load the model
```python
import torch
from model import load_model
from inference import predict

pred_idx, label, probs = predict("example.jpg", "model.pth")
print(label)
```

### CLI usage
```bash
python inference.py path/to/image.jpg --weights model.pth
```

---

## 📁 Repository Structure

```
model.py                # CNN architecture + load_model()
inference.py            # Run prediction on an input image
model.pth               # Trained weights
labels.json             # Class index → label
classification_report.txt
assets/                 # Confusion matrix, training curves (optional)
```

---

## 🧪 Training Data

**Dataset:** HAM10000  
**License:** CC BY-NC 4.0  

**Preprocessing:**
- resize to 28×28
- normalized to [0,1]
- no dataset augmentation used in the published version

---

## ⚠️ Limitations

- The model is trained only on HAM10000 at 28×28 resolution.
- Predictions must **not** be used for medical diagnosis.
- HAM10000 includes class imbalance, which may affect edge cases.
- Performance on clinical camera photos is not guaranteed.

---

## 📄 License

**Model weights:** CC BY-NC 4.0 (non-commercial use only)  
**Code:** MIT License  

This restriction comes from the HAM10000 dataset license.

---

## ✏️ Citation

If you use this model, please cite the HAM10000 dataset:

```
Tschandl, P., Rosendahl, C., & Kittler, H. (2018). 
The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. 
Scientific Data, 5, 180161.
```
