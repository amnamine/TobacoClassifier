# 📄 Tobacco-3482 Document Image Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

A highly optimized deep learning project designed to classify vintage corporate document images into 10 distinct categories using the **Tobacco-3482 dataset**. 

This repository features two state-of-the-art approaches:
1.  **CNN-based:** Fine-tuned **EfficientNetV2-S** (Keras/TensorFlow).
2.  **Transformer-based:** Fine-tuned **Vision Transformer (ViT-B/16)** (PyTorch).

Both models are engineered to train successfully within strictly constrained memory environments (like the 12.7 GB RAM limit of Google Colab's free tier) and include robust, CPU-forced graphical interfaces for local deployment of the CNN model.

---

## 📂 Repository Structure

The repository contains the following core files:

*   **`Tobaco_Code_CNN.ipynb`**: Google Colab notebook for the CNN pipeline using **EfficientNetV2-S**. Includes data augmentation (MixUp, CutMix), two-phase training, and evaluation.
*   **`Tobaco_Code_ViT.ipynb`**: Google Colab notebook for the **Vision Transformer (ViT-B/16)** pipeline. This model achieves the highest accuracy in the repository.
*   **`streamlit_app.py`**: A modern web application built with Streamlit for running local inference (currently configured for the CNN model).
*   **`tkinter_interface.py`**: A standalone desktop application built with Tkinter for local inference (currently configured for the CNN model).
*   **`accuracy_loss_curves_CNN.png`**: Training performance curves for the EfficientNet model.
*   **`confusion_matrix_CNN.png`**: Heatmap showing classification results for the EfficientNet model.
*   **`training_curves_ViT.png`**: Training performance curves for the ViT model.
*   **`confusion_matrix_ViT.png`**: Heatmap showing classification results for the ViT model.

*(Note: The trained model weights `efficientnetv2s_tobacco3482.h5` must be generated via the CNN notebook or placed manually into the root directory to run the UI apps).*

---

## 📊 The Dataset

The models are trained on the **Tobacco-3482** dataset, available on Kaggle:
👉 **[Tobacco-3482 Dataset on Kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg)**

It is a subset of the RVL-CDIP database consisting of 3,482 grayscale/RGB document images belonging to the following **10 classes**:
`ADVE`, `Email`, `Form`, `Letter`, `Memo`, `News`, `Note`, `Report`, `Resume`, `Scientific`.

---

## 🧠 Model Architectures & Performance

### 1. Vision Transformer (ViT-B/16) - *Top Performer*
The **ViT-B/16** model (from `torchvision`) achieved the best results:
*   **Validation Accuracy:** **89.20%**
*   **Framework:** PyTorch
*   **Training:** Two-phase training (Linear Probe + Full Fine-tuning) with Cosine Decay.

### 2. EfficientNetV2-S
The **EfficientNetV2-S** model (21.7 million parameters) provides high efficiency:
*   **Validation Accuracy:** **86.22%**
*   **Framework:** Keras 3 / TensorFlow
*   **Key Features:** GeM Pooling, Label Smoothing, and heavy augmentation (MixUp, CutMix, Random Erasing).

### Low-RAM Optimization (Colab Survival Guide)
Both pipelines are optimized for low-RAM environments:
*   **No `.cache()`:** Bypassed standard caching to prevent memory hoarding.
*   **Restricted Parallelism:** Limited CPU threads and prefetch buffers to avoid OOM crashes on machines with < 13GB RAM.

---

## 🛠️ The Keras 3 "Quantization" Bug Fix

**The Problem:** Models trained and saved in newer versions of Google Colab (TensorFlow 2.16+ / Keras 3) inject a `quantization_config` argument into `Dense`, `Dropout`, and `BatchNormalization` layers. Loading these `.h5` files on local machines with older/different TensorFlow versions causes a fatal `ValueError: Unrecognized keyword arguments passed to Dense`.

**The Solution:** Both `streamlit_app.py` and `tkinter_interface.py` include a custom architecture re-builder and a **Safe Layer Wrapper**. 
These scripts intercept the corrupt Keras configuration and forcefully delete the `quantization_config` keyword *before* the model parses it, allowing modern Colab models to load flawlessly on any machine.

---

## 🚀 How to Run the Applications

Both provided applications are configured to **force CPU usage** (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`). This ensures they will not crash on local machines lacking configured Nvidia CUDA drivers.

### Prerequisites
1. Ensure Python 3.9+ is installed.
2. Install the required libraries:
```bash
pip install tensorflow numpy pillow streamlit
```
3. Ensure your trained `efficientnetv2s_tobacco3482.h5` file is in the same directory as the scripts.

### Option A: Run the Streamlit Web App (Recommended)
Streamlit provides a beautiful, responsive, and modern browser interface.
1. Open your terminal/command prompt.
2. Navigate to the repository folder.
3. Run the following command:
```bash
streamlit run streamlit_app.py
```
4. The app will open automatically in your default web browser.

### Option B: Run the Tkinter Desktop App
For a native, standalone windowed application:
1. Open your terminal/command prompt.
2. Navigate to the repository folder.
3. Run the following command:
```bash
python tkinter_interface.py
```

---

## 📈 Results
The model achieves a highly competitive **~85.2% Validation Accuracy** entirely through visual layout recognition (without utilizing Optical Character Recognition (OCR) to read the text). Please review `confusion_matrix.png` and `accuracy_loss_curves.png` for detailed per-class precision and recall metrics.