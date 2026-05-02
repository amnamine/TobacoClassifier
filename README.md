```markdown
# 📄 Tobacco-3482 Document Image Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

A highly optimized deep learning project designed to classify vintage corporate document images into 10 distinct categories using the **Tobacco-3482 dataset**. 

This project utilizes a fine-tuned **EfficientNetV2-S** architecture. It is uniquely engineered to train successfully within strictly constrained memory environments (like the 12.7 GB RAM limit of Google Colab's free tier) and includes robust, CPU-forced graphical interfaces for local deployment.

---

## 📂 Repository Structure

Based on the project archive[cite: 2], the repository contains the following files:

*   **`TobacoCode2.ipynb`**: The main Google Colab notebook containing the complete data pipeline, augmentations, model architecture, two-phase training strategy, and evaluation metrics[cite: 2].
*   **`streamlit_app.py`**: A modern, lightweight web application built with Streamlit for running local inference via browser[cite: 2].
*   **`tkinter_interface.py`**: A standalone desktop application built with Tkinter for users who prefer a native OS graphical interface[cite: 2].
*   **`accuracy_loss_curves.png`**: A visualization of the model's training performance across both training phases[cite: 2].
*   **`confusion_matrix.png`**: A detailed heatmap showing the model's classification accuracy and common misclassifications[cite: 2].

*(Note: The trained model weights `efficientnetv2s_tobacco3482.h5` must be generated via the notebook or placed manually into the root directory to run the UI apps).*

---

## 📊 The Dataset

The model is trained on the **Tobacco-3482** dataset, a subset of the RVL-CDIP database. It consists of 3,482 grayscale/RGB document images belonging to the following **10 classes**:
`ADVE`, `Email`, `Form`, `Letter`, `Memo`, `News`, `Note`, `Report`, `Resume`, `Scientific`.

---

## 🧠 Model Architecture & Training Highlights

### 1. The Model: EfficientNetV2-S
Instead of a lightweight baseline (like EfficientNetB0), this project utilizes the much more powerful **EfficientNetV2-S** (21.7 million parameters). The classification head was fully custom-built featuring:
*   **GeM Pooling:** A concatenation of Global Average Pooling and Global Max Pooling to capture both document layout and distinct visual markers (like logos).
*   **Dense Layers:** Two Dense layers (512 and 256 units) utilizing `gelu` activations.
*   **Regularization:** Heavy Dropout (35%) and Batch Normalization to prevent overfitting.

### 2. Low-RAM Optimization (Colab Survival Guide)
To prevent out-of-memory (OOM) crashes on machines with < 13GB RAM, the `tf.data` pipeline was heavily modified:
*   **No `.cache()`:** Bypassed the standard caching mechanism to prevent memory hoarding of 384x384 uncompressed images.
*   **Restricted Parallel Calls:** Hardcoded `num_parallel_calls=2` instead of using `tf.data.AUTOTUNE` to limit background CPU thread memory consumption.
*   **Limited Prefetch:** Prefetch buffer strictly set to `1`.

### 3. Advanced Augmentation
To combat the dataset's small size, the model utilizes:
*   **MixUp & CutMix:** Blending images and labels dynamically to force the model to learn localized features.
*   **Random Erasing:** Randomly blacking out chunks of documents so the model doesn't over-rely on a single feature (like a header).
*   **Standard Augmentation:** Random rotations, zooms, flips, and color/brightness jitter.

### 4. Two-Phase Training
*   **Phase 1 (Warmup):** 10 Epochs with a frozen backbone. Trains only the new custom head using a high learning rate (3e-4) to prevent gradient shock.
*   **Phase 2 (Fine-Tuning):** 50 Epochs with all layers unfrozen. Uses the `AdamW` optimizer with a **Cosine Decay** learning rate schedule, peaking at 5e-5 and settling at 1e-7.

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
```