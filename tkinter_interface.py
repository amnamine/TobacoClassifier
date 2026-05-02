# ==============================================================================
# 1. FORCE 100% CPU & SILENCE TENSORFLOW WARNINGS (MUST BE AT THE VERY TOP)
# ==============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely hides the GPU!
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Silences the oneDNN rounding warning

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# ==============================================================================
# 2. THE ULTIMATE FIX FOR THE "QUANTIZATION_CONFIG" BUG
# ==============================================================================
# We create "Safe" versions of the layers that intercept and delete the 
# corrupt 'quantization_config' argument before TensorFlow crashes.

class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) # 💥 NUKE THE CORRUPT ARGUMENT
        super().__init__(*args, **kwargs)

class SafeBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class SafeDropout(tf.keras.layers.Dropout):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# ==============================================================================
# 3. CONSTANTS & CONFIGURATION
# ==============================================================================
MODEL_PATH = "efficientnetv2s_tobacco3482.h5"
IMG_SIZE = (384, 384)
CLASS_NAMES = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

# ==============================================================================
# 4. TKINTER DESKTOP APPLICATION
# ==============================================================================
class DocumentClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tobacco-3482 AI Classifier (CPU Mode)")
        self.root.geometry("600x750")
        self.root.configure(bg="#f4f4f9")
        self.root.resizable(False, False)

        self.model = None
        self.current_image_path = None
        self.display_image = None

        self.setup_ui()
        self.load_ai_model()

    def load_ai_model(self):
        """Loads the TensorFlow model while applying our Safe Layer fix."""
        self.result_label.config(text="Bypassing Keras Bug & Loading Model...", fg="#d9534f")
        self.root.update()
        
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Could not find '{MODEL_PATH}' in the same folder as this script.")
            
            # 🛡️ THE MAGIC: We tell TensorFlow to use our Safe layers when it reads the .h5 file
            custom_objects = {
                'Dense': SafeDense,
                'BatchNormalization': SafeBatchNormalization,
                'Dropout': SafeDropout
            }
            
            with tf.keras.utils.custom_object_scope(custom_objects):
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            self.result_label.config(text="Model Loaded Successfully! Ready for input.", fg="#5cb85c")
        
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
            self.result_label.config(text="Critical Error loading model.", fg="#d9534f")

    def setup_ui(self):
        """Builds the modern Tkinter Interface."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=10, background="#ecf0f1")
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=20)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="Document Image Classifier", 
                               font=("Segoe UI", 18, "bold"), bg="#2c3e50", fg="white")
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, text="Powered by EfficientNetV2-S (Local CPU)", 
                                  font=("Segoe UI", 10), bg="#2c3e50", fg="#bdc3c7")
        subtitle_label.pack()

        # Image Canvas
        self.canvas_frame = tk.Frame(self.root, bg="#f4f4f9", pady=20)
        self.canvas_frame.pack()
        
        self.canvas = tk.Canvas(self.canvas_frame, width=384, height=384, 
                                bg="#e0e0e0", highlightthickness=2, highlightbackground="#bdc3c7")
        self.canvas.pack()
        self.placeholder_text = self.canvas.create_text(192, 192, text="No Image Loaded", 
                                                        font=("Segoe UI", 14, "bold"), fill="#7f8c8d")

        # Results Area
        self.result_label = tk.Label(self.root, text="", font=("Segoe UI", 14, "bold"), 
                                     bg="#f4f4f9", fg="#333333", pady=10)
        self.result_label.pack()
        
        self.confidence_label = tk.Label(self.root, text="", font=("Segoe UI", 12), 
                                         bg="#f4f4f9", fg="#7f8c8d")
        self.confidence_label.pack()

        # Control Buttons
        button_frame = tk.Frame(self.root, bg="#f4f4f9", pady=20)
        button_frame.pack()

        self.btn_load = ttk.Button(button_frame, text="📂 Load Image", command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=10)

        self.btn_predict = ttk.Button(button_frame, text="🔍 Predict", command=self.predict, state=tk.DISABLED)
        self.btn_predict.grid(row=0, column=1, padx=10)

        self.btn_reset = ttk.Button(button_frame, text="🗑️ Reset", command=self.reset)
        self.btn_reset.grid(row=0, column=2, padx=10)

    def load_image(self):
        """Handles opening the file explorer and previewing the image."""
        file_path = filedialog.askopenfilename(
            title="Select a Document Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if not file_path:
            return

        self.current_image_path = file_path
        
        # Visually resize for the UI canvas (keeps aspect ratio)
        img = Image.open(file_path).convert("RGB")
        img.thumbnail(IMG_SIZE, Image.Resampling.LANCZOS)
        
        display_bg = Image.new('RGB', IMG_SIZE, (224, 224, 224))
        offset = ((IMG_SIZE[0] - img.width) // 2, (IMG_SIZE[1] - img.height) // 2)
        display_bg.paste(img, offset)

        self.display_image = ImageTk.PhotoImage(display_bg)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        
        self.result_label.config(text="Image loaded. Ready to predict.", fg="#333333")
        self.confidence_label.config(text="")
        self.btn_predict.config(state=tk.NORMAL)

    def predict(self):
        """Passes the image through the AI model and outputs the prediction."""
        if not self.model or not self.current_image_path:
            return

        self.result_label.config(text="Analyzing on CPU... Please wait.", fg="#f39c12")
        self.root.update()

        try:
            # 1. Load image natively through TensorFlow (exactly as done in training)
            img = tf.keras.utils.load_img(self.current_image_path, target_size=IMG_SIZE)
            
            # 2. Convert to array and cast to float32
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.cast(img_array, tf.float32)
            
            # 3. Add batch dimension -> (1, 384, 384, 3)
            img_batch = tf.expand_dims(img_array, 0)
            
            # 4. Predict!
            predictions = self.model.predict(img_batch, verbose=0)
            
            # 5. Extract class and confidence
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = CLASS_NAMES[class_idx]

            # Update UI
            self.result_label.config(text=f"Prediction: {predicted_class}", fg="#2980b9")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during inference:\n{str(e)}")
            self.result_label.config(text="Prediction failed.", fg="#d9534f")

    def reset(self):
        """Clears the UI so the user can test another image."""
        self.current_image_path = None
        self.display_image = None
        self.canvas.delete("all")
        self.placeholder_text = self.canvas.create_text(192, 192, text="No Image Loaded", 
                                                        font=("Segoe UI", 14, "bold"), fill="#7f8c8d")
        self.result_label.config(text="Interface Reset.", fg="#333333")
        self.confidence_label.config(text="")
        self.btn_predict.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentClassifierApp(root)
    root.mainloop()