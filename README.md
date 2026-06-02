# 🐱🐶 Cats vs. Dogs — CNN Image Classifier

A binary image-classification project built with **TensorFlow / Keras**. The model is a Convolutional Neural Network (CNN) trained to distinguish images of cats from dogs using the classic [Kaggle *Dogs vs. Cats*](https://www.kaggle.com/c/dogs-vs-cats) dataset.

This repo is a hands-on learning project: building a CNN from scratch, wiring up an image pipeline, training with data augmentation, and — importantly — **debugging a subtle data-pipeline bug** that produced misleadingly perfect accuracy. (More on that in [Lessons Learned](#-lessons-learned).)

---

## 🧠 Model Architecture

A compact CNN with two convolutional blocks followed by a fully connected classifier:

| Layer | Configuration | Output Shape | Purpose |
|-------|--------------|--------------|---------|
| Input | 64 × 64 × 3 | (64, 64, 3) | RGB image |
| Conv2D | 32 filters, 3×3, ReLU | (62, 62, 32) | Learn low-level features (edges, textures) |
| MaxPooling2D | 2×2 | (31, 31, 32) | Downsample, add translation invariance |
| Conv2D | 32 filters, 3×3, ReLU | (29, 29, 32) | Learn higher-level features |
| MaxPooling2D | 2×2 | (14, 14, 32) | Downsample |
| Flatten | — | (6272,) | Vectorize feature maps |
| Dense | 128 units, ReLU | (128,) | Combine features |
| Dropout | rate = 0.5 | (128,) | Regularization |
| Dense | 1 unit, Sigmoid | (1,) | Output probability (cat vs. dog) |

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy
- **Trainable params:** ~813K (the first Dense layer alone holds ~803K)

---

## 📁 Project Structure

```
.
├── archive/
│   ├── train/
│   │   ├── cats/        # cat.*.jpg
│   │   └── dogs/        # dog.*.jpg
│   └── test/
│       ├── cats/
│       └── dogs/
├── cnn_classifier.ipynb
├── log.csv               # per-epoch training log
└── README.md
```

> ⚠️ **Critical:** Keras `flow_from_directory()` infers labels from **subfolder names**, not filenames. The Kaggle dataset ships as flat folders (`cat.0.jpg`, `dog.0.jpg` all together), so you **must** sort images into `cats/` and `dogs/` subfolders before training. See [Lessons Learned](#-lessons-learned).

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow pandas numpy matplotlib livelossplot
```

> 💡 **GPU note (Windows):** TensorFlow ≥ 2.11 dropped native-Windows GPU support. For GPU acceleration use **WSL2** or the **TensorFlow-DirectML** plugin. CPU works fine for this small model.

---

## 🚀 Usage

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data: augment to reduce overfitting
train_gen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
# Test data: ONLY rescale — never augment evaluation data
test_gen = ImageDataGenerator(rescale=1./255)

training_set = train_gen.flow_from_directory(
    'archive/train', target_size=(64, 64), batch_size=32, class_mode='binary'
)
testing_set = test_gen.flow_from_directory(
    'archive/test', target_size=(64, 64), batch_size=32, class_mode='binary'
)

# Sanity check — should print {'cats': 0, 'dogs': 1}, NOT a single class
print(training_set.class_indices)

cnn_model.fit(training_set, epochs=25, validation_data=testing_set)
```

### Single-image prediction

```python
import numpy as np
from keras.preprocessing import image

img = image.load_img(r'archive/test/cats/cat.8428.jpg', target_size=(64, 64))
arr = np.expand_dims(image.img_to_array(img) / 255., axis=0)   # remember to rescale!
prob = cnn_model.predict(arr)[0][0]

label = 'dog' if prob > 0.5 else 'cat'   # threshold at 0.5, not == 1
print(f"{label} ({prob:.3f})")
```

---

## 🐛 Lessons Learned

This project's biggest takeaway came from a bug, not the model:

- **The "100% accuracy" trap.** An early run reported 100% train *and* validation accuracy in 2 epochs. `class_indices` revealed the cause — the generator saw only **one class** because images weren't sorted into `cats/`/`dogs/` subfolders. With a single label, predicting a constant scores 100%. **Lesson:** suspiciously perfect metrics usually mean a broken pipeline, not a brilliant model. Always check `class_indices` and class balance first.
- **Never augment your test set.** Augmentation (shear, zoom, flips) belongs only on training data. Evaluating on augmented data gives a distorted, non-reproducible picture of performance.
- **Sigmoid outputs need a threshold.** Compare `prob > 0.5`, not `prob == 1` — a sigmoid never returns exactly 1.
- **Rescale at inference too.** Single-image predictions must apply the same `1./255` rescaling used in training.

---

## 🔭 Next Steps

- [ ] Re-train on the correctly structured dataset and report honest accuracy (a model like this realistically lands around 75–85%)
- [ ] Add `EarlyStopping` and `ModelCheckpoint` callbacks
- [ ] Try transfer learning (e.g., MobileNetV2, VGG16) for a strong accuracy boost
- [ ] Plot a confusion matrix and a few misclassified examples
- [ ] Increase input resolution (e.g., 128×128) and compare

---

##The fix
- After retraining the data on a perfectely structured dataset and setting the epochs value to 5, I ended up with a realistic accuracy of about 80%
The maccuracy would increase till an epoch value of anywhere between 15-25. Anything about it may cause overfitting and ruin the accuracy

## 🛠️ Tech Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Pandas` · `Matplotlib` · `livelossplot`
