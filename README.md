#CIFAR-10 Image Classification using Transfer Learning

Comparative analysis of VGG16, ResNet50, and MobileNetV2 architectures for CIFAR-10 image classification using transfer learning with pre-trained ImageNet weights.

## 📋 Overview

This project implements and compares three state-of-the-art CNN architectures (VGG16, ResNet50, MobileNetV2) for image classification on the CIFAR-10 dataset. The models utilize transfer learning by freezing pre-trained ImageNet weights and training custom classifier heads.

**Key Results:**
- VGG16: **60.74%** test accuracy (best performer)
- ResNet50: 50.55% test accuracy
- MobileNetV2: 34.17% test accuracy (fastest training)

## 🎯 Features

- Transfer learning with frozen pre-trained weights
- Custom classifier head architecture
- Data augmentation (rotation, shifts, flips, zoom)
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Training visualization (accuracy/loss curves)
- Per-class performance analysis
- Confusion matrix generation
- Model checkpointing and early stopping

## 🛠️ Requirements
See [requirements.txt](requirements.txt) for dependencies.

```
tensorflow>=2.19.0
numpy>=2.0.2
matplotlib>=3.10.0
seaborn>=0.13.2
pandas>=2.2.2
scikit-learn>=1.6.1
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cifar10-transfer-learning.git
cd cifar10-transfer-learning

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

Run the complete pipeline in Google Colab or Jupyter Notebook:

```python
# The notebook will automatically:
# 1. Load and preprocess CIFAR-10 dataset
# 2. Train all three models
# 3. Generate visualizations
# 4. Save results and trained models
```

### Training Individual Models

```python
# Train VGG16
model = build_transfer_model('VGG16', input_shape=(32, 32, 3), num_classes=10)
history = model.fit(train_data, validation_data=val_data, epochs=30)

# Train ResNet50
model = build_transfer_model('ResNet50', input_shape=(32, 32, 3), num_classes=10)

# Train MobileNetV2
model = build_transfer_model('MobileNetV2', input_shape=(32, 32, 3), num_classes=10)
```

## 📊 Dataset

**CIFAR-10** consists of 60,000 32×32 color images in 10 classes:
- Training: 50,000 images (split to 40,000 train / 10,000 validation)
- Test: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 🏗️ Model Architecture

Each model follows the same structure:

```
Input (32×32×3)
    ↓
Pre-trained Base (frozen)
    ↓
Global Average Pooling
    ↓
Batch Normalization
    ↓
Dense(256, ReLU) + Dropout(0.5)
    ↓
Batch Normalization
    ↓
Dense(128, ReLU) + Dropout(0.3)
    ↓
Dense(10, Softmax)
```

**Base Models:**
- **VGG16**: 14.9M total params, 167K trainable
- **ResNet50**: 24.2M total params, 563K trainable
- **MobileNetV2**: 2.6M total params, 365K trainable

## 📈 Results

### Overall Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| VGG16 | **60.74%** | 0.606 | 0.607 | 0.602 | 15.4 min |
| ResNet50 | 50.55% | 0.503 | 0.506 | 0.492 | 16.0 min |
| MobileNetV2 | 34.17% | 0.351 | 0.342 | 0.338 | 11.4 min |

### Per-Class Accuracy (VGG16)

| Class | Accuracy |
|-------|----------|
| Ship | 72.1% |
| Automobile | 71.2% |
| Airplane | 69.1% |
| Frog | 68.3% |
| Truck | 67.2% |
| Horse | 66.1% |
| Deer | 54.4% |
| Dog | 52.1% |
| Bird | 47.8% |
| Cat | 44.1% |

## 📁 Project Structure

```
├── CV_DL_ASSGN4.ipynb          # Main notebook with all experiments
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── results/
│   ├── model_comparison.csv    # Numerical results
│   ├── detailed_results.json   # Complete metrics
│   ├── training_curves.png     # Training/validation curves
│   ├── confusion_matrices.png  # Confusion matrices
│   ├── metrics_comparison.png  # Bar chart comparisons
│   └── per_class_accuracy.png  # Per-class performance
└── models/
    ├── VGG16_best_model.h5
    ├── ResNet50_best_model.h5
    └── MobileNetV2_best_model.h5
```

## 🔧 Training Configuration

**Hyperparameters:**
- Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.98)
- Batch size: 64
- Epochs: 30 (with early stopping)
- Loss: Sparse categorical cross-entropy
- Callbacks: EarlyStopping (patience=5), ModelCheckpoint, ReduceLROnPlateau

**Data Augmentation:**
- Rotation: ±15°
- Width/height shift: 10%
- Horizontal flip: 50% probability
- Zoom: 10%

## 💡 Key Findings

1. **VGG16 outperforms complex architectures** - Simple uniform architecture works best for low-resolution images
2. **Transfer learning limitations** - Frozen pre-trained features struggle with 32×32 resolution (designed for 224×224)
3. **Vehicle classes easiest** - All models perform best on ship, automobile, truck
4. **Animal classes challenging** - Cat, dog, bird show lowest accuracy across all models
5. **Efficiency trade-off** - MobileNetV2 fastest but sacrifices 26% accuracy compared to VGG16

## 🔬 Future Work

- [ ] Fine-tune top layers instead of freezing all weights
- [ ] Upsample CIFAR-10 images to 224×224 for better transfer learning
- [ ] Implement ensemble methods combining multiple models
- [ ] Test advanced augmentation (MixUp, CutMix, AutoAugment)
- [ ] Compare with training from scratch
- [ ] Add attention mechanisms

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@misc{samantaray2025cifar10transfer,
  author = {Samantaray, Prabhupada},
  title = {Comparative Analysis of Pre-trained CNN Architectures for CIFAR-10},
  year = {2025},
  institution = {KIIT University},
  howpublished = {\url{https://github.com/prabhu-313/cifar10-transfer-learning}}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Prabhupada Samantaray**
- Institution: KIIT University, Bhubaneswar, Odisha, India
- Email: 22052483@kiit.ac.in

## 🙏 Acknowledgments

- KIIT University for computational resources
- TensorFlow/Keras team for excellent framework
- CIFAR-10 dataset creators

## 📚 References

1. Simonyan & Zisserman, "Very Deep Convolutional Networks" (VGG16)
2. He et al., "Deep Residual Learning for Image Recognition" (ResNet)
3. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
4. Krizhevsky & Hinton, "Learning Multiple Layers of Features from Tiny Images" (CIFAR-10)

---

⭐ If you find this project helpful, please consider giving it a star!

