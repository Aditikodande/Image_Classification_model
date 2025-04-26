COMPANY:CODETECH IT SOLUTIONS NAME:Aditi Kodande INTERN ID::CT04DA652 DOMAIN:Machine learning BATCH DURATION:April 10th, 2025 to May 10th, 2025. MENTOR NAME:NEELA SANTOSH
---

# Image Classification using CNN on CIFAR-10 🖼️🤖

This project builds a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 categories.

---

## 📚 Project Overview

Steps covered:

- **Install and Import Required Libraries**
- **Load and Preprocess the CIFAR-10 Dataset**
- **Build a CNN Model**
- **Compile and Train the Model**
- **Evaluate the Model**
- **Visualize Training/Validation Accuracy and Loss**
- **Make Predictions on Test Data**

---

## 📦 Requirements

Install the necessary Python libraries:

```bash
pip install tensorflow matplotlib numpy
```

---

## 🚀 How to Run

1. **Clone or download** the project files.
2. **Install the required libraries** using pip.
3. **Run the script** to:
   - Load and preprocess the CIFAR-10 dataset
   - Build and train the CNN
   - Evaluate its performance
   - Visualize the training process
   - Predict a sample output

---

## 🛠 Project Details

### Dataset:
- **CIFAR-10:**  
  60,000 32x32 color images in 10 classes, with 6,000 images per class.
  Classes include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

### Model Architecture:
- **Conv2D + ReLU** activation
- **MaxPooling2D** layers
- **Flatten** layer
- **Dense** layers
- **Softmax** output for multi-class classification

### Compilation:
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metric:** Accuracy

### Training:
- **Epochs:** 10
- **Validation:** Performed on test set during training

---

## 📈 Results

- **Test Accuracy** is printed after evaluation.
- **Accuracy and Loss graphs** are plotted for both training and validation datasets.

Example output:
- Test Accuracy: ~70%–80% after 10 epochs (depending on training randomness)

---

## 📊 Visualizations

- **Training vs Validation Accuracy Plot**
- **Training vs Validation Loss Plot**

These plots help in identifying overfitting or underfitting during model training.

---

## 🔥 Model Prediction Example

- After training, the model makes a prediction on a sample test image.
- Predicted label is printed.

---

## 🧠 Notes

- **Normalization:**  
  Pixel values are scaled between 0 and 1 to improve training performance.
  
- **Softmax Output:**  
  Final layer uses `softmax` activation to predict the probability distribution over 10 classes.

- **Single Prediction:**  
  Model predicts the class label for one test image using `np.argmax()` on model output.

---

## 💬 Acknowledgements

- TensorFlow for providing powerful deep learning tools.
- CIFAR-10 dataset from Canadian Institute for Advanced Research.

---

