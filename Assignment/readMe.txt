# Skin Cancer Classification using HAM10000 Dataset

## Project Overview
This project implements a deep learning solution for skin cancer classification using the HAM10000 ("Human Against Machine with 10000 training images") dataset. The model uses transfer learning with EfficientNetB0 to classify skin lesions into 7 different categories or perform binary classification (benign vs malignant).

## Dataset Information
- Dataset: HAM10000 from Kaggle
- Total Images: 10,015 dermatoscopic images
- Classes: 7 different types of skin lesions
- Binary Classification: 
  - Benign: 8,061 images
  - Malignant: 1,954 images

## Requirements

### Python Libraries
- TensorFlow 2.19.0
- Keras 3.10.0
- Matplotlib 3.10.0
- Scikit-learn 1.6.1
- Pandas
- NumPy

### Hardware
- GPU: NVIDIA T4 (recommended)
- Storage: ~5.2GB for dataset

## Project Structure

### File Organization

/content/
├── HAM10000_metadata.csv # Dataset metadata
├── HAM10000_images_part_1/ # Image folder 1
├── HAM10000_images_part_2/ # Image folder 2
├── ham10000_binary/ # Binary classification dataset
│ ├── benign/ # 8,061 benign images
│ └── malignant/ # 1,954 malignant images
└── hmnist_*.csv # Preprocessed image data


### Key Features
1. Data Preprocessing
   - Automatic dataset download from Kaggle
   - Image organization into binary classes
   - Data augmentation (rotation, zoom, flip)
   - Train-validation split (80-20)

2. Model Architecture
   - Base model: EfficientNetB0 (pre-trained on ImageNet)
   - Custom classification head
   - Global Average Pooling
   - Dropout layers for regularization
   - Dense layers with ReLU activation
   - Final softmax layer for multi-class classification

3. Training Configuration
   - Input size: 224x224x3
   - Batch size: 32
   - Optimizer: Adam (learning rate: 1e-3)
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy

## Setup Instructions

### 1. Install Dependencies

!pip install tensorflow keras matplotlib scikit-learn

### 2. Kaggle Setup

Upload kaggle.json for API authentication

Set up Kaggle configuration in /root/.kaggle/

### 3. Download Dataset

!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p /content --unzip

### 4. Data Preparation
The notebook automatically:

Loads metadata from HAM10000_metadata.csv

Maps images to binary classes (benign/malignant)

Creates organized directory structure

Sets up data generators with augmentation

### 5. Model Training
The model is built and compiled with:

Frozen base EfficientNetB0 layers

Custom classification head

Appropriate loss function and optimizer

Model Architecture Summary
Base Model: EfficientNetB0 (4,049,564 parameters)

Custom Layers:

Global Average Pooling

Dropout (0.4)

Dense (128 units, ReLU)

Dropout (0.4)

Dense (7 units, Softmax) for multi-class

Usage
For Binary Classification

# Use the binary dataset in /content/ham10000_binary/
train_gen = datagen.flow_from_directory(
    '/content/ham10000_binary',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

2. For Multi-class Classification

# Use the original 7-class dataset
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


Performance
Training samples: 8,013 images

Validation samples: 2,002 images

Classes: 7 skin lesion types

Model ready for training with pre-trained weights

Notes
The dataset is imbalanced (more benign than malignant cases)

Data augmentation helps with generalization

Transfer learning with EfficientNet provides good feature extraction

Model can be fine-tuned by unfreezing base layers after initial training
