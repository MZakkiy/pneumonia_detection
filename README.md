# Pneumonia Detection from Chest X-Rays using Deep Learning

This repository contains a deep learning project to classify chest X-ray images as 'Normal' or 'Pneumonia'. The entire workflow, from data exploration to model evaluation, is documented in a single Jupyter Notebook. The model is built using PyTorch and leverages the ResNet50 architecture.

A pre-trained model file (`best_resnet50_model.pth`) is included in this repository for demonstration and inference purposes.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)

---

## Project Overview

The goal of this project is to develop a reliable and accurate deep learning model to assist medical professionals in diagnosing pneumonia. The key objective was to build a model with high sensitivity (Recall) to minimize false negatives, ensuring that potential pneumonia cases are not missed. This tool is intended to act as a diagnostic aid, helping to prioritize cases for review by a radiologist.

---

## Dataset

**Important:** The dataset is not included in this repository due to its large size.

The project uses the **Chest X-Ray Images (Pneumonia)** dataset, which is publicly available on Kaggle. You must download it from the following link:
- **Dataset Link:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## Methodology

The project workflow is detailed within the `pneumonia-detection.ipynb` file and covers:

1.  **Data Exploration:** Analyzing the dataset's characteristics, including class distribution and image properties, which revealed a significant class imbalance.
2.  **Data Preprocessing:** Creating a pipeline to standardize the data. This includes resizing images, converting them to RGB, and normalization. Data augmentation was used on the training set to improve model generalization.
3.  **Modeling:** Using a **ResNet50** model pre-trained on ImageNet and adapting it for our binary classification task via transfer learning.
4.  **Training:** Training the model with a `WeightedRandomSampler` to handle the class imbalance effectively.
5.  **Evaluation:** Evaluating the trained model on a held-out test set using metrics like accuracy, precision, recall, and a confusion matrix.

---

## Results

The model achieved excellent performance, particularly in identifying pneumonia cases correctly.

| Metric                  | Value   |
| ----------------------- | ------- |
| **Overall Accuracy** | 89.3%   |
| **Pneumonia Recall** | **93.6%** |
| **Pneumonia Precision** | 86.7%   |

The high recall (sensitivity) of 93.6% indicates that the model is highly effective at its primary goal of not missing potential pneumonia cases.

