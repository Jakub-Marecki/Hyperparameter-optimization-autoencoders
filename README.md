# A Comparative Study of Hyperparameter Optimization Methods in Deep Learning  
## Autoencoder-Based Classification of Fish Pond Types

### Author  
Jakub Marecki  

---

## 1. Introduction

Hyperparameter optimization plays a critical role in the performance of deep learning models. The choice of optimization strategy affects not only predictive accuracy but also computational efficiency and convergence stability.

This repository presents a comparative study of selected hyperparameter optimization frameworks applied to neural network models in a controlled experimental setting. The analysis focuses on autoencoder-based classification and baseline MLP architectures evaluated on a synthetic dataset simulating environmental characteristics of fish ponds.

The primary objective is to assess differences in optimization efficiency, model performance, and practical usability of modern tuning frameworks.

---

## 2. Compared Optimization Methods

The following hyperparameter optimization approaches are evaluated:

- **Random Search**
- **Hyperband**
- **Optuna**
  - with pruning
  - without pruning
- **Ray Tune**
  - with ASHA scheduler
  - without ASHA scheduler

The comparison considers:
- validation accuracy
- convergence behavior
- computational characteristics
- stability across trials

---

## 3. Model Architectures

### 3.1 Multi-Layer Perceptron (MLP)

A fully connected feed-forward neural network trained directly on standardized input features.  
Serves as a baseline deep learning classifier.

### 3.2 Autoencoder + Classifier

An autoencoder is first trained to learn compressed feature representations.  
The encoder’s latent representation is then used as input for a downstream classification network.

This setup enables evaluation of representation learning combined with hyperparameter tuning.

---

## 4. Dataset Description

The dataset is synthetically generated within the notebooks to simulate environmental and biological characteristics of fish ponds.

Target variable:
- `typ_stawu ∈ {0, 1, 2}`

Example input features include:

- water temperature  
- pH level  
- dissolved oxygen  
- nitrates  
- phosphates  
- water transparency  
- pond depth  
- vegetation density  
- average fish weight  

All features are standardized using `StandardScaler`.

---

## 5. Experimental Design

- Train / validation / test split  
- Controlled random seeds for reproducibility  
- Consistent evaluation metric across frameworks  
- Accuracy used as the primary evaluation metric  

Each optimization framework searches over comparable hyperparameter spaces to ensure fairness of comparison.

---

## 6. Implementation Details

Frameworks and libraries used:

- TensorFlow / Keras  
- Scikit-learn  
- Keras Tuner  
- Optuna  
- Ray Tune  
- NumPy / Pandas  
- Matplotlib / Seaborn  

Experiments are implemented in:

- `MLP-klasyfikacja.ipynb`
- `Autoenkoder-klasyfikacja.ipynb`
