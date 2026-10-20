# Breast Cancer Diagnosis Classifier

A complete end-to-end machine learning project that classifies breast tumours as **malignant** or **benign** from 30 measurements of cell nuclei. Built as my **Final AI & ML Project** (Module 6), bringing together everything from the course: data analysis with Pandas/NumPy, visualisation with Matplotlib, model training with Scikit-learn, and an interactive Streamlit app.

> **Disclaimer:** This is an educational project. It is **not** a medical tool and must not be used for real diagnosis.

## Problem Statement
Given measurements computed from a digitised image of a fine needle aspirate of a breast mass (radius, texture, perimeter, area, concavity, etc.), predict whether the tumour is malignant or benign.

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, included in Scikit-learn (`load_breast_cancer`), so no download is needed
- **Samples:** 569 (212 malignant, 357 benign)
- **Features:** 30 numeric features, with no missing values

![Class distribution](images/class_distribution.png)

## Machine Learning Workflow
1. **Load & explore** the data (shape, missing values, class balance)
2. **Split** into 80% train / 20% test, stratified to keep class proportions
3. **Preprocess** with `StandardScaler`, so features on different scales are comparable
4. **Compare 4 models** using 5-fold cross-validation on the training set
5. **Select the best model** using cross-validation only, so the test set stays untouched
6. **Evaluate** the final model on unseen test data (accuracy, precision, recall, F1, ROC AUC, confusion matrix)
7. **Interpret** which features matter most
8. **Save** the trained model and build a demo app

## Results
| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|------------:|--------------:|----------:|-------:|---:|--------:|
| **Logistic Regression** | 0.9780 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| SVM (RBF) | 0.9692 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| Random Forest | 0.9626 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9932 |
| K-Nearest Neighbors | 0.9626 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9788 |

**Best model: Logistic Regression** (selected by cross-validation), with about **98% accuracy** on the held-out test set of 114 samples.

| Model comparison | Confusion matrix |
|:---:|:---:|
| ![Comparison](images/model_comparison.png) | ![Confusion matrix](images/confusion_matrix.png) |

![ROC curve](images/roc_curve.png)

### Key Insights
- Even a simple, interpretable model (Logistic Regression) performs very well on this dataset, so a more complex model is not always better.
- The most informative features relate to the **size and shape of the cell nuclei** (worst perimeter, worst area, concave points, radius).
- In medical settings, **recall** (not missing real malignant cases) matters as much as overall accuracy. This is why precision, recall and the confusion matrix are reported.

![Feature importance](images/feature_importance.png)

## Project Structure
```
breast-cancer-ml-classifier/
├── train.py             # Full pipeline: explore, compare, evaluate, save
├── predict.py           # Load saved model and predict
├── app.py               # Streamlit demo app
├── results.csv          # Model comparison table
├── model/
│   ├── best_model.joblib
│   └── metadata.json
├── images/              # Charts used in this README
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run
```bash
git clone https://github.com/tahoorfatima001/breast-cancer-ml-classifier.git
cd breast-cancer-ml-classifier
pip install -r requirements.txt

python train.py          # retrain, re-evaluate and regenerate charts
streamlit run app.py     # launch the interactive demo
```

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Scikit-learn · Joblib · Streamlit

## Limitations & Future Work
- Small dataset (569 samples). Results may not generalise to other populations.
- Hyperparameter tuning (GridSearchCV) could improve the models further.
- Could add model explainability (SHAP) and test on external data.
- Real medical use would need clinical validation, which is far beyond the scope of this project.

## Author
**Tahoor Fatima**: BS Economics and Data Science
GitHub: [tahoorfatima001](https://github.com/tahoorfatima001)
