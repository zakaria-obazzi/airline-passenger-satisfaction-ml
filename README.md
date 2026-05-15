# Airline Passenger Satisfaction — ML Classification Project

> Predicting whether airline passengers are satisfied or not, using 6 machine learning models.

**Réalisé par :** Obazzi & Magaddi

---

## Project Overview

This project aims to classify airline passenger satisfaction using a real-world dataset from Kaggle.
We trained, tuned, and compared 6 classification models to find the best performer.

---

## Dataset

- **Source:** [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Size:** 129,880 passengers · 25 features
- **Target variable:** `satisfaction`
  - 56.5% → Neutral or Dissatisfied
  - 43.5% → Satisfied
- **Split:** 80% training / 20% testing (stratified)

### Features include:
- Passenger info: Gender, Age, Customer Type, Type of Travel, Class
- Flight info: Flight Distance, Departure/Arrival Delay
- Service ratings (1–5): Inflight wifi, Food & drink, Seat comfort, Online boarding, and more

---

##  Models & Results

| Model | Accuracy | Notes |
|---|---|---|
|  Random Forest | **95%** | Best model — GridSearchCV (depth=25, 200 trees) |
| KNN | **94%** | Best k=11, tuned with GridSearchCV |
| SVM | **94%** | RBF kernel, C=10 |
| Decision Tree | **90%** | criterion=entropy |
| Naive Bayes | **89%** | CategoricalNB variant |
| Logistic Regression | **87%** | max_iter=1000 |

###  Best Model — Random Forest
```
              precision    recall  f1-score
Not Satisfied    0.95      0.97      0.96
Satisfied        0.96      0.93      0.94
Accuracy                             0.95
```

---

## Methodology

1. **Data Cleaning** — Imputed missing values in `Arrival Delay in Minutes` with the mean
2. **Encoding** — OrdinalEncoder for `Class` and `satisfaction`; One-Hot Encoding for `Gender`, `Customer Type`, `Type of Travel`
3. **Scaling** — StandardScaler applied for KNN, SVM, and Logistic Regression
4. **Hyperparameter Tuning** — GridSearchCV with cross-validation (cv=4)
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
6. **Feature Importance** — Top 15 features visualized using Random Forest

---

## Project Structure

```
├── Predicting_airline_passenger_satisfaction.ipynb   # Main notebook (full pipeline)
├── train.csv              # Training data
├── test.csv               # Test data
└── README.md
```

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Launch the notebook:
```bash
jupyter notebook Predicting_airline_passenger_satisfaction.ipynb
```

##  License
This project is for educational purposes.
