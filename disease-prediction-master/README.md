# Disease Prediction using Machine Learning
### Predicting Heart Disease & Diabetes with Python

---

## What is this project? (For the Teacher)

This is a **supervised machine learning project** that predicts whether a patient is at risk of **Heart Disease** or **Diabetes** based on their medical data such as age, blood pressure, cholesterol, glucose levels, and more.

### The Core Idea

We use **historical patient data** (already labeled as diseased / not diseased) to train machine learning models. These models learn patterns from the data — for example, *"patients with high cholesterol, elevated blood pressure, and low maximum heart rate tend to have heart disease"*. Once trained, the model can predict the outcome for a **new, unseen patient** based on their medical values.

This is called **binary classification** — the output is either 0 (no disease) or 1 (disease).

### Why is this useful?

Early and accurate disease prediction can:
- Assist doctors in identifying high-risk patients faster
- Reduce the cost of extensive medical testing
- Enable preventive care before a disease progresses

---

## Datasets Used

| Dataset | Source | Size | Target |
|---|---|---|---|
| Heart Disease | [Kaggle — Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | 918 rows, 11 features | HeartDisease (0 or 1) |
| Diabetes | [Kaggle — Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 rows, 8 features | Outcome (0 or 1) |

### Heart Disease Features
| Feature | Description |
|---|---|
| Age | Age of the patient in years |
| Sex | 1 = Male, 0 = Female |
| ChestPainType | Type of chest pain (0–3) |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1 = true) |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (1 = yes) |
| Oldpeak | ST depression induced by exercise |

### Diabetes Features
| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| BMI | Body mass index |
| Insulin | 2-hour serum insulin |
| DiabetesPedigreeFunction | Diabetes hereditary score |
| Age | Age of the patient |

---

## Machine Learning Pipeline

The project follows a standard ML pipeline:

```
Raw Data
   ↓
Step 1 — Load & Inspect Data        (understand shape, types, class balance)
   ↓
Step 2 — Exploratory Data Analysis  (visualise distributions, correlations)
   ↓
Step 3 — Preprocessing              (fix missing values, encode, scale)
   ↓
Step 4 — Train Models               (fit 5 different ML algorithms)
   ↓
Step 5 — Evaluate & Compare         (confusion matrix, ROC curve, F1-score)
   ↓
Step 6 — Predict for New Patient    (input values → get risk prediction)
```

---

## Models Trained & Compared

We train **5 different machine learning models** and compare them to find the best one:

| Model | How It Works | Strength |
|---|---|---|
| Logistic Regression | Fits a mathematical boundary between classes | Simple, interpretable, good baseline |
| Random Forest | Builds 100 decision trees, takes majority vote | High accuracy, handles noise well |
| XGBoost | Builds trees sequentially, each fixing previous errors | Usually the best performer |
| SVM | Finds the widest margin boundary between classes | Works well on small/medium datasets |
| KNN | Finds K most similar past patients, copies their label | Simple, intuitive distance-based method |

The best model is automatically selected based on **F1-Score** and saved as a `.pkl` file.

---

## How We Evaluate (Metrics)

In medical prediction, **accuracy alone is not enough**. A model that always predicts "no disease" would be 90% accurate but completely useless. So we use:

| Metric | What it means |
|---|---|
| Precision | Of all patients predicted as diseased, how many actually are? |
| Recall | Of all actually diseased patients, how many did we correctly catch? *(most critical in medical)* |
| F1-Score | Balance between precision and recall — our primary metric |
| ROC-AUC | How well the model separates the two classes across all thresholds |
| Confusion Matrix | Visual breakdown of true positives, false positives, true negatives, false negatives |

### Results Achieved

| Dataset | Best Model | Accuracy | F1-Score |
|---|---|---|---|
| Heart Disease | KNN / XGBoost | ~89% | ~0.89 |
| Diabetes | XGBoost / Random Forest | ~78% | ~0.76 |

---

## Project Structure

```
disease_prediction/
│
├── data/                        ← Place downloaded CSV datasets here
│   ├── heart.csv
│   └── diabetes.csv
│
├── models/                      ← Saved trained models (auto-generated on run)
│   ├── heart_disease_best_model.pkl
│   ├── diabetes_best_model.pkl
│   └── ... (scalers, feature names, all 5 models per dataset)
│
├── outputs/                     ← All plots saved here (auto-generated on run)
│   ├── class_distribution.png
│   ├── heart_correlation.png
│   ├── heart_disease_evaluation.png
│   ├── diabetes_evaluation.png
│   └── ... (boxplots, feature importance, model comparison charts)
│
├── notebooks/
│   └── disease_prediction.ipynb ← MAIN FILE — open and run this!
│
├── requirements.txt             ← All Python libraries needed
└── README.md                    ← This file
```

---

## For Teammates — How to Set Up & Run

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed — download from [python.org](https://www.python.org/downloads/)
- Git installed — download from [git-scm.com](https://git-scm.com/)
- A free Kaggle account — sign up at [kaggle.com](https://www.kaggle.com)
- VS Code with the **Jupyter** and **Python** extensions installed

---

### Step 1 — Clone the repository
```bash
git clone https://github.com/Aditya240602/disease-prediction.git
cd disease-prediction
```

---

### Step 2 — Create a virtual environment

**Windows:**
```bash
python -m venv disease_env
disease_env\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv disease_env
source disease_env/bin/activate
```

You should see `(disease_env)` appear at the start of your terminal line.

---

### Step 3 — Install all required libraries
```bash
pip install -r requirements.txt
```
This installs pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter, and more.

---

### Step 4 — Download the datasets from Kaggle

> Note: datasets are NOT included in the repo due to Kaggle's terms of use. You must download them yourself (free account required).

**Heart Disease dataset:**
1. Go to https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
2. Click Download
3. Extract the zip and copy `heart.csv` into the `data/` folder

**Diabetes dataset:**
1. Go to https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Click Download
3. Extract the zip and copy `diabetes.csv` into the `data/` folder

---

### Step 5 — Register the kernel in VS Code
```bash
python -m ipykernel install --user --name=disease_env --display-name "Python (disease_env)"
```

---

### Step 6 — Open and run the notebook
1. Open VS Code → Open Folder → select `disease-prediction`
2. Navigate to `notebooks/disease_prediction.ipynb`
3. Click the kernel name in the top right → select **"Python (disease_env)"**
4. Click **Restart** → then **Run All**
5. Wait ~2 minutes for all cells to finish

---

### Step 7 — View the results
- All charts and plots → `outputs/` folder
- Saved trained models → `models/` folder
- Inline outputs (tables, charts) visible directly in the notebook

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Wrong kernel — select "Python (disease_env)" in VS Code top right |
| `FileNotFoundError: ../data/heart.csv` | Download datasets from Kaggle and place in `data/` folder |
| `pip install` fails with version errors | Make sure virtual environment is activated — check for `(disease_env)` in terminal |
| Kernel not showing in VS Code | Run: `python -m ipykernel install --user --name=disease_env --display-name "Python (disease_env)"` |
| `git push` fails | Use a Personal Access Token instead of password — generate at github.com → Settings → Developer Settings |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `xgboost` | Gradient boosting model |
| `matplotlib` | Plotting charts and graphs |
| `seaborn` | Statistical visualisations |
| `joblib` | Saving and loading trained models |
| `jupyter` | Interactive notebook environment |

---

## Author

**Aditya** — Semester 4, ML Lab Project

Dataset sources: [Kaggle](https://www.kaggle.com) & [UCI Machine Learning Repository](https://archive.ics.uci.edu)
