# 🧠 Nexus Predict – AI Disease Prediction System

A **full-stack AI-powered healthcare web app** that predicts **Heart Disease & Diabetes risk** using Machine Learning, enhanced with **advanced analytics, risk scoring, and AI-style explanations**.

---

## 🚀 Features

### 🔍 Prediction Engine

* Heart Disease Risk Prediction
* Diabetes Risk Prediction
* Built using **Scikit-learn (RandomForest models)**

### 📊 Advanced Analytics

* Risk Percentage (ML probability + scoring logic)
* Health Score (0–100)
* Interactive Doughnut Chart (Chart.js)
* Animated Progress Bar

### 🧠 AI Insights (LLM-style)

* Personalized explanation based on user inputs
* Explains *why* the prediction was made

### 🥗 Smart Recommendations

* Dynamic Diet Plan
* Personalized Exercise Suggestions
* Based on:

  * Blood Pressure
  * Cholesterol
  * Age
  * Risk Level

### 🎨 UI/UX

* Modern **glassmorphism + gradient UI**
* Smooth animations & transitions
* Skeleton loading (no stuck UI)
* Fully responsive layout
* 🌙 Dark Mode Toggle (with persistence)

---

## 🛠 Tech Stack

| Layer     | Technology            |
| --------- | --------------------- |
| Frontend  | HTML, CSS, JavaScript |
| Backend   | Flask (Python)        |
| ML Models | Scikit-learn          |
| Charts    | Chart.js              |

---

## ⚙️ Project Structure

```
projectml/
│
├── backend/
│   ├── app.py
│   ├── train_models.py
│   └── artifacts/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

---

### 2️⃣ Train Models

```bash
python3 backend/train_models.py
```

---

### 3️⃣ Run Backend (IMPORTANT ⚠️)

```bash
PORT=5001 python3 backend/app.py
```

👉 Backend runs on:

```
http://127.0.0.1:5001
```

---

### 4️⃣ Run Frontend

Open:

```
frontend/index.html
```

👉 Use **Live Server** (recommended)

---

## 🌐 API

### POST `/predict`

#### Example Request:

```json
{
  "model": "heart",
  "age": 45,
  "sex": "Male",
  ...
}
```

#### Example Response:

```json
{
  "ok": true,
  "model": "heart",
  "result": "Low Risk"
}
```

---

## 🧠 Core Concepts

* ML + Custom Risk Scoring Combined
* Feature-based health analysis
* AI-style explanation generation (rule-based simulation)
* Real-time UI feedback & analytics

---

## 🏆 Highlights

* End-to-end ML + Web integration
* Production-style UI/UX
* Smart health recommendations engine
* Fully interactive frontend dashboard
* Designed for **portfolio & interviews**

---

## ⚠️ Note

* This is a **prediction system for educational purposes only**
* Not a substitute for professional medical advice

---

## 👨‍💻 Author

**Adil Rasheed**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
