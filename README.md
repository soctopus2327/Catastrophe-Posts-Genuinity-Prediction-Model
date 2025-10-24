# 🌪️ Catastrophe Post Genuinity Prediction

This project aims to classify whether a given social media post is related to a **catastrophe (disaster)** or not, using **Logistic Regression** and **TF-IDF vectorization** combined with textual feature analysis.

---

## 🧠 Project Overview

During crisis events, it's crucial to quickly identify and filter posts that indicate **emergency situations**.  
This model helps predict if a post is **catastrophic** (e.g., about floods, earthquakes, fires) or **non-catastrophic**.

It uses:
- **TF-IDF** to represent text data numerically.  
- **Statistical features** (like word count, punctuation, and character count).  
- **Logistic Regression** as the final predictive model.

---

## 📊 Dataset

The dataset contains social media posts and their binary labels:

| Column | Description |
|---------|--------------|
|`id` | unique identifier for each post|
| `text` | The content of the post |
|`keyword`| particular keyword from the post|
|`location`| location the post was sent from |
| `target` | 1 = related to disaster, 0 = not related |

---

## ⚙️ Features Extracted

1. **TF-IDF Features (Text-based)**  
   Represent textual information numerically based on word importance.

2. **Numeric Features (Statistical)**  
   - Character Count  
   - Word Count  
   - Punctuation Count  

These are standardized using `StandardScaler` before being combined with TF-IDF features.

---

## 🧩 Model Pipeline

| Step | Description |
|------|--------------|
| 1 | Data Cleaning (stopword removal, punctuation stripping, lowercasing) |
| 2 | TF-IDF Vectorization |
| 3 | Feature Engineering (char/word/punct counts) |
| 4 | Feature Scaling |
| 5 | Model Training (Logistic Regression) |
| 6 | Model Evaluation & Deployment |

---

## 🧪 Exploratory Data Analysis (EDA)

The EDA explores patterns in the dataset, such as:
- Distribution of target labels (catastrophe vs non-catastrophe)
- Common words in each class
- Text length distributions
- Correlations between numeric features and class labels

---

## 🧪 Model Selection

Accuracy and F1 Scores were compared across various models like
- Logistic Regression
- Naive Bayes (Multinomial)
- Random Forest
- XGBoost

Hyperparameters were further tuned for Logistic Regression and XGBoost.

Ensemble Soft Voting Models were also tried for
- Logistic Regression + Naive Bayes + Random Forest + XGBoost
- Logistic Regression + Naive Bayes
- Logistic Regression + XGBoost

But best Accuracy(0.82) and F1 score(0.77) was found for Logistic Regression Model with default hyperparameters.

---

🚀 [https://catastrophe-post-genuinity-predictor.streamlit.app/](Deployed Streamlit App)

