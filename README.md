# 💬 Customer Sentiment Analysis Dashboard

> An end-to-end NLP sentiment analysis pipeline on 568K+ Amazon Fine Food Reviews using BERT, Logistic Regression, Random Forest, and VADER — deployed as an interactive Streamlit dashboard.

## 🚀 Live Demo
👉 [Click here to view the live dashboard](https://customersentimentanalysis-inzsbwycayhl8odurwdh5h.streamlit.app/)

## 📌 Project Overview
- Analyzed **568,454 Amazon Fine Food Reviews**
- Built and compared **4 NLP models**
- Deployed **interactive dashboard** with real-time sentiment predictor
- Achieved **99.2% confidence** on live review predictions

## 📊 Model Comparison
| Model | Accuracy | Notes |
|-------|----------|-------|
| VADER | 79.4% | Rule-based baseline |
| Random Forest | 81.0% | Poor on Neutral class |
| BERT (RoBERTa) | 80.9% | Limited by sample size |
| **Logistic Regression** | **85.1% 🏆** | Best overall performance |

> **Key Finding:** Logistic Regression outperformed BERT on this imbalanced domain-specific dataset — proving that traditional ML with proper feature engineering can beat complex transformers when fine-tuning data is limited.

## ✨ Dashboard Features
- **🏠 Overview** — KPI cards, sentiment distribution, star ratings, review trends over time
- **📊 Deep Dive** — WordClouds per sentiment, top 15 keywords, review length analysis
- **🤖 Live Predictor** — Type any review → get instant sentiment + confidence score
- **📈 Model Comparison** — Accuracy comparison, insights table, key findings

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python |
| NLP | NLTK, VADER, HuggingFace Transformers (RoBERTa) |
| ML | Scikit-learn, TF-IDF, Logistic Regression, Random Forest |
| Visualization | Plotly, Matplotlib, WordCloud |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |

## 📁 Dataset
- **Source:** [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Size:** 568,454 reviews
- **Fields:** ProductId, UserId, Score (1–5), Review Text, Summary, Helpfulness, Timestamp
- **Label Mapping:** 1–2 stars = Negative | 3 stars = Neutral | 4–5 stars = Positive

## 📂 Project Structure
