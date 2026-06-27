import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .positive { color: #2ecc71; font-weight: bold; font-size: 1.2rem; }
    .negative { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
    .neutral  { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models & Data ────────────────────────────────────
@st.cache_resource
def load_models():
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return lr_model, tfidf

@st.cache_data
def load_data():
    df = pd.read_csv('processed_reviews.csv')
    bert_df = pd.read_csv('bert_results.csv')
    return df, bert_df

lr_model, tfidf = load_models()
df, bert_df = load_data()
sia = SentimentIntensityAnalyzer()

# ─── Text Cleaning ─────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/sentiment-analysis.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📊 Deep Dive",
    "🤖 Live Predictor",
    "📈 Model Comparison"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Amazon Fine Food Reviews")
st.sidebar.markdown("**Records:** 50,000 reviews")
st.sidebar.markdown("**Models:** VADER, LR, RF, BERT")

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">💬 Customer Sentiment Analysis</p>',
                unsafe_allow_html=True)
    st.markdown("### Amazon Fine Food Reviews — 50,000 Reviews Analyzed")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    pos = len(df[df['Sentiment'] == 'Positive'])
    neg = len(df[df['Sentiment'] == 'Negative'])
    neu = len(df[df['Sentiment'] == 'Neutral'])

    col1.metric("Total Reviews", f"{total:,}")
    col2.metric("Positive 😊", f"{pos:,}", f"{pos/total*100:.1f}%")
    col3.metric("Negative 😞", f"{neg:,}", f"{neg/total*100:.1f}%")
    col4.metric("Neutral 😐", f"{neu:,}", f"{neu/total*100:.1f}%")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df['Sentiment'].value_counts().reset_index(),
            x='Sentiment', y='count',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#f39c12'
            },
            title='Sentiment Distribution',
            text='count'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        score_counts = df['Score'].value_counts().sort_index()
        fig2 = px.pie(
            values=score_counts.values,
            names=[f'{i} Star' for i in score_counts.index],
            title='Star Rating Distribution',
            color_discrete_sequence=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#27ae60'],
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Reviews over time
    import datetime
    if 'year' not in df.columns:
        df['year'] = df['Time'].apply(
            lambda x: datetime.datetime.fromtimestamp(x).year)

    yearly = df.groupby(['year', 'Sentiment']).size().reset_index(name='count')
    fig3 = px.line(
        yearly, x='year', y='count',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        },
        title='Review Volume Over Time',
        markers=True
    )
    fig3.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════
elif page == "📊 Deep Dive":
    st.title("📊 Deep Dive Analysis")
    st.markdown("---")

    # WordClouds
    st.subheader("WordClouds by Sentiment")
    col1, col2, col3 = st.columns(3)
    sentiments = ['Positive', 'Negative', 'Neutral']
    cols = [col1, col2, col3]
    cmaps = ['Greens', 'Reds', 'Oranges']

    for col, sentiment, cmap in zip(cols, sentiments, cmaps):
        text = ' '.join(df[df['Sentiment'] == sentiment]['clean_text'].dropna())
        if text.strip():
            wc = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap=cmap,
                max_words=80
            ).generate(text)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(sentiment, fontweight='bold')
            col.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Top keywords
    st.subheader("Top 15 Keywords per Sentiment")
    col1, col2, col3 = st.columns(3)
    colors_bar = ['#2ecc71', '#e74c3c', '#f39c12']

    for col, sentiment, color in zip(
            [col1, col2, col3], sentiments, colors_bar):
        words = ' '.join(
            df[df['Sentiment'] == sentiment]['clean_text'].dropna()
        ).split()
        top_words = Counter(words).most_common(15)
        if top_words:
            words_list, counts = zip(*top_words)
            fig = px.bar(
                x=list(counts)[::-1],
                y=list(words_list)[::-1],
                orientation='h',
                title=f'{sentiment} Keywords',
                color_discrete_sequence=[color]
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='white',
                height=400
            )
            col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Review length
    st.subheader("Review Length Distribution")
    if 'review_length' not in df.columns:
        df['review_length'] = df['Text'].apply(
            lambda x: len(str(x).split()))

    fig = px.histogram(
        df, x='review_length',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        },
        nbins=50,
        title='Review Length by Sentiment',
        range_x=[0, 300],
        barmode='overlay',
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🤖 Live Predictor":
    st.title("🤖 Live Sentiment Predictor")
    st.markdown("Type any review — get instant sentiment prediction!")
    st.markdown("---")

    user_input = st.text_area(
        "Enter your review here:",
        placeholder="e.g. This product is absolutely amazing! Great quality...",
        height=150
    )

    col1, col2 = st.columns(2)
    model_choice = col1.selectbox(
        "Choose Model:",
        ["Logistic Regression (85.1%)", "VADER (79.4%)"]
    )

    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            cleaned = clean_text(user_input)

            if "Logistic" in model_choice:
                vec = tfidf.transform([cleaned])
                pred = lr_model.predict(vec)[0]
                proba = lr_model.predict_proba(vec)[0]
                classes = lr_model.classes_
                confidence = max(proba) * 100
            else:
                score = sia.polarity_scores(user_input)['compound']
                if score >= 0.05:
                    pred = 'Positive'
                elif score <= -0.05:
                    pred = 'Negative'
                else:
                    pred = 'Neutral'
                confidence = abs(score) * 100
                classes = ['Negative', 'Neutral', 'Positive']
                proba = [0, 0, 0]

            # Result display
            emoji = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐'}
            color_class = {
                'Positive': 'positive',
                'Negative': 'negative',
                'Neutral': 'neutral'
            }

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", f"{emoji[pred]} {pred}")
            col2.metric("Confidence", f"{confidence:.1f}%")
            col3.metric("Model Used", model_choice.split("(")[0].strip())

            # Probability bar chart
            if "Logistic" in model_choice:
                fig = px.bar(
                    x=list(classes),
                    y=[p * 100 for p in proba],
                    color=list(classes),
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Negative': '#e74c3c',
                        'Neutral': '#f39c12'
                    },
                    title='Prediction Confidence per Class',
                    labels={'x': 'Sentiment', 'y': 'Confidence (%)'},
                    text=[f'{p*100:.1f}%' for p in proba]
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='white',
                    yaxis_range=[0, 110]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a review first!")

# ══════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Performance Comparison")
    st.markdown("---")

    # Accuracy comparison
    models = ['VADER', 'Random Forest', 'BERT (RoBERTa)', 'Logistic Regression']
    accuracies = [79.4, 81.0, 80.9, 85.1]
    colors = ['#f39c12', '#2ecc71', '#9b59b6', '#3498db']

    fig = px.bar(
        x=models, y=accuracies,
        color=models,
        color_discrete_sequence=colors,
        title='Model Accuracy Comparison',
        labels={'x': 'Model', 'y': 'Accuracy (%)'},
        text=[f'{a}%' for a in accuracies]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insights table
    st.markdown("---")
    st.subheader("Model Insights")

    insights = pd.DataFrame({
        'Model': models,
        'Accuracy': ['79.4%', '81.0%', '80.9%', '85.1% 🏆'],
        'Speed': ['Fast', 'Medium', 'Slow', 'Fast'],
        'Neutral Detection': ['Poor', 'Poor', 'Fair', 'Good'],
        'Best For': [
            'Quick baseline',
            'Balanced classes',
            'Complex language',
            'Production use'
        ]
    })
    st.dataframe(insights, use_container_width=True, hide_index=True)

    # Key finding
    st.markdown("---")
    st.info("""
    **Key Finding:** Logistic Regression (85.1%) outperformed BERT (80.9%) on this dataset.
    This demonstrates that for domain-specific, imbalanced datasets, 
    traditional ML with proper feature engineering can outperform 
    complex transformer models — especially when training data is limited for BERT fine-tuning.
    """)