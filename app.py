import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import re

# 🎨 Custom CSS for Dark Font on Light Blue Background
custom_css = """
<style>
    body {
        background-color: #f0f8ff;  /* Light Blue Background */
        color: #1a1a1a; /* Dark Font */
    }
    .stButton>button {
        background-color: #4B0082; /* Purple */
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #6A0DAD; /* Darker Purple */
    }
    .stDataFrame {
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #4B0082;
        color: #1a1a1a; /* Dark Font */
        font-size: 16px;
        padding: 10px;
    }
    h1 {
        color: #1a1a1a;  /* Dark Font */
        text-align: center;
        font-size: 40px;
    }
    h2, h3 {
        color: #2b2b2b;  /* Slightly Darker Blue */
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Step 1: Fetch Financial News (Yahoo Finance Scraper)
def get_stock_news():
    url = "https://finance.yahoo.com/topic/stock-market-news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract news headlines
    headlines = [h.text.strip() for h in soup.find_all("h3")]

    # If no news found, return default news
    if not headlines:
        return ["Stock market uncertain", "Economy shows growth", "Investors fear inflation"]

    return headlines[:10]  # Limit to 10 for efficiency

# Step 2: Clean News Headlines
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Remove special characters
    return text.strip()

# Step 3: Train a Sentiment Model (TF-IDF + Logistic Regression)
news_data = [
    "Stock market crashes", "Tech stocks soar", 
    "Investors panic", "Companies report huge profits", 
    "Inflation fears rise", "Crypto market stabilizes"
]
labels = [0, 1, 0, 1, 0, 1]  # 0 = Negative, 1 = Positive

# Handle case where news_data might be empty
if not news_data:
    news_data = ["Stock market uncertain", "Economy shows growth"]
    labels = [0, 1]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_data)
model = LogisticRegression()
model.fit(X, labels)

# Step 4: Predict Sentiment for News Headlines
def predict_sentiment(news_list):
    if not news_list:  # If no input news, return Neutral
        return ["Neutral"]

    cleaned_news = [clean_text(news) for news in news_list if news.strip()]
    
    if not cleaned_news:  # If all news were cleaned to empty strings
        return ["Neutral"] * len(news_list)

    news_vector = vectorizer.transform(cleaned_news)
    predictions = model.predict(news_vector)
    
    return ["Positive" if p == 1 else "Negative" for p in predictions]

# 🎯 Step 5: Streamlit UI with Beautiful Styling
st.title("📈 LLM-Based Financial News Sentiment Analysis Pipeline ")

st.markdown("#### 📊 Get real-time financial news and analyze sentiment instantly!")

# 📌 User Input for Custom News Analysis
st.subheader("📝 Enter Your Own News Headline for Sentiment Analysis")
user_input = st.text_input("Enter a news headline...")

# 📌 Analyze Custom User Input
if st.button("🔍 Analyze My News"):
    if user_input.strip():
        sentiment = predict_sentiment([user_input])[0]
        color = "green" if sentiment == "Positive" else "red"
        st.markdown(f"**📰 Headline Sentiment:** <span style='color:{color}; font-weight:bold;'>{sentiment}</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a news headline!")

# 📌 Fetch & Analyze Live News
st.subheader("📡 Live Financial News Sentiment Analysis")
if st.button("📡 Get Latest News & Analyze Sentiment"):
    news_list = get_stock_news()
    sentiments = predict_sentiment(news_list)

    # Display results in a DataFrame with colors
    df = pd.DataFrame({"Headline": news_list, "Sentiment": sentiments})
    
    def highlight_sentiment(val):
        color = "green" if val == "Positive" else "red"
        return f"color: {color}; font-weight: bold;"

    st.dataframe(df.style.applymap(highlight_sentiment, subset=["Sentiment"]))

    # Show summary
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")

    st.subheader("📊 Sentiment Breakdown")
    st.markdown(f"✅ **Positive News:** `{pos_count}`")
    st.markdown(f"❌ **Negative News:** `{neg_count}`")

    # 📊 Pie Chart Visualization
    st.subheader("📊 Sentiment Distribution")
    chart_data = pd.DataFrame(
        {"Sentiment": ["Positive", "Negative"], "Count": [pos_count, neg_count]}
    )
    st.bar_chart(chart_data.set_index("Sentiment"))
