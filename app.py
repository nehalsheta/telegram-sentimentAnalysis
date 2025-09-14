import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression   #  LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# ------------------- Text Cleaning -------------------
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|@\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        return text
    else:
        return ""

def preprocess_text(text):
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    try:
        WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download('wordnet')

    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# ------------------- Load Data -------------------
df = pd.read_csv(r"telegram_channels_messages14021213_with_sentiment.csv")

df['cleaned_text'] = df['text'].apply(clean_text)
df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
df['processed_text'] = df['processed_text'].fillna('')
df['sentiment_type'] = df['sentiment_type'].fillna('neutral')

# ------------------- Vectorization -------------------
X = df['processed_text']
y = df['sentiment_type']

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# ------------------- Visualize class distribution before oversampling -------------------
st.subheader("📊 Class Distribution Before Oversampling")
fig1, ax1 = plt.subplots()
sns.countplot(x=y, ax=ax1)
ax1.set_title("Before Oversampling")
st.pyplot(fig1)

# ------------------- Apply Random Oversampling -------------------
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_tfidf, y)

# ------------------- Visualize class distribution after oversampling -------------------
st.subheader("📊 Class Distribution After Oversampling")
fig2, ax2 = plt.subplots()
sns.countplot(x=y_resampled, ax=ax2)
ax2.set_title("After Oversampling")
st.pyplot(fig2)

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ------------------- Train Model -------------------
model = LogisticRegression(max_iter=1000, random_state=42) 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ------------------- Streamlit App -------------------
st.title("📊 Sentiment Analysis App")
st.write("Enter text to analyze sentiment: Positive, Negative, or Neutral")
st.write(f"📈 Model Accuracy: **{acc*100:.2f}%**")

# ------------------- User Input -------------------
user_input = st.text_area("✍️ Enter your text here:")

if st.button("Analyze"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        processed = preprocess_text(cleaned)
        vec = tfidf_vectorizer.transform([processed])
        prediction = model.predict(vec)[0]

        st.subheader("✅ Result:")
        if prediction.lower() == "positive":
            st.success("🌟 Positive")
        elif prediction.lower() == "negative":
            st.error("🚨 Negative")
        else:
            st.info("😐 Neutral")
    else:
        st.warning("⚠️ Please enter some text first.")
