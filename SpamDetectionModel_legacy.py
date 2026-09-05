import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data = data.dropna()
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['message'] = data['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'],
    test_size=0.2, random_state=42, stratify=data['label']
)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

strong_spam_keywords = ["lottery", "prize", "winner", "claim", "free", "cash", "scan and win", "win a chance","rs","$","reward","credited","won"]

def strong_keyword_detect(text):
    for word in strong_spam_keywords:
        if word in text:
            return True
    return False

while True:
    msg = input("\nEnter message (type exit to stop): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)

    if strong_keyword_detect(msg_clean):
        print("SPAM (Strong keyword detected)")
        continue

    msg_vec = vectorizer.transform([msg_clean])
    spam_prob = model.predict_proba(msg_vec)[0][1]

    if spam_prob > 0.35:
        print(f"SPAM (Confidence: {spam_prob*100:.2f}%)")
    else:
        print(f"NOT SPAM (Confidence: {(1-spam_prob)*100:.2f}%)")
