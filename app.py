import streamlit as st

from spam_model import dataset_statistics, predict_message, train_model

st.set_page_config(
    page_title="SpamShield | SMS Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; color: #172033; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stSidebar"] * { color: #e8edf6; }
    .brand { padding: 0.3rem 0 1.2rem; }
    .brand h1 { margin: 0; color: #fff; font-size: 1.55rem; }
    .brand p { color: #9eacc2; margin: .25rem 0 0; font-size: .82rem; }
    .hero { background: linear-gradient(120deg,#172b4d,#2463a8); color:white;
            padding:2rem 2.2rem; border-radius:18px; margin-bottom:1.4rem; }
    .hero h1 { margin:0; font-size:2.35rem; } .hero p { color:#dceaff; margin:.4rem 0 0; }
    .result { padding:1.35rem 1.5rem; border-radius:16px; border:1px solid; margin:1rem 0; }
    .result.spam { background:#fff1f0; border-color:#ffb5ad; color:#8b1e16; }
    .result.ham { background:#ecfbf3; border-color:#9bdcba; color:#12653b; }
    .result h2 { margin:0 0 .35rem; } .muted { color:#60708a; }
    .card { background:white; padding:1.15rem; border:1px solid #e3e8f0; border-radius:14px;
            min-height:92px; box-shadow:0 2px 8px rgba(20,38,70,.04); }
    .card small { color:#687892; display:block; } .card strong { font-size:1.35rem; }
    .footer { text-align:center; color:#72809a; padding:2.5rem 0 1rem; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training SpamShield on the SMS dataset…")
def get_bundle():
    return train_model()


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="card"><small>{label}</small><strong>{value}</strong></div>', unsafe_allow_html=True)


try:
    bundle = get_bundle()
except Exception as exc:
    st.error(f"SpamShield could not load its model: {exc}")
    st.info("Check that the supplied CSV is available as spam.csv or in attached_assets.")
    st.stop()

stats = dataset_statistics(bundle["data"])
with st.sidebar:
    st.markdown('<div class="brand"><h1>🛡️ SpamShield</h1><p>AI-Powered SMS Detection</p></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["🔍 Detect Message", "📊 Dashboard", "🤖 Model Information", "⚙️ How It Works", "ℹ️ About Project"])
    st.divider()
    st.caption(f"{stats['total']:,} messages • Hybrid detection")

if page == "🔍 Detect Message":
    st.markdown('<div class="hero"><h1>🛡️ SpamShield</h1><p>AI-Powered SMS Spam Detection System</p><p>Analyze SMS messages using a hybrid spam detection approach combining machine learning with rule-based keyword detection.</p></div>', unsafe_allow_html=True)
    st.subheader("Enter SMS Message")
    message = st.text_area(
        "Message", placeholder="Example: Congratulations! You have won a prize. Claim your reward now...",
        height=150, label_visibility="collapsed", key="message_input",
    )
    analyze, clear = st.columns([3, 1])
    with analyze:
        submitted = st.button("🔍 Analyze Message", type="primary", width="stretch")
    with clear:
        if st.button("Clear", width="stretch"):
            st.session_state.message_input = ""
            st.session_state.pop("last_result", None)
            st.rerun()
    if submitted:
        if not message.strip():
            st.warning("Please enter an SMS message before analyzing.")
        elif len(message.strip()) < 3:
            st.warning("Please enter a little more text for a meaningful analysis.")
        else:
            try:
                st.session_state.last_result = predict_message(message, bundle)
            except ValueError as exc:
                st.warning(str(exc))
    result = st.session_state.get("last_result")
    if result:
        spam = result["is_spam"]
        title = "🔴 SPAM DETECTED" if spam else "🟢 NOT SPAM"
        cls = "spam" if spam else "ham"
        st.markdown(f'<div class="result {cls}"><h2>{title}</h2><strong>Confidence: {result["confidence"]*100:.2f}%</strong></div>', unsafe_allow_html=True)
        st.progress(min(result["confidence"], 1.0), text=f"Confidence {result['confidence']*100:.2f}%")
        st.subheader("Detection Analysis")
        st.info(result["reason"])
        if result["keyword_triggered"]:
            st.caption("The rule-based detector contributed to this decision before the ML probability was applied.")
        else:
            st.caption(f"Model spam probability: {result['spam_probability']*100:.2f}% (threshold: 35%).")
    st.subheader("🧪 Try Sample Messages")
    samples = {
        "Spam example": "Congratulations! You have won a cash prize. Claim your reward now!",
        "Normal example": "Hey, are you coming to class tomorrow?",
    }
    sample_cols = st.columns(2)
    for col, (label, sample) in zip(sample_cols, samples.items()):
        with col:
            st.caption(label)
            if st.button(sample, key=label, width="stretch"):
                st.session_state.message_input = sample
                st.rerun()

elif page == "📊 Dashboard":
    st.title("📊 Dataset Overview")
    metric_cards([("Total Messages", f"{stats['total']:,}"), ("Spam Messages", f"{stats['spam']:,}"), ("Ham Messages", f"{stats['ham']:,}"), ("Spam Percentage", f"{stats['spam_pct']:.2f}%")])
    st.subheader("Spam vs Ham Distribution")
    st.bar_chart({"Messages": {"Ham": stats["ham"], "Spam": stats["spam"]}})
    st.caption("Statistics are calculated from the supplied SMS Spam Collection dataset.")

elif page == "🤖 Model Information":
    st.title("🤖 Model Information")
    info = [
        ("Dataset", "SMS Spam Collection"), ("Algorithm", "Logistic Regression"),
        ("Feature Extraction", "TF-IDF"), ("N-gram Range", "Unigrams + Bigrams"),
        ("Train/Test Split", "80% / 20%"), ("Random State", "42"),
        ("Spam Threshold", "35%"), ("Text Preprocessing", "Regex-based cleaning"),
        ("Detection Strategy", "Hybrid ML + Rule-Based"),
    ]
    for start in range(0, len(info), 3):
        metric_cards(info[start:start + 3])
        st.write("")
    st.subheader("📈 Model Performance")
    m = bundle["metrics"]
    metric_cards([("Accuracy", f"{m['accuracy']*100:.2f}%"), ("Precision", f"{m['precision']*100:.2f}%"), ("Recall", f"{m['recall']*100:.2f}%"), ("F1 Score", f"{m['f1']*100:.2f}%")])
    st.subheader("Confusion Matrix")
    st.dataframe({"Actual Ham": [m["confusion_matrix"][0, 0], m["confusion_matrix"][1, 0]], "Actual Spam": [m["confusion_matrix"][0, 1], m["confusion_matrix"][1, 1]]}, hide_index=True, width="stretch")

elif page == "⚙️ How It Works":
    st.title("⚙️ How SpamShield Works")
    steps = ["Data Collection", "Text Cleaning", "Train/Test Split", "TF-IDF Feature Extraction", "Logistic Regression", "Strong Keyword Detection", "Final Spam Prediction"]
    for index, step in enumerate(steps):
        st.markdown(f"**{index + 1}. {step}**")
        if index < len(steps) - 1:
            st.markdown("<div style='color:#7990ad;text-align:center;font-size:1.25rem'>↓</div>", unsafe_allow_html=True)
    st.info("SpamShield is hybrid: known high-signal keywords can trigger a rule-based decision, while other messages are classified by the trained TF-IDF + Logistic Regression model.")

else:
    if page == "ℹ️ About Project":
        st.title("ℹ️ About SpamShield")
        st.write("SpamShield is an SMS spam detection system using Natural Language Processing and Machine Learning.")
        st.write("Its objective is to automatically identify potentially malicious or unwanted SMS messages and provide users with an understandable prediction and confidence score.")
        st.subheader("Technologies Used")
        st.write("Python • Pandas • Scikit-learn • TF-IDF • Logistic Regression • Rule-based keyword detection • Streamlit")

st.markdown('<div class="footer">SpamShield | AI-Based SMS Detection | Machine Learning Project</div>', unsafe_allow_html=True)