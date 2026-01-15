import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3, h4 {color: #00e5ff;}
.metric-container {background-color:#161b22; padding:15px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.markdown("""
**An intelligent cybersecurity solution that detects malicious network activity  
using Machine Learning (Random Forest Classifier).**
""")

st.info("""
🔍 **Purpose:** Detect cyber attacks such as DDoS, port scans, and abnormal traffic  
📊 **Approach:** Supervised Machine Learning  
🧠 **Model:** Random Forest Classifier  
""")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000

    data = {
        'Destination_Port': np.random.randint(1, 65535, n_samples),
        'Flow_Duration': np.random.randint(100, 100000, n_samples),
        'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
        'Packet_Length_Mean': np.random.uniform(10, 1500, n_samples),
        'Active_Mean': np.random.uniform(0, 1000, n_samples),
        'Label': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Attack behavior patterns
    df.loc[df['Label'] == 1, 'Total_Fwd_Packets'] += np.random.randint(
        50, 200, size=df[df['Label'] == 1].shape[0]
    )
    df.loc[df['Label'] == 1, 'Flow_Duration'] = np.random.randint(
        1, 1000, size=df[df['Label'] == 1].shape[0]
    )

    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ System Controls")
st.sidebar.markdown("Adjust model and monitoring parameters")

split_size = st.sidebar.slider("Training Data (%)", 60, 90, 80)
n_estimators = st.sidebar.slider("Random Forest Trees", 50, 300, 150)

st.sidebar.markdown("---")
st.sidebar.markdown("📁 **Dataset:** Simulated CIC-IDS2017-like traffic")

# ---------------- PREPROCESSING ----------------
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - split_size) / 100, random_state=42
)

# ---------------- MODEL TRAINING ----------------
st.divider()
st.subheader("🚀 Model Training & Evaluation")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🔄 Train AI Model"):
        with st.spinner("Training the intrusion detection model..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.success("✅ Model trained successfully!")

    if 'model' in st.session_state:
        st.success("🧠 AI Model is Active")

with col2:
    if 'model' in st.session_state:
        model = st.session_state['model']
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Accuracy", f"{acc*100:.2f}%")
        m2.metric("📦 Total Traffic Samples", len(df))
        m3.metric("🚨 Detected Attacks", int(np.sum(y_pred)))

        st.markdown("### 🔎 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Train the model to view metrics")

# ---------------- FEATURE IMPORTANCE ----------------
st.divider()
st.subheader("📊 Feature Importance (What the AI Learns)")

if 'model' in st.session_state:
    importances = st.session_state['model'].feature_importances_
    fi_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**  
    These features contribute most to attack detection.  
    High packet volume + short duration strongly indicate DDoS behavior.
    """)

# ---------------- LIVE TRAFFIC SIMULATOR ----------------
st.divider()
st.subheader("🧪 Live Network Traffic Analyzer")

st.markdown("""
Simulate incoming network traffic and let the AI decide whether it is **safe or malicious**.
""")

c1, c2, c3, c4 = st.columns(4)
flow_duration = c1.number_input("Flow Duration (ms)", 0, 100000, 500)
total_packets = c2.number_input("Total Packets", 0, 500, 120)
packet_length = c3.number_input("Packet Length Mean", 0, 1500, 600)
active_mean = c4.number_input("Active Mean Time", 0, 1000, 80)

if st.button("🔍 Analyze Traffic"):
    if 'model' in st.session_state:
        input_data = np.array([[80, flow_duration, total_packets, packet_length, active_mean]])
        prediction = st.session_state['model'].predict(input_data)[0]
        probability = st.session_state['model'].predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 MALICIOUS TRAFFIC DETECTED Attack Probability: **{probability*100:.2f}%**")
            st.markdown("""
            **Possible Causes:**
            - High packet flooding
            - Very short flow duration
            - Abnormal traffic behavior
            """)
        else:
            st.success(f"✅ BENIGN TRAFFIC Attack Probability: **{probability*100:.2f}%**")
    else:
        st.warning("Train the model first!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9da5b4;">
    🛡️ <b>AI-Based Network Intrusion Detection System</b><br>
    Developed by <b>Roni Seikh</b><br><br>
    🔗 <a href="https://github.com/Roni-Seikh" target="_blank">GitHub: Roni-Seikh</a> |
    💼 <a href="https://www.linkedin.com/in/roniseikh" target="_blank">LinkedIn: roniseikh</a>
</div>
""", unsafe_allow_html=True)
