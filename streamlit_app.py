
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Breast Cancer Diagnosis AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f8ff; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px; }
    h1 { color: #1f77b4; text-align: center; }
    .prediction-box { border-radius: 15px; padding: 20px; text-align: center; }
    .malignant { background-color: #ffcccc; color: #cc0000; }
    .benign { background-color: #ccffcc; color: #009900; }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    with open('breast_cancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('model_card.json', 'r') as f:
        model_card = json.load(f)
    return model, scaler, le, model_card

model, scaler, le, model_card = load_models()

# Title and Description
st.title("🏥 Breast Cancer Diagnosis AI")
st.markdown("### AI-Powered Diagnostic Support System")
st.markdown("""
This application uses a trained machine learning model to assist clinicians in 
breast cancer diagnosis based on Fine Needle Aspiration (FNA) cytology features.
**⚠️ Note**: This is a decision support tool and should be used alongside clinical judgment.
""")

st.divider()

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Algorithm", model_card['model_name'])
    st.metric("Test Accuracy", f"{model_card['test_accuracy']:.2%}")
    st.metric("ROC-AUC", f"{model_card['test_auc']:.4f}")
    st.metric("Sensitivity", f"{model_card['test_sensitivity']:.2%}")
    st.metric("Specificity", f"{model_card['test_specificity']:.2%}")
    
    st.divider()
    st.header("📋 Input Method")
    input_method = st.radio("Select input method:", ["Manual Entry", "Upload CSV"])

# Main Content
tabs = st.tabs(["🔮 Prediction", "📈 Model Performance", "ℹ️ About"])

with tabs[0]:  # Prediction Tab
    st.header("Patient Diagnosis Prediction")
    
    col1, col2 = st.columns(2)
    
    if input_method == "Manual Entry":
        st.subheader("Enter FNA Cytology Features")
        
        # Create sliders for features
        features = []
        feature_cols = st.columns(3)
            
    # Feature names mapping for user-friendly labels
    feature_names = {
        0: "Radius (Cell Size)",
        1: "Texture (Surface Variation)",
        2: "Perimeter (Cell Boundary)",
        3: "Area (Cell Size)",
        4: "Smoothness (Surface Texture)",
        5: "Compactness (Cell Shape)",
        6: "Concavity (Cell Indentation)",
        7: "Concave Points (Irregular Points)",
        8: "Symmetry (Cell Balance)",
        9: "Fractal Dimension (Complexity)",
        10: "Radius Variation",
        11: "Texture Variation",
        12: "Perimeter Variation",
        13: "Area Variation",
        14: "Smoothness Variation",
        15: "Compactness Variation",
        16: "Concavity Variation",
        17: "Concave Points Variation",
        18: "Symmetry Variation",
        19: "Fractal Dimension Variation",
        20: "Radius (Worst Case)",
        21: "Texture (Worst Case)",
        22: "Perimeter (Worst Case)",
        23: "Area (Worst Case)",
        24: "Smoothness (Worst Case)",
        25: "Compactness (Worst Case)",
        26: "Concavity (Worst Case)",
        27: "Concave Points (Worst Case)",
        28: "Symmetry (Worst Case)",
        29: "Fractal Dimension (Worst Case)"
    }
        
        for i in range(30):
            col = feature_cols[i % 3]
            with col:
                # Provide reasonable ranges based on typical clinical values
                min_val = 0 if i < 10 else (0 if i < 20 else 0)
                max_val = 30 if i % 10 == 0 else (40 if i % 10 == 1 else 200)
                
                feature_val = st.slider(
feature_names[i],
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(max_val/2),
                    step=0.1,
                    key=f"feature_{i}"
                )
                features.append(feature_val)
        
        if st.button("🔍 Make Prediction", key="predict_btn", use_container_width=True):
            # Prepare data
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # Display Results
            st.divider()
            st.subheader("🎯 Prediction Result")
            
            diagnosis = le.inverse_transform([prediction])[0]
            confidence = max(probability) * 100
            
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            
            with col_pred1:
                st.metric("Diagnosis", diagnosis)
            
            with col_pred2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            with col_pred3:
                risk_level = "🔴 High Risk" if diagnosis == "Malignant" and confidence > 90 else "🟡 Moderate" if confidence > 70 else "🟢 Low Risk"
                st.metric("Risk Level", risk_level)
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(x=['Benign', 'Malignant'], y=probability*100, 
                       marker_color=['#00cc00', '#ff0000'])
            ])
            fig.update_layout(title="Prediction Probabilities (%)", yaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical Interpretation
            st.divider()
            st.subheader("📋 Clinical Interpretation")
            
            if diagnosis == "Malignant":
                st.warning("""
                **Findings suggest MALIGNANT characteristics**
                - Recommend immediate clinical follow-up
                - Consider biopsy or advanced imaging
                - Close monitoring advised
                """)
            else:
                st.success("""
                **Findings suggest BENIGN characteristics**
                - Low risk of malignancy
                - Regular screening recommended
                - Continue routine monitoring
                """)

with tabs[1]:  # Performance Tab
    st.header("📊 Model Performance Metrics")
    
    metric_cols = st.columns(4)
    metrics = {
        "Accuracy": model_card['test_accuracy'],
        "Sensitivity": model_card['test_sensitivity'],
        "Specificity": model_card['test_specificity'],
        "ROC-AUC": model_card['test_auc']
    }
    
    for col, (metric_name, value) in zip(metric_cols, metrics.items()):
        with col:
            st.metric(metric_name, f"{value:.4f}" if value < 1 else f"{value:.2%}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Goals")
        targets = {
            "Accuracy": 0.95,
            "Sensitivity": 0.98,
            "Specificity": 0.92,
            "F1-Score": 0.95
        }
        
        fig_goals = go.Figure(data=[
            go.Bar(name="Target", x=list(targets.keys()), y=list(targets.values()), marker_color='lightblue'),
            go.Bar(name="Achieved", x=list(metrics.keys()), y=list(metrics.values()), marker_color='darkblue')
        ])
        fig_goals.update_layout(title="Performance vs Targets", barmode='group')
        st.plotly_chart(fig_goals, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Information")
        st.info(f"""
        - **Total Samples**: {model_card['training_samples'] + model_card['test_samples']}
        - **Training Samples**: {model_card['training_samples']}
        - **Test Samples**: {model_card['test_samples']}
        - **Features**: {model_card['features']}
        - **Classes**: Benign, Malignant
        - **Dataset**: Breast Cancer Wisconsin Diagnostic
        """)

with tabs[2]:  # About Tab
    st.header("ℹ️ About This Application")
    
    st.subheader("🎯 Purpose")
    st.write("""
    This application provides AI-assisted diagnosis for breast cancer detection
    based on Fine Needle Aspiration (FNA) cytology features. It uses a Support
    Vector Machine (SVM) classifier trained on the Breast Cancer Wisconsin dataset.
    """)
    
    st.subheader("📚 Model Details")
    st.json(model_card)
    
    st.subheader("⚠️ Disclaimer")
    st.warning("""
    **Important**: This tool is designed to assist medical professionals and should
    NOT be used as a standalone diagnostic tool. All clinical decisions must be made
    by qualified healthcare professionals after comprehensive clinical evaluation.
    """)
    
    st.subheader("🔬 Features Explained")
    st.info("""
    The model uses 30 features derived from FNA samples:
    - Radius, Texture, Perimeter, Area
    - Smoothness, Compactness, Concavity
    - Concave Points, Symmetry, Fractal Dimension
    Each feature is computed for mean, standard error, and worst case.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; font-size: 12px; color: #666;'>
    <p><b>Powered by:</b> Nuraini Usman</p>
    <p>Email: <a href='mailto:message2nuraini@gmail.com'>message2nuraini@gmail.com</a></p>
    <p>&copy; 2025 Nuraini Usman. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
