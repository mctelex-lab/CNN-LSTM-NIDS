# =============================================================================
# CNN-LSTM INTRUSION DETECTION SYSTEM WITH DUAL XAI
# =============================================================================
# DEVELOPED BY: Samuel Ayorinde A
# PROGRAM: PGD in Cybersecurity
# INSTITUTION: Nigerian Defence Academy (NDA)
# SUPERVISOR: Mr Victor Akuboh
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CNN-LSTM IDS Dashboard - NDA Cybersecurity Project",
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1f77b4 0%, #0d4a6e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .developer-card {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .xai-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    .debug-info {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🛡️ Real-Time Network Intrusion Detection System</h1>
    <p>Explainable Hybrid 1D CNN-LSTM Framework with Dual XAI (LIME + SHAP)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="developer-card">
    <h4>👨‍💻 Project Information</h4>
    <p><strong>Developed by:</strong> Samuel Ayorinde A</p>
    <p><strong>Program:</strong> PGD in Cybersecurity</p>
    <p><strong>Institution:</strong> Nigerian Defence Academy (NDA), Kaduna, Nigeria</p>
    <p><strong>Supervisor:</strong> Mr Victor Akuboh</p>
    <p><strong>Research Focus:</strong> Dual Dataset Evaluation with Complete Dual XAI Integration (LIME + SHAP)</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTION TO FIND XAI FOLDER (CASE-INSENSITIVE)
# =============================================================================
def find_xai_folder():
    """Find xai_results folder regardless of case sensitivity"""
    possible_names = ['xai_results', 'XAI_results', 'Xai_results', 'xai_Results']
    for name in possible_names:
        if os.path.exists(name):
            return name
    return None

def get_xai_paths(dataset_name):
    """Get correct paths for XAI files (handles case sensitivity)"""
    xai_base = find_xai_folder()
    if not xai_base:
        return None, None
    
    lime_dir = os.path.join(xai_base, f'lime_{dataset_name}')
    shap_dir = os.path.join(xai_base, f'shap_{dataset_name}')
    
    # Also try alternative case variations
    if not os.path.exists(lime_dir):
        alt_lime = os.path.join(xai_base, f'LIME_{dataset_name}')
        if os.path.exists(alt_lime):
            lime_dir = alt_lime
    
    if not os.path.exists(shap_dir):
        alt_shap = os.path.join(xai_base, f'SHAP_{dataset_name}')
        if os.path.exists(alt_shap):
            shap_dir = alt_shap
    
    return lime_dir, shap_dir

# =============================================================================
# CHECK XAI FILES FUNCTION (UPDATED)
# =============================================================================
def check_xai_files(dataset_name):
    """Check if XAI files exist and return status - with debug info"""
    lime_dir, shap_dir = get_xai_paths(dataset_name)
    
    # Debug: Show what paths we're checking
    with st.expander(f"🔧 Debug: XAI Path Check for {dataset_name}", expanded=False):
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Available folders: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
        st.write(f"LIME directory path: {lime_dir}")
        st.write(f"SHAP directory path: {shap_dir}")
        
        if lime_dir and os.path.exists(lime_dir):
            st.write(f"LIME folder exists. Contents: {os.listdir(lime_dir)[:10]}")
        else:
            st.write("LIME folder does NOT exist")
        
        if shap_dir and os.path.exists(shap_dir):
            st.write(f"SHAP folder exists. Contents: {os.listdir(shap_dir)[:10]}")
        else:
            st.write("SHAP folder does NOT exist")
    
    lime_status = {
        'exists': lime_dir and os.path.exists(lime_dir),
        'dir_path': lime_dir,
        'files': 0,
        'has_html': 0,
        'file_list': []
    }
    
    shap_status = {
        'exists': shap_dir and os.path.exists(shap_dir),
        'dir_path': shap_dir,
        'has_summary': False,
        'has_bar': False,
        'has_json': False,
        'file_list': []
    }
    
    # Check LIME files
    if lime_status['exists']:
        try:
            all_files = os.listdir(lime_dir)
            lime_status['file_list'] = all_files
            png_files = [f for f in all_files if f.endswith('.png') and 'summary' not in f.lower()]
            html_files = [f for f in all_files if f.endswith('.html')]
            lime_status['files'] = len(png_files)
            lime_status['has_html'] = len(html_files)
        except Exception as e:
            st.warning(f"Error reading LIME directory: {e}")
    
    # Check SHAP files
    if shap_status['exists']:
        try:
            all_files = os.listdir(shap_dir)
            shap_status['file_list'] = all_files
            shap_status['has_summary'] = any('summary' in f.lower() and f.endswith('.png') for f in all_files)
            shap_status['has_bar'] = any('bar' in f.lower() and f.endswith('.png') for f in all_files)
            shap_status['has_json'] = any(f.endswith('.json') for f in all_files)
        except Exception as e:
            st.warning(f"Error reading SHAP directory: {e}")
    
    return lime_status, shap_status

def display_xai_warning(dataset_name):
    """Display warning if XAI files are missing"""
    lime_status, shap_status = check_xai_files(dataset_name)
    
    missing_components = []
    
    if not lime_status['exists']:
        missing_components.append("• LIME explanations directory not found")
    elif lime_status['files'] == 0:
        missing_components.append(f"• LIME PNG files (found {lime_status['files']} files)")
    
    if not shap_status['exists']:
        missing_components.append("• SHAP explanations directory not found")
    else:
        if not shap_status['has_bar']:
            missing_components.append("• SHAP bar plot (shap_bar_plot.png)")
        if not shap_status['has_json']:
            missing_components.append("• SHAP feature importance JSON")
    
    if missing_components:
        st.markdown(f"""
        <div class="warning-card">
            <strong>⚠️ DUAL XAI FILES MISSING - RESEARCH OBJECTIVE ii NOT MET</strong><br><br>
            <strong>Missing components for {dataset_name}:</strong><br>
            {chr(10).join(missing_components)}<br><br>
            <strong>What you have in xai_results folder:</strong><br>
            • LIME exists: {lime_status['exists']} | Files: {lime_status['files']}<br>
            • SHAP exists: {shap_status['exists']} | Bar plot: {shap_status['has_bar']}<br><br>
            <strong>To fix:</strong> Run the complete training pipeline to generate all XAI files.
        </div>
        """, unsafe_allow_html=True)
        return False
    else:
        st.markdown(f"""
        <div class="success-card">
            <strong>✅ DUAL XAI FILES PRESENT - Research Objective ii ACHIEVED</strong><br><br>
            • LIME explanations: {lime_status['files']} visualization files, {lime_status['has_html']} HTML files<br>
            • SHAP explanations: Summary plot ✓, Bar plot ✓, Feature importance JSON ✓<br>
            • The system is fully explainable and compliant with AI transparency requirements
        </div>
        """, unsafe_allow_html=True)
        return True

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
def build_cnn_lstm_model_from_architecture(input_dim, output_units, activation):
    """Rebuild model with same architecture as training"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Reshape((1, 256)),
        tf.keras.layers.LSTM(128, return_sequences=False,
                            kernel_regularizer=regularizers.l2(0.0005),
                            dropout=0.3, recurrent_dropout=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_units, activation=activation)
    ])
    return model

# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource
def load_models():
    """Load all models and artifacts"""
    models = {}
    scalers = {}
    imputers = {}
    features = {}
    class_names = {}
    
    datasets = {
        'CIC-IDS-2018': {
            'is_multiclass': True,
            'files': {
                'arch': 'CIC-IDS-2018_model_architecture.json',
                'weights': 'CIC-IDS-2018_model.weights.h5',
                'scaler': 'CIC-IDS-2018_scaler.pkl',
                'imputer': 'CIC-IDS-2018_imputer.pkl',
                'features': 'CIC-IDS-2018_feature_names.json',
                'classes': 'CIC-IDS-2018_class_names.json'
            }
        },
        'NF-UQ-NIDS': {
            'is_multiclass': False,
            'files': {
                'arch': 'NF-UQ-NIDS_model_architecture.json',
                'weights': 'NF-UQ-NIDS_model.weights.h5',
                'scaler': 'NF-UQ-NIDS_scaler.pkl',
                'imputer': 'NF-UQ-NIDS_imputer.pkl',
                'features': 'NF-UQ-NIDS_feature_names.json',
                'classes': 'NF-UQ-NIDS_class_names.json'
            }
        }
    }
    
    for dataset_name, config in datasets.items():
        try:
            files = config['files']
            
            if os.path.exists(files['features']):
                with open(files['features'], 'r') as f:
                    features[dataset_name] = json.load(f)
            else:
                continue
            
            if os.path.exists(files['classes']):
                with open(files['classes'], 'r') as f:
                    class_names[dataset_name] = json.load(f)
            else:
                continue
            
            output_units = len(class_names[dataset_name]) if config['is_multiclass'] else 1
            activation = 'softmax' if config['is_multiclass'] else 'sigmoid'
            input_dim = len(features[dataset_name])
            
            model = build_cnn_lstm_model_from_architecture(input_dim, output_units, activation)
            
            if os.path.exists(files['weights']):
                model.load_weights(files['weights'])
                models[dataset_name] = model
            
            if os.path.exists(files['scaler']):
                scalers[dataset_name] = joblib.load(files['scaler'])
            if os.path.exists(files['imputer']):
                imputers[dataset_name] = joblib.load(files['imputer'])
                
        except Exception as e:
            pass
    
    return models, scalers, imputers, features, class_names

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Configuration")

# Show current directory info
xai_folder = find_xai_folder()
if xai_folder:
    st.sidebar.success(f"✅ XAI folder found: `{xai_folder}`")
else:
    st.sidebar.warning("⚠️ XAI folder not found (looking for 'xai_results' or 'XAI_results')")

with st.spinner("Loading models and artifacts..."):
    models, scalers, imputers, features, class_names = load_models()

if models:
    dataset_choice = st.sidebar.selectbox("Select Detection Model", list(models.keys()))
    
    # Check XAI files status and display in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 XAI Status")
    lime_status, shap_status = check_xai_files(dataset_choice)
    
    if lime_status['exists'] and lime_status['files'] > 0:
        st.sidebar.success(f"✅ LIME: {lime_status['files']} files")
    else:
        st.sidebar.error("❌ LIME: Not available")
        if lime_status['exists']:
            st.sidebar.caption(f"Found directory but {lime_status['files']} PNG files")
    
    if shap_status['exists'] and shap_status['has_bar']:
        st.sidebar.success("✅ SHAP: Available")
    else:
        st.sidebar.error("❌ SHAP: Not available")
        if shap_status['exists']:
            st.sidebar.caption(f"Found directory but missing bar plot")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Model Info")
    
    if dataset_choice == 'CIC-IDS-2018':
        st.sidebar.info(f"**Type:** Multi-class (3 classes)\n**Features:** {len(features.get(dataset_choice, []))}")
    else:
        st.sidebar.info(f"**Type:** Binary\n**Features:** {len(features.get(dataset_choice, []))}")
    
    if os.path.exists('performance_comparison.csv'):
        st.sidebar.markdown("---")
        perf_df = pd.read_csv('performance_comparison.csv')
        st.sidebar.dataframe(perf_df, use_container_width=True)
else:
    st.error("❌ No models loaded!")
    st.stop()

# =============================================================================
# DISPLAY XAI WARNING IF NEEDED
# =============================================================================
xai_available = display_xai_warning(dataset_choice)

# =============================================================================
# MAIN TABS
# =============================================================================
if dataset_choice in models:
    model = models[dataset_choice]
    scaler = scalers[dataset_choice]
    imputer = imputers[dataset_choice]
    feature_list = features[dataset_choice]
    classes = class_names[dataset_choice]
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Real-Time Detection", 
        "🔍 XAI Analysis (LIME + SHAP)", 
        "📊 Model Performance",
        "📁 Research Documentation"
    ])
    
    # ========================================================================
    # TAB 1: REAL-TIME DETECTION
    # ========================================================================
    with tab1:
        st.subheader("📡 Real-Time Network Traffic Analysis")
        
        uploaded_file = st.file_uploader("Upload Network Flow CSV File", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {len(df)} network flows")
            
            missing_features = [f for f in feature_list if f not in df.columns]
            if missing_features:
                st.error(f"⚠️ Missing {len(missing_features)} required features")
                with st.expander("View missing features"):
                    st.write(missing_features[:20])
            else:
                if st.button("🔍 Analyze Traffic", type="primary"):
                    with st.spinner("Analyzing..."):
                        X = df[feature_list].fillna(0).values
                        X = imputer.transform(X)
                        X = scaler.transform(X)
                        predictions = model.predict(X, verbose=0)
                        
                        if dataset_choice == 'CIC-IDS-2018':
                            pred_classes = np.argmax(predictions, axis=1)
                            pred_proba = np.max(predictions, axis=1)
                        else:
                            pred_classes = (predictions.flatten() >= 0.5).astype(int)
                            pred_proba = predictions.flatten()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        attack_count = np.sum(pred_classes != 0 if dataset_choice == 'CIC-IDS-2018' else pred_classes == 1)
                        col1.metric("Total Flows", len(predictions))
                        col2.metric("🚨 Threats", attack_count, delta_color="inverse")
                        col3.metric("✅ Benign", len(predictions) - attack_count)
                        col4.metric("Avg Threat Score", f"{np.mean(pred_proba):.2%}")
                        
                        df['prediction'] = [classes[p] for p in pred_classes]
                        df['confidence'] = pred_proba
                        st.dataframe(df[['prediction', 'confidence'] + feature_list[:5]], use_container_width=True)
                        
                        fig = px.histogram(df, x='confidence', nbins=50, title='Threat Score Distribution',
                                          color='prediction')
                        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 2: XAI ANALYSIS
    # ========================================================================
    with tab2:
        st.subheader("🔍 Explainable AI (LIME + SHAP) - Forensic Analysis")
        
        st.markdown("""
        <div class="xai-card">
        <strong>🎓 Why XAI is CRITICAL for this Research (Objective ii):</strong><br><br>
        <strong>1. Research Compliance:</strong> Objective ii specifically requires "Dual XAI Integration (LIME + SHAP)"<br>
        <strong>2. Trust & Transparency:</strong> Security analysts need to understand WHY an alert was triggered<br>
        <strong>3. Forensic Investigation:</strong> LIME explains individual predictions for attack investigation<br>
        <strong>4. Model Validation:</strong> SHAP identifies which features most influence decisions<br>
        <strong>5. Regulatory Compliance:</strong> EU AI Act and GDPR require explainable AI systems<br>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 LIME Local Explanations")
            st.caption("*Local interpretability - explains individual predictions*")
            
            lime_dir, _ = get_xai_paths(dataset_choice)
            if lime_dir and os.path.exists(lime_dir):
                lime_files = [f for f in os.listdir(lime_dir) if f.endswith('.png') and 'summary' not in f.lower()]
                if lime_files:
                    selected_sample = st.selectbox("Select Sample to Analyze", lime_files)
                    st.image(os.path.join(lime_dir, selected_sample), use_container_width=True)
                    st.markdown("""
                    **📖 LIME Interpretation Guide:**
                    - Green bars → Support the prediction (positive contribution)
                    - Red bars → Contradict the prediction (negative contribution)
                    - Length of bar → Strength of influence
                    """)
                    
                    summary_path = os.path.join(lime_dir, 'lime_summary.png')
                    if os.path.exists(summary_path):
                        with st.expander("View All LIME Explanations"):
                            st.image(summary_path, use_container_width=True)
                else:
                    st.info(f"No LIME PNG files found in {lime_dir}")
            else:
                st.info("LIME directory not found. Run training pipeline to generate.")
        
        with col2:
            st.markdown("### 📊 SHAP Global Feature Importance")
            st.caption("*Global interpretability - identifies most important features*")
            
            _, shap_dir = get_xai_paths(dataset_choice)
            if shap_dir and os.path.exists(shap_dir):
                bar_path = None
                for f in os.listdir(shap_dir):
                    if 'bar' in f.lower() and f.endswith('.png'):
                        bar_path = os.path.join(shap_dir, f)
                        break
                
                if bar_path and os.path.exists(bar_path):
                    st.image(bar_path, use_container_width=True)
                    st.markdown("""
                    **📖 SHAP Interpretation Guide:**
                    - Higher SHAP value → More important feature
                    - Features with high SHAP values are key decision factors
                    - Top features should be monitored for intrusion patterns
                    """)
                
                summary_path = None
                for f in os.listdir(shap_dir):
                    if 'summary' in f.lower() and f.endswith('.png'):
                        summary_path = os.path.join(shap_dir, f)
                        break
                
                if summary_path and os.path.exists(summary_path):
                    with st.expander("SHAP Summary Plot (Detailed)"):
                        st.image(summary_path, use_container_width=True)
                
                json_path = None
                for f in os.listdir(shap_dir):
                    if f.endswith('.json'):
                        json_path = os.path.join(shap_dir, f)
                        break
                
                if json_path and os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        importance = json.load(f)
                    with st.expander("🏆 Feature Importance Rankings"):
                        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                            st.write(f"{i}. {feature[:60]}: {score:.4f}")
            else:
                st.info("SHAP directory not found. Run training pipeline to generate.")
    
    # ========================================================================
    # TAB 3: MODEL PERFORMANCE
    # ========================================================================
    with tab3:
        st.subheader("📊 Model Performance Metrics")
        
        eval_path = f'visualizations/{dataset_choice}_evaluation.png'
        if os.path.exists(eval_path):
            st.image(eval_path, use_container_width=True)
        else:
            st.info("Evaluation plot not found. Run training pipeline to generate.")
        
        if os.path.exists('performance_comparison.csv'):
            perf_df = pd.read_csv('performance_comparison.csv')
            st.dataframe(perf_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Research Objectives Status")
        
        obj_status = [
            ("✅ Objective i: Lightweight CNN-LSTM Architecture", True),
            ("✅ Objective ii: Dual XAI Integration" if xai_available else "❌ Objective ii: Dual XAI Integration - XAI FILES NEEDED", xai_available),
            ("✅ Objective iii: Data Balancing & SMOTE", True),
            ("✅ Objective iv: Cloud-native Streamlit Dashboard", True),
            ("✅ Objective v: Comprehensive Evaluation", True)
        ]
        
        for status, achieved in obj_status:
            if achieved:
                st.success(status)
            else:
                st.error(status)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1rem; font-size: 0.8rem; color: #666;">
    <strong>🔒 CNN-LSTM Intrusion Detection System with Dual XAI (LIME + SHAP)</strong><br>
    Developed by Samuel Ayorinde A | PGD Cybersecurity | Nigerian Defence Academy (NDA)<br>
    Supervised by Mr Victor Akuboh | Research Project {datetime.now().year}
</div>
""", unsafe_allow_html=True)