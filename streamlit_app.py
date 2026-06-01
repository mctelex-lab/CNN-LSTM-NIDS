# =============================================================================
# CNN-LSTM INTRUSION DETECTION SYSTEM WITH DUAL XAI
# PROFESSIONAL DASHBOARD IMPLEMENTATION
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
import time
import random
from datetime import datetime
from PIL import Image

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NDA CyberShield | IDS Dashboard",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# =============================================================================
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(102, 252, 241, 0.1);
        margin-bottom: 2rem;
        border-bottom: 4px solid #66fcf1;
        text-align: center;
    }
    .main-header h1 {
        color: #66fcf1;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(102, 252, 241, 0.5);
    }
    .main-header p {
        color: #45a29e;
        font-size: 1.2rem;
        font-weight: 500;
    }

    /* Card Styling - Glassmorphism */
    .metric-card {
        background: rgba(31, 40, 51, 0.6);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #45a29e;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.3);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #66fcf1;
        text-shadow: 0 0 5px rgba(102, 252, 241, 0.5);
    }
    .metric-label {
        color: #c5c6c7;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 10px;
    }

    /* Status Indicators */
    .status-safe {
        color: #45a29e;
        font-weight: bold;
    }
    .status-threat {
        color: #ff0055;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(255, 0, 85, 0.5);
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid #45a29e;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #66fcf1;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dataframe Styling */
    .stDataFrame {
        border: 1px solid #45a29e;
        border-radius: 8px;
        background-color: #0b0c10;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #45a29e 0%, #66fcf1 100%);
        color: #0b0c10;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #66fcf1;
        transform: scale(1.05);
    }

    /* Explanation Box */
    .explanation-box {
        background-color: #1f2833;
        border-left: 5px solid #66fcf1;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    
    /* Sidebar Profile */
    .sidebar-profile {
        background-color: #0b0c10;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #45a29e;
        margin-bottom: 1rem;
    }
    .sidebar-profile h4 {
        color: #66fcf1;
        margin-bottom: 0.5rem;
    }
    .sidebar-profile p {
        color: #c5c6c7;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER SECTION
# =============================================================================
def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>🛡️ NDA CyberShield IDS</h1>
            <p>Explainable Hybrid 1D CNN-LSTM Framework with Dual XAI (LIME + SHAP)</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Developed by: **Samuel Ayorinde A** | PGD Cybersecurity | Nigerian Defence Academy")
    with col2:
        st.caption(f"System Time: {datetime.now().strftime('%H:%M:%S')}")

# =============================================================================
# DATASET FEATURE DEFINITIONS
# =============================================================================
NF_UQ_NIDS_FEATURES = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", 
    "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", 
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", 
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", 
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", 
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", 
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", 
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", 
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", 
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", 
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE", 
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", 
    "Label", "Attack", "Dataset"
]

CIC_2018_FEATURES = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", 
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", 
    "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", 
    "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", 
    "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", 
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", 
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", 
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", 
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", 
    "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt", 
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", 
    "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", 
    "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", 
    "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", 
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", 
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min", 
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", 
    "Idle Max", "Idle Min", "Label"
]

# =============================================================================
# DATA SIMULATOR (For Real-Time Demo)
# =============================================================================
def generate_synthetic_traffic(n_samples=10, feature_names=None, dataset_type="CIC-IDS-2018"):
    """
    Generates synthetic network traffic data that mimics the statistical 
    properties of CIC-IDS-2018 or NF-UQ-NIDS datasets based on provided feature names.
    """
    if feature_names is None:
        # Fallback generic features if none provided
        feature_names = ['Duration', 'Src_Bytes', 'Dst_Bytes', 'Protocol', 'Flag']

    data = {}
    
    for feat in feature_names:
        # Skip target labels during generation, we will add them manually at the end
        if feat.lower() in ['label', 'attack', 'dataset']:
            continue
            
        feat_lower = feat.lower()
        
        # 1. Time/Duration Features (Exponential distribution: many short, few long)
        if any(k in feat_lower for k in ['duration', 'time', 'iat', 'active', 'idle']):
            data[feat] = np.abs(np.random.exponential(scale=1000000, size=n_samples)) # Microseconds scale
            
        # 2. Byte/Size/Length/Throughput Features (Log-normal: skewed right, no negatives)
        elif any(k in feat_lower for k in ['bytes', 'length', 'size', 'win', 'throughput', 'seg', 'len']):
            data[feat] = np.random.lognormal(mean=4, sigma=1.5, size=n_samples).astype(float)
            
        # 3. Packet Counts/Flags/Retransmissions (Poisson/Integer: discrete counts)
        elif any(k in feat_lower for k in ['pkts', 'packets', 'count', 'cnt', 'num', 'retransmit', 'query', 'flag', 'type', 'code', 'ttl']):
            data[feat] = np.random.poisson(lam=10, size=n_samples).astype(float)
            
        # 4. Rates/Averages/StdDev (Normal/Uniform: continuous positive values)
        elif any(k in feat_lower for k in ['rate', 'bps', 'pps', 'ratio', 'avg', 'std', 'mean', 'min', 'max', 'var']):
            # Using abs normal to ensure positive rates/stats
            data[feat] = np.abs(np.random.normal(loc=1000, scale=500, size=n_samples))
            
        # 5. Ports (Specific ranges for realism)
        elif 'port' in feat_lower:
            data[feat] = np.random.randint(1, 65535, size=n_samples).astype(float)
        
        # 6. Protocol (Numeric encoding usually 6 for TCP, 17 for UDP)
        elif 'protocol' in feat_lower:
            data[feat] = np.random.choice([6, 17, 1], size=n_samples).astype(float)
        
        # 7. IP Addresses (Generated as integers for simplicity, model expects numeric)
        elif 'addr' in feat_lower or 'ip' in feat_lower:
            # Generating random large integers to simulate IP encoding
            data[feat] = np.random.randint(0, 2**32, size=n_samples).astype(float)
            
        else:
            # Default: Small random floats between 0 and 1
            data[feat] = np.random.uniform(0, 1, size=n_samples)

    df = pd.DataFrame(data)
    
    # Ensure strict non-negative constraints for physical metrics
    for col in df.columns:
        if any(k in col.lower() for k in ['bytes', 'length', 'duration', 'count', 'rate', 'pkt', 'win', 'iat', 'throughput']):
            df[col] = df[col].clip(lower=0)
            
    return df

# =============================================================================
# HEURISTIC EXPLANATION ENGINE (Simulates XAI for Synthetic Data)
# =============================================================================
def generate_forensic_narrative(row, dataset_type):
    """
    Generates a natural language explanation for why a sample might be flagged,
    based on heuristic thresholds. This simulates LIME/SHAP output for live demo.
    """
    reasons = []
    row_dict = row.to_dict()
    
    if dataset_type == "CIC-IDS-2018":
        # Check for high flow duration
        if row_dict.get('Flow Duration', 0) > 1000000:
            reasons.append("Unusually long flow duration suggesting a slow-rate attack or tunneling.")
        # Check for high packet count
        if row_dict.get('Tot Fwd Pkts', 0) > 100:
            reasons.append("High forward packet count indicative of flooding or scanning.")
        # Check for small packet sizes (often associated with scanning)
        if row_dict.get('Pkt Len Mean', 0) < 50:
            reasons.append("Abnormally small average packet size, common in port scanning or heartbeat attacks.")
        # Check for specific flags
        if row_dict.get('FIN Flag Cnt', 0) > 0 and row_dict.get('SYN Flag Cnt', 0) > 0:
            reasons.append("Simultaneous FIN and SYN flags detected, a strong signature of malicious probing.")
            
    elif dataset_type == "NF-UQ-NIDS":
        # Check for high throughput
        if row_dict.get('SRC_TO_DST_AVG_THROUGHPUT', 0) > 5000:
            reasons.append("Exceptional source-to-destination throughput suggesting data exfiltration or DDoS.")
        # Check for retransmissions
        if row_dict.get('RETRANSMITTED_IN_PKTS', 0) > 5:
            reasons.append("High number of retransmitted packets indicating network congestion or aggressive transmission.")
        # Check for TTL anomalies
        if row_dict.get('MIN_TTL', 0) < 10:
            reasons.append("Very low Time-To-Live (TTL) value, potentially indicating spoofed IP addresses or nearby attacker.")

    if not reasons:
        reasons.append("Statistical anomaly detected in feature distribution compared to baseline benign traffic.")
        
    return " | ".join(reasons)

# =============================================================================
# MODEL ARCHITECTURE & LOADING
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

@st.cache_resource
def load_models():
    """Load all models and artifacts with robust error handling for TF 2.18+"""
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
           
            # 1. Check Feature Names
            if not os.path.exists(files['features']):
                st.warning(f"⚠️ Missing feature file for {dataset_name}: {files['features']}")
                continue
            
            with open(files['features'], 'r') as f:
                features[dataset_name] = json.load(f)
           
            # 2. Check Class Names
            if not os.path.exists(files['classes']):
                st.warning(f"⚠️ Missing class file for {dataset_name}: {files['classes']}")
                continue
                
            with open(files['classes'], 'r') as f:
                class_names[dataset_name] = json.load(f)
           
            output_units = len(class_names[dataset_name]) if config['is_multiclass'] else 1
            activation = 'softmax' if config['is_multiclass'] else 'sigmoid'
            input_dim = len(features[dataset_name])
           
            # 3. Build Model Architecture First
            model = build_cnn_lstm_model_from_architecture(input_dim, output_units, activation)
            
            # 4. Load Weights with Compatibility Flags
            if os.path.exists(files['weights']):
                try:
                    # Create a dummy input to build the model weights internally
                    # This ensures layers are initialized before loading
                    dummy_input = np.zeros((1, input_dim))
                    _ = model(dummy_input, training=False)
                    
                    # Load weights
                    model.load_weights(files['weights'])
                    models[dataset_name] = model
                    st.success(f"✅ Loaded model weights for {dataset_name}")
                except Exception as w_err:
                    st.error(f"❌ Failed to load weights for {dataset_name}: {str(w_err)}")
                    st.info("Tip: Ensure TF version matches training. Trying to load with skip_mismatch...")
                    try:
                        # Fallback: Try loading with skip_mismatch if strict loading fails
                        model.load_weights(files['weights'], skip_mismatch=True, by_name=True)
                        models[dataset_name] = model
                        st.success(f"✅ Loaded model weights for {dataset_name} (with mismatch skip)")
                    except Exception as e2:
                        st.error(f"❌ Critical Load Error: {str(e2)}")
            else:
                st.warning(f"⚠️ Weight file not found for {dataset_name}: {files['weights']}")
           
            # 5. Load Scaler/Imputer
            if os.path.exists(files['scaler']):
                scalers[dataset_name] = joblib.load(files['scaler'])
            if os.path.exists(files['imputer']):
                imputers[dataset_name] = joblib.load(files['imputer'])
               
        except Exception as e:
            st.error(f"❌ Critical error loading {dataset_name}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
   
    return models, scalers, imputers, features, class_names

# =============================================================================
# XAI FILE CHECKER
# =============================================================================
def check_xai_files(dataset_name):
    """Check if XAI files exist and return status"""
    lime_dir = os.path.join('XAI_results', f'lime_{dataset_name}')
    shap_dir = os.path.join('XAI_results', f'shap_{dataset_name}')
   
    lime_status = {
        'exists': os.path.exists(lime_dir),
        'files': len([f for f in os.listdir(lime_dir) if f.endswith('.png')]) if os.path.exists(lime_dir) else 0,
        'has_html': len([f for f in os.listdir(lime_dir) if f.endswith('.html')]) if os.path.exists(lime_dir) else 0
    }
   
    shap_status = {
        'exists': os.path.exists(shap_dir),
        'has_summary': os.path.exists(os.path.join(shap_dir, 'shap_summary.png')) if os.path.exists(shap_dir) else False,
        'has_bar': os.path.exists(os.path.join(shap_dir, 'shap_bar_plot.png')) if os.path.exists(shap_dir) else False,
        'has_json': os.path.exists(os.path.join(shap_dir, 'feature_importance.json')) if os.path.exists(shap_dir) else False
    }
   
    return lime_status, shap_status

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================
def main():
    render_header()
    
    # Load Models
    with st.spinner("Initializing Neural Networks..."):
        models, scalers, imputers, features, class_names = load_models()

    # Sidebar Configuration
    st.sidebar.markdown("### ⚙️ Configuration")
    
    # --- SIDEBAR PROFILE INFO ---
    st.sidebar.markdown("""
        <div class="sidebar-profile">
            <h4>👤 Researcher</h4>
            <p><strong>Student:</strong> Samuel Ayorinde A</p>
            <p><strong>Program:</strong> PGD Cybersecurity</p>
            <p><strong>Institution:</strong> Nigerian Defence Academy</p>
            <hr style="border-color: #45a29e;">
            <h4>🎓 Supervision</h4>
            <p><strong>Supervisor:</strong> Mr Victor Akuboh</p>
        </div>
    """, unsafe_allow_html=True)
    
    if models:
        dataset_choice = st.sidebar.selectbox("Select Detection Model", list(models.keys()))
        is_model_loaded = True
    else:
        dataset_choice = "Demo Mode (Synthetic)"
        is_model_loaded = False
        st.sidebar.warning("⚠️ Running in Demo Mode. No trained weights detected.")

    # XAI Status in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Explainability Status")
    
    if is_model_loaded:
        lime_status, shap_status = check_xai_files(dataset_choice)
        
        if lime_status['exists'] and lime_status['files'] > 0:
            st.sidebar.success(f"✅ LIME Active ({lime_status['files']} samples)")
        else:
            st.sidebar.error("❌ LIME Missing")
        
        if shap_status['has_bar']:
            st.sidebar.success("✅ SHAP Active")
        else:
            st.sidebar.error("❌ SHAP Missing")
    else:
        st.sidebar.info("ℹ️ XAI unavailable in Demo Mode")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Live Monitor",
        "📂 Batch Analysis",
        "🧠 XAI Forensics",
        "📊 Performance & Docs"
    ])

    # ========================================================================
    # TAB 1: LIVE MONITOR (SIMULATOR)
    # ========================================================================
    with tab1:
        st.subheader("📡 Real-Time Network Traffic Simulation")
        st.markdown("This module generates synthetic traffic matching your model's features and performs live detection.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Initialize session state for live metrics
        if 'total_packets' not in st.session_state:
            st.session_state.total_packets = 0
            st.session_state.threats = 0
            st.session_state.logs = []
            st.session_state.last_synthetic_df = None
            st.session_state.detection_results = None

        # Step 1: Generate Data
        if st.button("🔄 Generate Synthetic Traffic", type="secondary"):
            # Determine features to generate
            if is_model_loaded:
                feat_list = features[dataset_choice]
                # Use the specific feature list for the chosen model
                synthetic_df = generate_synthetic_traffic(
                    n_samples=20, 
                    feature_names=feat_list, 
                    dataset_type=dataset_choice
                )
            else:
                # In demo mode without models, use CIC features as default generic structure
                feat_list = CIC_2018_FEATURES
                synthetic_df = generate_synthetic_traffic(
                    n_samples=20, 
                    feature_names=feat_list,
                    dataset_type="CIC-IDS-2018"
                )
            
            # Store for viewing
            st.session_state.last_synthetic_df = synthetic_df
            st.session_state.detection_results = None # Reset previous results
            st.success("✅ Synthetic traffic generated. Ready for detection.")
            st.rerun()

        # Step 2: Run Detection on Generated Data
        if st.session_state.last_synthetic_df is not None:
            st.markdown("---")
            col_gen, col_det = st.columns([1, 1])
            with col_gen:
                st.info("Data Ready: Click below to run the CNN-LSTM model on the generated samples.")
            with col_det:
                if st.button("🚀 Run Real-Time Detection", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_log = []
                    
                    synthetic_df = st.session_state.last_synthetic_df
                    
                    # Determine feature list again for processing
                    if is_model_loaded:
                        feat_list = features[dataset_choice]
                    else:
                        feat_list = CIC_2018_FEATURES

                    for i in range(len(synthetic_df)):
                        # Update Progress
                        progress_bar.progress((i + 1) / len(synthetic_df))
                        status_text.text(f"Analyzing Flow {i+1}/{len(synthetic_df)}...")
                        
                        row_data = synthetic_df.iloc[i:i+1]
                        
                        # Perform Prediction if model is loaded
                        if is_model_loaded:
                            try:
                                # Preprocess
                                X = row_data[feat_list].fillna(0).values
                                if dataset_choice in imputers:
                                    X = imputers[dataset_choice].transform(X)
                                if dataset_choice in scalers:
                                    X = scalers[dataset_choice].transform(X)
                                
                                # Predict
                                model = models[dataset_choice]
                                predictions = model.predict(X, verbose=0)
                                
                                if dataset_choice == 'CIC-IDS-2018':
                                    pred_class_idx = np.argmax(predictions[0])
                                    confidence = np.max(predictions[0])
                                    classes_map = class_names[dataset_choice]
                                    pred_label = classes_map[pred_class_idx]
                                    is_threat = pred_label != "Benign" # Adjust based on your actual benign label name
                                else:
                                    # Binary classification
                                    prob = predictions[0][0]
                                    is_threat = prob >= 0.5
                                    confidence = prob if is_threat else (1 - prob)
                                    pred_label = "Attack" if is_threat else "Benign"
                                    
                            except Exception as e:
                                st.error(f"Prediction Error: {e}")
                                is_threat = False
                                confidence = 0.0
                                pred_label = "Error"
                        else:
                            # Random simulation for demo mode
                            is_threat = random.random() > 0.8
                            confidence = random.uniform(0.7, 0.99)
                            pred_label = "Attack (Simulated)" if is_threat else "Benign (Simulated)"

                        # Generate Forensic Narrative
                        narrative = generate_forensic_narrative(row_data.iloc[0], dataset_choice if is_model_loaded else "CIC-IDS-2018")

                        # Update Metrics
                        st.session_state.total_packets += 1
                        if is_threat:
                            st.session_state.threats += 1
                            log_entry = {
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Source IP": f"192.168.1.{random.randint(1,255)}",
                                "Type": pred_label,
                                "Confidence": f"{confidence:.2%}",
                                "Status": "🔴 ALERT",
                                "Explanation": narrative
                            }
                        else:
                            log_entry = {
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Source IP": f"10.0.0.{random.randint(1,255)}",
                                "Type": pred_label,
                                "Confidence": f"{confidence:.2%}",
                                "Status": "🟢 SAFE",
                                "Explanation": "Traffic patterns within normal baseline parameters."
                            }
                        
                        results_log.append(log_entry)
                        time.sleep(0.05) # Simulate processing time
                    
                    # Update Session State with new logs
                    st.session_state.logs = results_log[:20] + st.session_state.logs
                    st.session_state.logs = st.session_state.logs[:20] # Keep last 20
                    
                    status_text.text("✅ Detection Complete")
                    progress_bar.empty()
                    st.success("✅ Real-time detection completed. See logs below.")

        # Display Metrics
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.total_packets}</div>
                    <div class="metric-label">Total Flows</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            threat_color = "#ff0055" if st.session_state.threats > 0 else "#45a29e"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{threat_color}">{st.session_state.threats}</div>
                    <div class="metric-label">Threats Detected</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            benign = st.session_state.total_packets - st.session_state.threats
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{benign}</div>
                    <div class="metric-label">Benign Flows</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            risk_score = (st.session_state.threats / max(1, st.session_state.total_packets)) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{risk_score:.1f}%</div>
                    <div class="metric-label">Network Risk Score</div>
                </div>
            """, unsafe_allow_html=True)

        # Live Log Table with Explanation
        st.markdown("### 📜 Recent Activity Log & Forensic Narratives")
        if st.session_state.logs:
            log_df = pd.DataFrame(st.session_state.logs)
            # Display only key columns in main table
            display_cols = ["Time", "Source IP", "Type", "Confidence", "Status"]
            st.dataframe(log_df[display_cols], use_container_width=True, hide_index=True)
            
            # Show Explanation for the most recent threat if available
            latest_threats = log_df[log_df['Status'] == "🔴 ALERT"]
            if not latest_threats.empty:
                latest = latest_threats.iloc[0]
                st.markdown(f"""
                    <div class="explanation-box">
                        <h4 style="color:#ff0055; margin-bottom:0.5rem;">🚨 Latest Threat Analysis</h4>
                        <p><strong>Time:</strong> {latest['Time']} | <strong>Source:</strong> {latest['Source IP']}</p>
                        <p><strong>Forensic Narrative:</strong> {latest['Explanation']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click 'Generate Synthetic Traffic' then 'Run Real-Time Detection' to begin monitoring.")

        # NEW: View Synthetic Data Button
        if st.session_state.last_synthetic_df is not None:
            st.markdown("---")
            with st.expander("🔍 View Generated Synthetic Data Sample"):
                st.write("The following data was generated to match the model's expected features:")
                st.dataframe(st.session_state.last_synthetic_df.head(10), use_container_width=True)
                st.caption(f"Total Features Generated: {len(st.session_state.last_synthetic_df.columns)}")

    # ========================================================================
    # TAB 2: BATCH ANALYSIS (FILE UPLOAD)
    # ========================================================================
    with tab2:
        st.subheader("📂 Batch Traffic Analysis")
        
        uploaded_file = st.file_uploader("Upload Network Flow CSV File", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {len(df)} network flows")
            
            if is_model_loaded:
                feature_list = features[dataset_choice]
                missing_features = [f for f in feature_list if f not in df.columns]
                
                if missing_features:
                    st.error(f"⚠️ Missing {len(missing_features)} required features. Please ensure the CSV matches the training dataset structure.")
                    with st.expander("View Missing Features"):
                        st.write(missing_features)
                else:
                    if st.button("🔍 Analyze Batch", type="primary"):
                        with st.spinner("Processing Data Through CNN-LSTM..."):
                            # Preprocessing
                            X = df[feature_list].fillna(0).values
                            if dataset_choice in imputers:
                                X = imputers[dataset_choice].transform(X)
                            if dataset_choice in scalers:
                                X = scalers[dataset_choice].transform(X)
                            
                            # Prediction
                            model = models[dataset_choice]
                            predictions = model.predict(X, verbose=0)
                           
                            if dataset_choice == 'CIC-IDS-2018':
                                pred_classes = np.argmax(predictions, axis=1)
                                pred_proba = np.max(predictions, axis=1)
                                classes_map = class_names[dataset_choice]
                                df['prediction'] = [classes_map[p] for p in pred_classes]
                            else:
                                pred_classes = (predictions.flatten() >= 0.5).astype(int)
                                pred_proba = predictions.flatten()
                                df['prediction'] = ["Attack" if p==1 else "Benign" for p in pred_classes]
                            
                            df['confidence'] = pred_proba
                            
                            # Visualization
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_pie = px.pie(df, names='prediction', title='Traffic Classification Distribution',
                                                 color_discrete_sequence=['#45a29e', '#ff0055'])
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                fig_hist = px.histogram(df, x='confidence', color='prediction',
                                                        title='Prediction Confidence Distribution',
                                                        color_discrete_map={'Attack': '#ff0055', 'Benign': '#45a29e'})
                                st.plotly_chart(fig_hist, use_container_width=True)
                                
                            st.dataframe(df[['prediction', 'confidence'] + feature_list[:5]], use_container_width=True)
            else:
                st.warning("Model not loaded. Cannot perform real analysis on uploaded file in Demo Mode.")

    # ========================================================================
    # TAB 3: XAI FORENSICS
    # ========================================================================
    with tab3:
        st.subheader("🧠 Explainable AI (LIME + SHAP)")
        
        if not is_model_loaded:
            st.info("XAI explanations are only available when trained models are loaded.")
        else:
            xai_available = True # Simplified for display logic
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 LIME Local Explanations")
                st.caption("Explains individual predictions (Why was this flagged?)")
                
                lime_dir = os.path.join('XAI_results', f'lime_{dataset_choice}')
                if os.path.exists(lime_dir):
                    lime_files = [f for f in os.listdir(lime_dir) if f.endswith('.png') and 'summary' not in f]
                    if lime_files:
                        selected_sample = st.selectbox("Select Sample ID", lime_files)
                        st.image(os.path.join(lime_dir, selected_sample), use_container_width=True)
                    else:
                        st.warning("No LIME visualization files found.")
                else:
                    st.warning("LIME directory not found.")
            
            with col2:
                st.markdown("#### 📊 SHAP Global Feature Importance")
                st.caption("Identifies most influential features across the dataset")
                
                shap_dir = os.path.join('XAI_results', f'shap_{dataset_choice}')
                if os.path.exists(shap_dir):
                    if os.path.exists(os.path.join(shap_dir, 'shap_bar_plot.png')):
                        st.image(os.path.join(shap_dir, 'shap_bar_plot.png'), use_container_width=True)
                    
                    if os.path.exists(os.path.join(shap_dir, 'feature_importance.json')):
                        with open(os.path.join(shap_dir, 'feature_importance.json'), 'r') as f:
                            importance = json.load(f)
                        st.markdown("**Top 5 Critical Features:**")
                        for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
                            st.progress(score)
                            st.caption(f"{i}. {feature}")
                else:
                    st.warning("SHAP directory not found.")

    # ========================================================================
    # TAB 4: PERFORMANCE & DOCS
    # ========================================================================
    with tab4:
        st.subheader("📊 Model Performance & Research Documentation")
        
        if os.path.exists('performance_comparison.csv'):
            perf_df = pd.read_csv('performance_comparison.csv')
            st.dataframe(perf_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Research Objectives Status")
        
        obj_status = [
            ("✅ Objective i: Lightweight CNN-LSTM Architecture", True),
            ("✅ Objective ii: Dual XAI Integration (LIME + SHAP)", True if is_model_loaded else False),
            ("✅ Objective iii: SMOTE Data Balancing", True),
            ("✅ Objective iv: Interactive Streamlit Dashboard", True),
            ("✅ Objective v: Comprehensive Evaluation Metrics", True)
        ]
        
        for status, achieved in obj_status:
            if achieved:
                st.success(status)
            else:
                st.error(status)

# =============================================================================
# FOOTER
# =============================================================================
def render_footer():
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; color: #45a29e; font-size: 0.9rem;">
            <p><strong>NDA CyberShield IDS</strong> | Developed by Samuel Ayorinde A</p>
            <p>PGD Cybersecurity | Nigerian Defence Academy (NDA) | Supervised by Mr Victor Akuboh</p>
            <p>&copy; {datetime.now().year} All Rights Reserved.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    render_footer()