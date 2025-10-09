import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time  # ADD THIS LINE - FIXES THE ERROR

# Set page config
st.set_page_config(
    page_title="MedScan Pro - Medical AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.8rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a5276, #2e86ab);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.6rem;
        color: #2e86ab;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1a5276;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Upload area */
    .upload-area {
        border: 3px dashed #1a5276;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2e86ab;
        background: linear-gradient(135deg, #e9ecef, #dee2e6);
    }
    
    /* Risk badges */
    .risk-high { 
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #28a745, #218838);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sidebar-logo {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Progress bars */
    .custom-progress {
        background-color: #e9ecef;
        height: 12px;
        border-radius: 10px;
        margin: 8px 0;
        overflow: hidden;
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Results cards */
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_medical_model():
    try:
        model = load_model('models/final_medical_model.h5')
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

model = load_medical_model()
CLASS_NAMES = ['Fungal Infection', 'Normal', 'Tuberculosis']
CLASS_COLORS = {'Fungal Infection': '#FF6B6B', 'Normal': '#4ECDC4', 'Tuberculosis': '#45B7D1'}

def preprocess_image(uploaded_image):
    """Preprocess uploaded image for model prediction"""
    try:
        img = Image.open(uploaded_image)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to model's expected input
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def analyze_image(uploaded_file, model):
    """Analyze a single image and return results"""
    try:
        # Preprocess image
        img_array, original_img = preprocess_image(uploaded_file)
        
        if img_array is None:
            return None
            
        # Get prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all probabilities
        probabilities = {
            class_name: float(predictions[0][i]) 
            for i, class_name in enumerate(CLASS_NAMES)
        }
        
        # Determine risk level
        if predicted_class == 'Normal':
            risk_level = 'low'
        else:
            risk_level = 'high' if confidence > 0.7 else 'medium'
        
        return {
            'filename': uploaded_file.name,
            'image': original_img,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'risk_level': risk_level,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
    except Exception as e:
        st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
        return None

def create_metric_card(value, label, icon="📊"):
    """Create a professional metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-number">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">🫁</div>
            <h2>MedScan Pro</h2>
            <p><em>AI-Powered Medical Imaging</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Actions")
        if st.button("🔄 Clear All Results", use_container_width=True, type="secondary"):
            st.session_state.analysis_results = []
            st.rerun()
            
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Export feature coming soon!")
            
        st.markdown("---")
        
        st.markdown("### 🔬 Model Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "66.7%")
        with col2:
            st.metric("Classes", "3")
            
        st.info("""
        **Hybrid CNN Architecture**
        - EfficientNet Backbone
        - Custom CNN Head
        - Transfer Learning
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        For medical professionals only. 
        Not for diagnostic use. 
        Always verify with clinical expertise.
        """)

    # --- MAIN CONTENT ---
    
    # Header Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🏥 MedScan Professional</div>', unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #6c757d; margin-bottom: 2rem;'>Advanced AI for Chest X-Ray Analysis</div>", unsafe_allow_html=True)
    
    # Top Metrics Row - IMPROVED
    st.markdown("### 📈 Dashboard Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_images = len(st.session_state.analysis_results)
        st.markdown(create_metric_card(total_images, "Images Analyzed", "🖼️"), unsafe_allow_html=True)
    
    with col2:
        high_risk = len([r for r in st.session_state.analysis_results if r.get('risk_level') == 'high'])
        st.markdown(create_metric_card(high_risk, "High Risk Cases", "⚠️"), unsafe_allow_html=True)
    
    with col3:
        status = "✅ Active" if model else "❌ Offline"
        st.markdown(create_metric_card(status, "AI Status", "🤖"), unsafe_allow_html=True)
    
    with col4:
        if st.session_state.analysis_results:
            avg_conf = np.mean([r['confidence'] for r in st.session_state.analysis_results])
            st.markdown(create_metric_card(f"{avg_conf:.1%}", "Avg Confidence", "🎯"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("N/A", "Avg Confidence", "🎯"), unsafe_allow_html=True)
    
    with col5:
        unique_diagnoses = len(set([r['prediction'] for r in st.session_state.analysis_results])) if st.session_state.analysis_results else 0
        st.markdown(create_metric_card(unique_diagnoses, "Diagnosis Types", "🔍"), unsafe_allow_html=True)
    
    # Upload Section - IMPROVED
    st.markdown("---")
    st.markdown('<div class="sub-header">📤 Upload Chest X-Ray Images</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-area">
        <h3>🚀 Drag & Drop X-Ray Images</h3>
        <p>Upload multiple chest X-rays for AI analysis</p>
        <div style="color: #6c757d; font-size: 0.9rem;">
            <strong>Supported:</strong> JPG, JPEG, PNG • <strong>Max:</strong> 200MB per file
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        " ",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Analysis Button
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Start AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🔄 AI is analyzing images..."):
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Skip if already analyzed
                        if any(r['filename'] == uploaded_file.name for r in st.session_state.analysis_results):
                            continue
                        
                        # Analyze image using your actual model
                        result = analyze_image(uploaded_file, model)
                        
                        if result:
                            st.session_state.analysis_results.append(result)
                        
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                        # Small delay to show progress (optional)
                        time.sleep(0.1)
                    
                    st.success(f"✅ Analysis complete! Processed {len(uploaded_files)} images.")
    
    # Results Section - IMPROVED
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)
        
        # Summary Statistics
        st.markdown("### 📊 Quick Summary")
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            diagnoses = [r['prediction'] for r in st.session_state.analysis_results]
            most_common = max(set(diagnoses), key=diagnoses.count) if diagnoses else "N/A"
            st.metric("Most Common", most_common)
        
        with summary_cols[1]:
            high_conf = len([r for r in st.session_state.analysis_results if r['confidence'] > 0.7])
            st.metric("High Confidence", high_conf)
        
        with summary_cols[2]:
            recent = len([r for r in st.session_state.analysis_results])
            st.metric("Recent Analyses", recent)
        
        with summary_cols[3]:
            if st.session_state.analysis_results:
                latest = st.session_state.analysis_results[-1]['timestamp']
                st.metric("Last Analysis", latest)
        
        # Individual Results
        st.markdown("### 📋 Detailed Results")
        for i, result in enumerate(st.session_state.analysis_results):
            with st.expander(f"📄 {result['filename']} • {result['prediction']} • {result['confidence']:.1%}", expanded=i==0):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display image if available
                    if 'image' in result:
                        st.image(result['image'], caption=result['filename'], use_container_width=True)
                    
                    # Display risk badge
                    risk_class = f"risk-{result['risk_level']}"
                    st.markdown(f'<div class="{risk_class}">🩺 {result["risk_level"].upper()} RISK</div>', unsafe_allow_html=True)
                    
                    # Confidence gauge
                    st.metric("AI Confidence", f"{result['confidence']:.1%}")
                    
                    # Quick stats
                    st.write(f"**File:** {result['filename']}")
                    st.write(f"**Time:** {result['timestamp']}")
                
                with col2:
                    # Diagnosis details
                    st.subheader(f"Diagnosis: {result['prediction']}")
                    
                    # Probability distribution
                    st.markdown("**Probability Distribution:**")
                    for condition, prob in result['probabilities'].items():
                        percentage = prob * 100
                        color = CLASS_COLORS.get(condition, '#6c757d')
                        st.write(f"**{condition}:** {percentage:.1f}%")
                        st.progress(float(prob))
                    
                    # Recommendations
                    st.markdown("**🎯 Recommended Actions:**")
                    if result['prediction'] == 'Tuberculosis':
                        st.error("""
                        - Infectious disease consultation
                        - Sputum culture & PCR testing
                        - Chest CT scan
                        - Public health notification
                        """)
                    elif result['prediction'] == 'Fungal Infection':
                        st.warning("""
                        - Pulmonology consultation  
                        - Fungal serology testing
                        - Sputum culture for fungi
                        - Immune status evaluation
                        """)
                    else:
                        st.success("""
                        - No significant abnormalities detected
                        - Routine follow-up recommended
                        - Consult if symptoms develop
                        """)
    
    else:
        # Empty state - IMPROVED
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h3>🎯 Ready to Analyze</h3>
                <p style='color: #6c757d;'>Upload chest X-ray images to begin AI analysis</p>
                <div style='font-size: 4rem; margin: 2rem 0;'>🫁</div>
                <p><strong>Supported Conditions:</strong></p>
                <div style='display: inline-block; text-align: left;'>
                    • Tuberculosis Detection<br>
                    • Fungal Infection Analysis<br>
                    • Normal vs Abnormal Classification
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()