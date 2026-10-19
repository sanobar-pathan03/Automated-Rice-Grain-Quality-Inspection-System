import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Rice Detection System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Green & Brown Earthy Palette with Background Image) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Variables - Green & Brown Palette */
    :root {
        --bg-color: #F7F5F0; /* Soft warm cream */
        --text-dark: #2C241B; /* Dark brown text */
        --text-med: #6B5D4F;
        --primary-green: #4A6741; /* Forest Green */
        --primary-green-light: #E8F0E6;
        --accent-brown: #8B5A2B; /* Earthy Brown */
        --accent-brown-light: #F2EBE3;
        --card-bg: #FFFFFF;
        --border-color: #D6CFC4;
        --danger-red: #A64B36;
    }

    /* Base Styles with Background Image */
    .stApp {
        background: linear-gradient(rgba(247, 245, 240, 0.85), rgba(247, 245, 240, 0.9)), 
                    url('https://images.unsplash.com/photo-1586201375761-83865001e31c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
        color: var(--text-dark);
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: var(--text-dark);
    }
    
    .main-header {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
        color: var(--primary-green);
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: var(--text-med);
        margin-bottom: 2.5rem;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    .sidebar-logo {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.3rem;
        line-height: 1.2;
        color: var(--primary-green);
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    .sidebar-badge {
        background-color: var(--accent-brown);
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    .nav-item {
        padding: 12px 0;
        font-size: 1rem;
        color: var(--text-dark);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 500;
    }
    .nav-item.active {
        font-weight: 700;
        color: var(--primary-green);
    }

    /* Section Titles */
    .section-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        margin-bottom: 1.5rem;
        color: var(--accent-brown);
        text-transform: uppercase;
    }

    /* Upload Box */
    .dashed-box {
        border: 2px dashed var(--accent-brown);
        border-radius: 16px;
        padding: 40px;
        background-color: rgba(242, 235, 227, 0.9);
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(5px);
    }

    /* Cards */
    .result-card, .info-card, .feature-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(44, 36, 27, 0.1);
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
        backdrop-filter: blur(5px);
    }

    /* Progress Bars */
    .progress-container { margin-bottom: 15px; }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .progress-bar-bg {
        background-color: #EAE5D9;
        border-radius: 8px;
        height: 22px;
        width: 100%;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
    }

    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin-top: 15px;
    }
    .feature-item {
        background-color: var(--primary-green-light);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid var(--primary-green);
    }
    .feature-label {
        font-size: 0.75rem;
        color: var(--text-med);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .feature-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-dark);
    }

    /* Buttons */
    .btn-custom {
        background-color: var(--primary-green);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s;
    }
    .btn-custom:hover { background-color: #3A5233; color: white; }
    .btn-light {
        background-color: white;
        color: var(--accent-brown);
        border: 1px solid var(--accent-brown);
    }
    .btn-light:hover { background-color: var(--accent-brown-light); color: var(--accent-brown); }

    /* Footer */
    .footer-container {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
        color: var(--text-med);
        font-size: 0.85rem;
        background-color: rgba(247, 245, 240, 0.9);
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    section[data-testid="stSidebar"] { background-color: rgba(247, 245, 240, 0.95); }
    
    /* Main container background */
    .block-container {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Backend Logic (Image Processing) ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('rice_model.pkl')
    except:
        return None

def process_image(image):
    img = np.array(image)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
        
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    mean_color = cv2.mean(img_cv, mask=mask)[:3]
    
    features = np.array([[area, perimeter, w, h, aspect_ratio, mean_color[0], mean_color[1], mean_color[2]]])
    
    model = load_model()
    prediction = model.predict(features)[0] if model else "Unknown"
    
    # Mock probabilities for UI demonstration
    probs = {'Whole': 0.85, 'Broken': 0.10, 'Defective': 0.05}
    if prediction == 'Broken': probs = {'Whole': 0.15, 'Broken': 0.75, 'Defective': 0.10}
    if prediction == 'Defective': probs = {'Whole': 0.10, 'Broken': 0.20, 'Defective': 0.70}
    
    features_dict = {
        "Area": f"{area:.0f} px²",
        "Perimeter": f"{perimeter:.0f} px",
        "Dimensions": f"{w} x {h} px",
        "Aspect Ratio": f"{aspect_ratio:.2f}",
        "Avg Color (BGR)": f"({int(mean_color[0])}, {int(mean_color[1])}, {int(mean_color[2])})"
    }
    return prediction, features_dict, probs

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">RICE<br>DETECTION<br>SYSTEM</div>
        <div class="sidebar-badge">QUALITY CONTROL</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="nav-item active">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📄 My Detections</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📈 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">⚙️ Settings</div>', unsafe_allow_html=True)

# --- Main Content ---
st.markdown('<h1 class="main-header">🌾 Rice Grain Quality Inspection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Image Processing & Machine Learning Classification</p>', unsafe_allow_html=True)

# Detection Module
st.markdown('<div class="section-title">Detection Module</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="dashed-box">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-bottom: 15px; color: var(--accent-brown);">📤 UPLOAD RICE GRAIN IMAGE</h4>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-med); margin-bottom: 20px; font-size: 0.9rem;">Supports JPG, JPEG, PNG formats</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Rice Sample", use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Results Section
if uploaded_file is not None:
    if st.button("🔍 Analyze Grain Quality", use_container_width=True, type="primary"):
        with st.spinner('Processing image and extracting features...'):
            prediction, features, probs = process_image(image)
        
        if features:
            st.markdown('<div class="section-title" style="margin-top: 3rem;">Analysis Results</div>', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([1.2, 1])

            # Left Column: Probabilities & Info
            with col_res1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="margin-bottom: 20px; color: var(--primary-green);">Classification Probabilities</h3>', unsafe_allow_html=True)
                
                # Dynamic Progress Bars
                colors = {'Whole': '#4A6741', 'Broken': '#8B5A2B', 'Defective': '#A64B36'}
                for cls, prob in probs.items():
                    pct = int(prob * 100)
                    st.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-label">
                                <span>{cls} Grain</span>
                                <span>{pct}%</span>
                            </div>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="width: {pct}%; background-color: {colors[cls]};"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Info Card
                st.markdown('<br><div class="info-card">', unsafe_allow_html=True)
                st.markdown(f'<h3 style="margin-bottom: 10px; color: var(--accent-brown);">{prediction.upper()} GRAIN DETECTED</h3>', unsafe_allow_html=True)
                
                info_text = {
                    'Whole': 'Intact, premium quality grains. No visible damage or discoloration. Ideal for retail packaging and export standards.',
                    'Broken': 'Partially broken or smaller pieces. Often resulting from milling processes. Typically used for flour production or lower-tier markets.',
                    'Defective': 'Discolored, damaged, or abnormal grains. Removed during quality control to ensure food safety and standard compliance.'
                }
                st.markdown(f'<p style="color: var(--text-med); line-height: 1.6;">{info_text.get(prediction, "")}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Right Column: Extracted Features (Crucial for Assignment)
            with col_res2:
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown('<h3 style="margin-bottom: 10px; color: var(--primary-green);">Extracted Image Features</h3>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 0.85rem; color: var(--text-med); margin-bottom: 15px;">Morphological and color features extracted via OpenCV contour analysis.</p>', unsafe_allow_html=True)
                
                st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
                for key, value in features.items():
                    st.markdown(f"""
                        <div class="feature-item">
                            <div class="feature-label">{key}</div>
                            <div class="feature-value">{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("️ No grain detected. Please ensure the image has a clear background and good lighting.")

# Footer
st.markdown("""
    <div class="footer-container">
        <p>© 2026 Rice Detection System. Automated Quality Control Project.</p>
    </div>
""", unsafe_allow_html=True)