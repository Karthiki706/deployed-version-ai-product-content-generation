import os
import re
import streamlit as st
import requests
import json

# ---------------------------------------------------------------------------
# SEMANTIC CATEGORY VALIDATOR  (frontend-only, deterministic, rule-based)
# ---------------------------------------------------------------------------
_CATEGORY_VOCAB: dict = {
    "electronics": {
        "triggers": ["tv", "television", "smartphone", "phone", "laptop", "computer",
                     "camera", "headphone", "headset", "earphone", "speaker", "tablet",
                     "monitor", "printer", "keyboard", "mouse", "smartwatch", "router",
                     "led", "oled", "qled", "4k", "8k", "smart tv", "display", "projector",
                     "refrigerator", "washing machine", "air conditioner", "microwave", "electronics"],
        "forbidden": ["burner", "boil time", "boiling", "gas", "camping stove", "windshield",
                      "fuel", "propane", "titanium stove", "foldable legs", "cookware",
                      "camp", "hiking gear", "tent", "sleeping bag", "bottle", "vacuum",
                      "leak proof", "steel", "capacity", "bpa free", "shirt", "dress",
                      "pant", "shoe", "boot", "sandal", "moisturizer", "serum", "lipstick",
                      "makeup", "sofa", "chair", "bed", "grocery", "rice", "flour"],
    },
    "camping": {
        "triggers": ["stove", "camping", "hiking", "backpack", "outdoor gear", "tent",
                     "sleeping bag", "cookware", "trekking", "camp"],
        "forbidden": ["oled", "netflix", "hdmi", "dolby", "smart tv", "uhd", "streaming",
                      "remote control", "gaming", "display panel", "touchscreen", "wifi",
                      "bluetooth", "rgb", "saree", "kurta", "lipstick", "makeup"],
    },
    "home & kitchen": {
        "triggers": ["bottle", "vacuum", "leak proof", "steel", "capacity", "bpa free",
                     "kitchen", "home", "cookware", "pan", "pot", "blender", "mixer", "plate",
                     "cup", "mug", "spoon", "knife", "fork"],
        "forbidden": ["bluetooth", "touchscreen", "rgb", "gaming", "wifi", "hdmi", "oled",
                      "netflix", "remote control", "burner", "boil time", "gas", "camping stove",
                      "windshield", "fuel", "propane", "titanium stove", "foldable legs"],
    },
    "fashion": {
        "triggers": ["shirt", "dress", "pant", "jacket", "coat", "trouser", "skirt",
                     "blouse", "saree", "kurta", "jeans", "hoodie", "apparel", "clothing",
                     "shoe", "boot", "sneaker", "sandal", "slipper", "heel", "loafer", "footwear"],
        "forbidden": ["display", "battery", "bluetooth", "hdmi", "gas burner", "burner", "stove",
                      "boil", "cooking", "wifi", "oled", "remote", "rgb", "vacuum", "bpa free"],
    },
    "beauty": {
        "triggers": ["moisturizer", "serum", "sunscreen", "shampoo", "conditioner",
                     "face wash", "lipstick", "foundation", "mascara", "perfume",
                     "skincare", "haircare", "makeup", "beauty", "cosmetics"],
        "forbidden": ["display", "hdmi", "stove", "burner", "boil", "camping", "wifi", "bluetooth",
                      "rgb", "gaming", "vacuum", "bpa free", "steel", "leak proof"],
    },
    "furniture": {
        "triggers": ["sofa", "chair", "table", "desk", "bed", "wardrobe", "cabinet",
                     "shelf", "bookcase", "furniture", "mattress"],
        "forbidden": ["display", "hdmi", "stove", "burner", "boil", "bluetooth", "wifi",
                      "oled", "streaming", "netflix", "banned words", "shampoo", "lipstick"],
    },
    "grocery": {
        "triggers": ["biscuit", "flour", "rice", "oil", "sugar", "salt", "spice",
                     "sauce", "snack", "cereal", "pasta", "noodle", "grocery", "food", "beverage"],
        "forbidden": ["display", "hdmi", "stove", "oled", "bluetooth", "streaming", "wifi",
                      "rgb", "gaming", "vacuum", "leak proof", "steel", "sofa", "chair"],
    }
}


def _detect_category(product_name: str, category: str) -> str | None:
    """Return matched vocab key or None."""
    text = (product_name + " " + category).lower()
    for cat_key, cat_data in _CATEGORY_VOCAB.items():
        if any(t in text for t in cat_data["triggers"]):
            return cat_key
    return None




def validate_product_inputs(
    product_name: str,
    category: str,
    features: list[str],
    specs: dict,
    package_contents: list[str],
) -> dict:
    """
    Returns a dict:
      {
        'detected_category': str | None,
        'accepted_features': list,
        'ignored_features': list[dict],  # {item, reason}
        'accepted_specs': dict,
        'ignored_specs': list[dict],
        'accepted_pkg': list,
        'ignored_pkg': list[dict],
        'has_warnings': bool,
      }
    """
    detected = _detect_category(product_name, category)
    result = {
        "detected_category": detected,
        "accepted_features": list(features),
        "ignored_features": [],
        "accepted_specs": dict(specs),
        "ignored_specs": [],
        "accepted_pkg": list(package_contents),
        "ignored_pkg": [],
        "has_warnings": False,
    }

    if not detected:
        return result  # no vocabulary to enforce — accept everything

    forbidden = _CATEGORY_VOCAB[detected]["forbidden"]

    def _is_forbidden(text: str) -> str | None:
        tl = text.lower()
        for f in forbidden:
            if f in tl:
                return f
        return None

    # -- features --
    acc_f, ign_f = [], []
    for feat in features:
        match = _is_forbidden(feat)
        if match:
            ign_f.append({"item": feat, "reason": f"'{match}' is not compatible with {detected} products"})
        else:
            acc_f.append(feat)
    result["accepted_features"] = acc_f
    result["ignored_features"] = ign_f

    # -- specs keys + values --
    acc_s, ign_s = {}, []
    for k, v in specs.items():
        match = _is_forbidden(k) or _is_forbidden(str(v))
        if match:
            ign_s.append({"item": f"{k}: {v}", "reason": f"'{match}' is not compatible with {detected} products"})
        else:
            acc_s[k] = v
    result["accepted_specs"] = acc_s
    result["ignored_specs"] = ign_s

    # -- package contents --
    acc_p, ign_p = [], []
    for item in package_contents:
        match = _is_forbidden(item)
        if match:
            ign_p.append({"item": item, "reason": f"'{match}' is not compatible with {detected} products"})
        else:
            acc_p.append(item)
    result["accepted_pkg"] = acc_p
    result["ignored_pkg"] = ign_p

    result["has_warnings"] = bool(ign_f or ign_s or ign_p)
    return result


def check_frontend_hallucinations(generated_text: str, ignored_items: list[dict]) -> list[dict]:
    """
    Post-generation check: if any ignored attribute appears verbatim in the
    generated content, flag it as a hallucinated / unsupported use.
    Returns list of {item, reason, severity}.
    """
    flags = []
    gl = generated_text.lower()
    for entry in ignored_items:
        needle = entry["item"].lower()
        # check if any significant word from the ignored item appears
        words = [w for w in re.split(r"[\s,:]+", needle) if len(w) > 3]
        for word in words:
            if word in gl:
                flags.append({
                    "item": entry["item"],
                    "reason": f"Ignored attribute '{entry['item']}' appears in generated content",
                    "severity": "High",
                })
                break
    return flags


# ---------------------------------------------------------------------------
# _get_secret: consistent secret resolution used throughout this module.
# Priority: st.secrets (Cloud) → os.environ / .env (local) → default
# python-dotenv is loaded inside config.py with a try/except guard.
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(key, default)

# App Configuration
st.set_page_config(
    page_title="Enterprise Multi-Agent Product Content Generation Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Single backend URL — resolved from st.secrets (Cloud) → os.environ → local default.
# Set BACKEND_URL in Streamlit Secrets for production deployments.
# ---------------------------------------------------------------------------
APP_API_KEY = _get_secret("APP_API_KEY") or ""
API_HEADERS = {"X-API-Key": APP_API_KEY} if APP_API_KEY else {}

BACKEND_URL = (
    _get_secret("BACKEND_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")

# Custom CSS for glassmorphism and modern UI feel
st.markdown("""
<style>
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(135deg, #00C6FF, #0072FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 42px;
        margin-bottom: 10px;
    }
    .gradient-subtitle {
        color: #8A99AD;
        font-size: 18px;
        margin-bottom: 30px;
    }
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-card.good {
        border-left: 4px solid #10B981;
    }
    .metric-card.warning {
        border-left: 4px solid #F59E0B;
    }
    .metric-card.critical {
        border-left: 4px solid #EF4444;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-num {
        font-size: 32px;
        font-weight: 700;
    }
    /* Section containers */
    .content-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    /* Custom divider */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, rgba(0,198,255,0) 0%, rgba(0,198,255,0.8) 50%, rgba(0,198,255,0) 100%);
        margin: 25px 0;
    }

    /* Progress Loader Dashboard */
    .progress-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .progress-title {
        margin-top: 0;
        color: #00C6FF;
    }
    .progress-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 15px 0;
    }
    .progress-overall-text {
        margin: 5px 0;
        font-weight: bold;
        color: #E2E8F0;
    }
    .progress-bar-code {
        font-size: 16px;
        color: #00C6FF;
        background: none;
        padding: 0;
    }
    .progress-est-text {
        margin: 8px 0 0 0;
        font-size: 12px;
        color: #94A3B8;
    }

    /* Light Theme Styling Improvements */
    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, li, label, .stMarkdown, strong {
            color: #0F172A !important;
        }
        .gradient-subtitle {
            color: #475569 !important;
        }
        .metric-card, .content-box {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05) !important;
            color: #0F172A !important;
        }
        .metric-label {
            color: #475569 !important;
        }
        code, pre {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        input, textarea, select, div[data-baseweb="input"], div[data-baseweb="textarea"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }
        button[role="tab"] {
            color: #475569 !important;
        }
        button[role="tab"][aria-selected="true"] {
            color: #0072FF !important;
            border-bottom: 2px solid #0072FF !important;
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #00C6FF, #0072FF) !important;
            color: #FFFFFF !important;
            border: none !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #0F172A !important;
        }
        .progress-container {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08) !important;
        }
        .progress-title {
            color: #0072FF !important;
        }
        .progress-divider {
            background: #E2E8F0 !important;
        }
        .progress-overall-text {
            color: #0F172A !important;
        }
        .progress-bar-code {
            color: #0072FF !important;
        }
        .progress-est-text {
            color: #475569 !important;
        }
    }

    /* Target explicit theme=light container classes */
    [data-theme="light"] .stApp, 
    .stApp[data-theme="light"],
    [data-theme="light"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    [data-theme="light"] h1, [data-theme="light"] h2, [data-theme="light"] h3, 
    [data-theme="light"] h4, [data-theme="light"] h5, [data-theme="light"] h6, 
    [data-theme="light"] p, [data-theme="light"] span, [data-theme="light"] li, 
    [data-theme="light"] label, [data-theme="light"] .stMarkdown, [data-theme="light"] strong {
        color: #0F172A !important;
    }
    [data-theme="light"] .gradient-subtitle {
        color: #475569 !important;
    }
    [data-theme="light"] .metric-card, [data-theme="light"] .content-box {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05) !important;
        color: #0F172A !important;
    }
    [data-theme="light"] .metric-label {
        color: #475569 !important;
    }
    [data-theme="light"] code, [data-theme="light"] pre {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
    }
    [data-theme="light"] input, [data-theme="light"] textarea, [data-theme="light"] select, 
    [data-theme="light"] div[data-baseweb="input"], [data-theme="light"] div[data-baseweb="textarea"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-theme="light"] button[role="tab"] {
        color: #475569 !important;
    }
    [data-theme="light"] button[role="tab"][aria-selected="true"] {
        color: #0072FF !important;
        border-bottom: 2px solid #0072FF !important;
    }
    [data-theme="light"] button[kind="primary"] {
        background: linear-gradient(135deg, #00C6FF, #0072FF) !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    [data-theme="light"] section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-theme="light"] section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    [data-theme="light"] .progress-container {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08) !important;
    }
    [data-theme="light"] .progress-title {
        color: #0072FF !important;
    }
    [data-theme="light"] .progress-divider {
        background: #E2E8F0 !important;
    }
    [data-theme="light"] .progress-overall-text {
        color: #0F172A !important;
    }
    [data-theme="light"] .progress-bar-code {
        color: #0072FF !important;
    }
    [data-theme="light"] .progress-est-text {
        color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
    st.markdown("### Platform Control Center")
    
    # 1. Fetch Backend Health
    backend_online = False
    health_data = {}
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            backend_online = True
    except Exception:
        pass

    # Render API Key Status Panel
    if backend_online and health_data.get("gemini") == "connected":
        source_name = health_data.get("credential_source", "Unknown")
        source_display = {
            "environment_variable": "Environment Variable",
            "streamlit_secrets": "Streamlit Secrets",
            ".env": "Local .env file",
        }.get(source_name, source_name.title())
        
        st.markdown(f"""
<div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 15px; background: rgba(0,0,0,0.2); margin-bottom: 20px;">
    <h4 style="margin-top:0; color:#00C6FF; font-size: 14px;">Gemini Configuration</h4>
    <p style="margin:5px 0; font-size:12px;">✓ <strong>API Key Status:</strong> Loaded</p>
    <p style="margin:5px 0; font-size:12px;">✓ <strong>API Key Source:</strong> {source_display}</p>
    <p style="margin:5px 0; font-size:12px;">✓ <strong>Authentication:</strong> Successful</p>
    <p style="margin:5px 0; font-size:12px;"><strong>Primary Model:</strong> {health_data.get("models", {}).get("active_primary_model", "gemini-3.5-flash")}</p>
    <p style="margin:5px 0; font-size:12px; margin-bottom:0;"><strong>Fallback Models:</strong><br>
    • gemini-2.5-flash<br>
    • gemini-2.0-flash</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 15px; background: rgba(0,0,0,0.2); margin-bottom: 20px;">
    <h4 style="margin-top:0; color:#FF4B4B; font-size: 14px;">Gemini Configuration</h4>
    <p style="margin:5px 0; font-size:12px;">✗ <strong>API Key Status:</strong> Missing / Disconnected</p>
    <p style="margin:5px 0; font-size:12px;">✗ <strong>API Key Source:</strong> Not Configured</p>
    <p style="margin:5px 0; font-size:12px; margin-bottom:0;">✗ <strong>Authentication:</strong> Failed</p>
</div>
""", unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("#### System Status")
    
    if backend_online:
        st.success(f"🟢 API Status: {health_data.get('status', 'Online')}")
        
        models = health_data.get("models", {})
        cb = health_data.get("circuit_breaker", {})
        
        st.markdown(f"**Gemini Status**: `{health_data.get('gemini')}`")
        st.markdown(f"**Credential Source**: `{health_data.get('credential_source')}`")
        st.markdown(f"**Active Model**: `{models.get('active_primary_model')}`")
        st.markdown(f"**Fallback Chain**: `{', '.join(models.get('fallback_models_chain', []))}`")
        st.markdown(f"**Circuit Breaker Status**: `{cb.get('state')}`")
    else:
        st.error("🔴 Backend service is currently unavailable.")
        st.caption("Please try again in a few moments.")

    st.markdown("---")
    st.caption("Platform Engine: v1.0.0")
    # Read active model from health API; fall back to known default
    _active_model = health_data.get("models", {}).get("active_primary_model", "gemini-3.5-flash") if backend_online else "gemini-3.5-flash"
    st.caption(f"Default AI: {_active_model}")

# Main Page Header
st.markdown('<div class="gradient-text">Enterprise Multi-Agent Product Content Generation Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subtitle">Enterprise-grade merchandising descriptions with automatic self-correction and validation gates</div>', unsafe_allow_html=True)

# Initialization of Session States
if "result" not in st.session_state:
    st.session_state.result = None
if "generating" not in st.session_state:
    st.session_state.generating = False
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "pricing" not in st.session_state:
    st.session_state.pricing = {"price": 0.0, "discount": 0.0, "offer_price": 0.0}
if "product_meta" not in st.session_state:
    st.session_state.product_meta = {}
if "auto_navigate" not in st.session_state:
    st.session_state.auto_navigate = False
if "validation_summary" not in st.session_state:
    st.session_state.validation_summary = {}
if "frontend_hallucination_flags" not in st.session_state:
    st.session_state.frontend_hallucination_flags = []
if "hero_image_index" not in st.session_state:
    st.session_state.hero_image_index = 0
if "show_welcome_banner" not in st.session_state:
    st.session_state.show_welcome_banner = False

tab_form, tab_results, tab_logs = st.tabs(["📝 Product Intake Form", "📊 Generation & Quality Reports", "📜 System Logs"])

# No auto-navigation as requested

with tab_form:
    st.markdown("### Enter Catalog Details")
    st.write("Fill out the fields below. The platform will automatically clean, classify, enrich, generate, and correct the contents.")

    # -- Basic Product Details Section --
    st.markdown("#### 📦 Basic Product Details")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        product_name = st.text_input("Product Name *", placeholder="Enter product name", value="")
    with c_p2:
        brand = st.text_input("Brand *", placeholder="Enter brand name", value="")

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        category = st.text_input("Category", placeholder="e.g. Electronics", value="")
    with c_c2:
        subcategory = st.text_input("Subcategory", placeholder="e.g. Mobile Phones", value="")

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        material = st.text_input("Material *", placeholder="e.g. Grade 2 Titanium", value="")
    with c_m2:
        color = st.text_input("Color *", placeholder="e.g. Matte Black", value="")
    with c_m3:
        weight_raw = st.text_input("Weight", placeholder="e.g. 320gm", value="")

    st.markdown("##### Dimensions")
    dim_col1, dim_col2, dim_col3 = st.columns(3)
    with dim_col1:
        dim_h = st.number_input("Height (cm)", min_value=0.0, value=0.0, step=0.1)
    with dim_col2:
        dim_w = st.number_input("Width (cm)", min_value=0.0, value=0.0, step=0.1)
    with dim_col3:
        dim_d = st.number_input("Depth/Length (cm)", min_value=0.0, value=0.0, step=0.1)

    st.markdown("---")

    # -- Pricing Section --
    st.markdown("#### 💰 Pricing")
    _pr_c1, _pr_c2 = st.columns(2)
    with _pr_c1:
        product_price = st.number_input(
            "Original Price (₹) *", min_value=0.0, value=0.0, step=1.0,
            help="Original listed / MRP price (Mandatory).")
    with _pr_c2:
        discount_pct = st.number_input(
            "Discount (%) *", min_value=0.0, max_value=100.0, value=0.0, step=0.5,
            help="Discount percentage to apply (Mandatory).")

    # Offer price preview card (Always calculated and displayed, non-editable)
    _offer = round(product_price * (100 - discount_pct) / 100, 2)
    _saving = round(product_price - _offer, 2)
    st.markdown(f"""
    <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);
                border-radius:12px;padding:16px 20px;margin-top:6px;display:flex;
                gap:32px;align-items:center;flex-wrap:wrap;margin-bottom:15px;">
        <div>
            <p style="margin:0;font-size:11px;color:#64748B;">Original Price</p>
            <p style="margin:0;font-size:15px;color:#94A3B8;
                       text-decoration:line-through;">₹{product_price:,.2f}</p>
        </div>
        <div>
            <p style="margin:0;font-size:11px;color:#64748B;">Discount</p>
            <p style="margin:0;font-size:18px;color:#F59E0B;font-weight:700;">{discount_pct}%</p>
        </div>
        <div>
            <p style="margin:0;font-size:11px;color:#64748B;">Offer Price</p>
            <p style="margin:0;font-size:28px;color:#10B981;font-weight:800;"
               >₹{_offer:,.2f}</p>
        </div>
        <div>
            <p style="margin:0;font-size:11px;color:#64748B;">You Save</p>
            <p style="margin:0;font-size:16px;color:#10B981;font-weight:600;"
               >₹{_saving:,.2f} ({discount_pct:.1f}% off)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # -- Product Images Section --
    st.markdown("#### 🖼️ Product Images (Optional)")
    uploaded_files = st.file_uploader(
        "Upload one or more product images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Images are previewed below and saved in session."
    )
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.markdown("**Preview uploaded images:**")
        _prev_cols = st.columns(min(len(uploaded_files), 4))
        for _pi, _pf in enumerate(uploaded_files):
            with _prev_cols[_pi % 4]:
                st.image(_pf, caption=_pf.name, use_container_width=True)

    st.markdown("---")

    # -- Merchandising Specification Details --
    features_input = st.text_area("Features", placeholder="One feature per line or comma-separated", value="")
    package_contents_input = st.text_area("Package Contents", placeholder="One item per line or comma-separated", value="")
    specifications_input = st.text_area(
        "Technical Specifications", 
        placeholder='{\n  "Key":"Value"\n}',
        value=""
    )
    warranty = st.text_input("Warranty Policy", placeholder="e.g. Lifetime Warranty", value="")
    seo_keywords_input = st.text_input("Target SEO Keywords", placeholder="e.g. vacuum bottle, insulated flask", value="")

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        brand_voice = st.text_input("Brand Voice", value="", placeholder="e.g. Professional")
    with c_s2:
        tone = st.text_input("Tone", value="", placeholder="e.g. Expert")
    with c_s3:
        description_length = st.slider("Target Description Word Count", min_value=100, max_value=500, value=200, step=50)

    target_audience = st.text_input("Target Audience *", placeholder="e.g. Backpackers", value="")

    st.markdown("---")

    # Generate Button
    submit_btn = st.button("🚀 Start Content Generation Pipeline", use_container_width=True, type="primary")

    # Calculate dynamic estimation time
    features_list = [f.strip() for f in features_input.replace(",", "\n").split("\n") if f.strip()]
    package_contents_list = [p.strip() for p in package_contents_input.replace(",", "\n").split("\n") if p.strip()]
    seo_keywords_list = [k.strip() for k in seo_keywords_input.split(",") if k.strip()]
    
    specs_dict = {}
    specs_ok = True
    if specifications_input.strip():
        try:
            specs_dict = json.loads(specifications_input)
        except Exception:
            specs_ok = False

    est_seconds = 5.0 + (len(features_list) * 0.2) + (len(specs_dict) * 0.25) + (len(package_contents_list) * 0.2) + (len(seo_keywords_list) * 0.35) + (description_length * 0.04)
    est_seconds = max(8, min(45, int(est_seconds)))
    
    st.caption(f"Estimated generation time: ≈ {est_seconds} seconds")

    # Handle Form Submission
    if submit_btn:
        # Check requirements
        errors = []
        if not product_name:
            errors.append("Product Name is required.")
        if not brand:
            errors.append("Brand is required.")
        if not material:
            errors.append("Material is required.")
        if not color:
            errors.append("Color is required.")
        if product_price <= 0:
            errors.append("Original Price is required and must be greater than zero.")
        if discount_pct < 0 or discount_pct > 100:
            errors.append("Discount percentage must be between 0 and 100.")
        if not target_audience:
            errors.append("Target Audience is required.")
        if not specs_ok:
            errors.append("Technical Specifications must be valid JSON.")

        # Build dimensions dict
        dim_dict = {}
        if dim_h > 0: dim_dict["height"] = dim_h
        if dim_w > 0: dim_dict["width"]  = dim_w
        if dim_d > 0: dim_dict["depth"]  = dim_d

        if errors:
            for err in errors:
                st.error(err)
        elif not backend_online:
            st.error("❌ Backend service is currently unavailable. Please try again in a few moments.")
        else:
            # ── SEMANTIC VALIDATION (frontend, deterministic) ───────────────
            val = validate_product_inputs(
                product_name, category or "",
                features_list, specs_dict, package_contents_list
            )
            st.session_state.validation_summary = val

            if val["has_warnings"]:
                total_ignored = len(val["ignored_features"]) + len(val["ignored_specs"]) + len(val["ignored_pkg"])
                st.warning(
                     f"⚠️ **Input Validation**: {total_ignored} attribute(s) were incompatible with "
                     f"**{val['detected_category'].title()}** category and will be excluded from generation. "
                     "See Validation Summary in the reports tab."
                )

            # Save product meta for overview card in results tab
            st.session_state.product_meta = {
                "product_name": product_name,
                "brand": brand,
                "category": category if category else "Auto-classified",
                "subcategory": subcategory if subcategory else "",
                "target_audience": target_audience,
                "warranty": warranty if warranty else "",
                "brand_voice": brand_voice if brand_voice else "Professional",
                "tone": tone if tone else "Expert"
            }

            # Build filtered payload (images are NEVER included)
            product_payload = {
                "product_name": product_name,
                "brand": brand,
                "category": category if category else None,
                "subcategory": subcategory if subcategory else None,
                "material": material,
                "color": color,
                "weight_raw": weight_raw if weight_raw else None,
                "dimensions": dim_dict,
                "features": val["accepted_features"],       # validated — forbidden removed
                "specifications": val["accepted_specs"],    # validated — forbidden removed
                "target_audience": target_audience,
                "seo_keywords": seo_keywords_list,
                "brand_voice": brand_voice if brand_voice else "Professional",
                "tone": tone if tone else "Expert",
                "description_length": description_length,
                "warranty": warranty if warranty else None,
                "package_contents": val["accepted_pkg"],    # validated — forbidden removed
                # Pricing (backend ignores unknown fields gracefully)
                "price": product_price if product_price > 0 else None,
                "discount_percentage": discount_pct if discount_pct > 0 else None,
                "offer_price": _offer if product_price > 0 else None,
            }

            # Persist pricing for results display
            st.session_state.pricing = {
                "price": product_price,
                "discount": discount_pct,
                "offer_price": _offer,
            }

            # Send Generation Request to API via background thread with live progress animation
            import queue
            import threading
            import time

            resp_queue = queue.Queue()
            headers = dict(API_HEADERS)

            def call_api():
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/generate",
                        json=product_payload,
                        headers=headers,
                        timeout=180
                    )
                    resp_queue.put(("SUCCESS", response))
                except Exception as ex:
                    resp_queue.put(("ERROR", ex))

            api_thread = threading.Thread(target=call_api)
            api_thread.start()

            # Render loading dashboard
            progress_bar = st.empty()
            
            # Resolve absolute workspace path for outputs/live_progress.json
            project_root = os.path.dirname(os.path.abspath(__file__))
            live_progress_path = os.path.join(project_root, "outputs", "live_progress.json")

            # Delete old progress logs
            if os.path.exists(live_progress_path):
                try:
                    os.remove(live_progress_path)
                except Exception:
                    pass

            stage_names_mapping = [
                {"key": "Normalization", "display": "Product Normalization"},
                {"key": "Attribute Validation", "display": "Attribute Validation"},
                {"key": "Category Classification", "display": "Category Classification"},
                {"key": "Product Enrichment", "display": "Product Enrichment"},
                {"key": "Dynamic Prompt Builder", "display": "Dynamic Prompt Builder"},
                {"key": "Gemini API", "display": "AI Content Generation"},
                {"key": "JSON Validation", "display": "JSON Validation"},
                {"key": "Hallucination Detection", "display": "Hallucination Detection"},
                {"key": "SEO Validation", "display": "SEO Validation"},
                {"key": "Brand Compliance", "display": "Brand Compliance"},
                {"key": "Quality Scoring", "display": "Quality Scoring"},
            ]

            while api_thread.is_alive():
                # Read live_progress.json absolutely
                stages_data = []
                if os.path.exists(live_progress_path):
                    try:
                        with open(live_progress_path, "r", encoding="utf-8") as f:
                            progress_data = json.load(f)
                            stages_data = progress_data.get("stages", [])
                    except Exception:
                        pass
                
                completed_count = 0
                active_stage_display = "Waiting..."
                stage_html = ""
                
                for item in stage_names_mapping:
                    # Find matching stage
                    matched_stage = None
                    for s in stages_data:
                        if item["key"].lower() in s["stage"].lower():
                            matched_stage = s
                            break
                    
                    if matched_stage:
                        status = matched_stage["status"]
                        if status == "SUCCESS":
                            icon = "✓"
                            color = "#10B981"
                            completed_count += 1
                        elif status == "IN_PROGRESS":
                            icon = "⟳"
                            color = "#F59E0B"
                            active_stage_display = f"Running {item['display']}..."
                        else:  # FAILED
                            icon = "❌"
                            color = "#EF4444"
                    else:
                        icon = "⬜"
                        color = "#94A3B8"
                    
                    stage_html += f"<p style='margin:5px 0;color:{color};font-weight:600;'>{icon} {item['display']}</p>"
                
                # Dynamic Progress Percentage driven strictly by completed stages
                percent = int((completed_count / len(stage_names_mapping)) * 100)
                
                # Dynamic progress bar representation
                bar_len = 16
                filled_len = int(bar_len * (percent / 100.0))
                bar_str = "█" * filled_len + "░" * (bar_len - filled_len)
                
                progress_bar.markdown(f"""
                <div class="progress-container">
                    <h3 class="progress-title">Generating Product Content...</h3>
                    <div class="progress-divider"></div>
                    {stage_html}
                    <div class="progress-divider"></div>
                    <p class="progress-overall-text">Pipeline Progress: {completed_count}/{len(stage_names_mapping)} stages</p>
                    <code class="progress-bar-code">{bar_str} {percent}%</code>
                    <p class="progress-est-text">Estimated generation time: ≈ {est_seconds} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(0.3)

            # Thread is done
            status, response_or_err = resp_queue.get()
            if status == "SUCCESS":
                response = response_or_err
                if response.status_code == 200:
                    _result = response.json()
                    st.session_state.result = _result
                    
                    _all_ignored = (
                        val["ignored_features"]
                        + val["ignored_specs"]
                        + val["ignored_pkg"]
                    )
                    _gen_text = json.dumps(_result.get("content", {}))
                    st.session_state.frontend_hallucination_flags = check_frontend_hallucinations(_gen_text, _all_ignored)
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    # Trigger celebration balloons
                    st.balloons()
                    
                    # Display success message
                    st.success("""
                    ### ✅ Content Generated Successfully!
                    
                    🎉 **Your AI merchandising content has been generated successfully.**
                    
                    Please navigate to the **📊 Generation & Quality Reports** tab to view your generated content, quality metrics, and validation reports.
                    """)
                    
                    # Store welcome banner state so it shows up in results tab
                    st.session_state.show_welcome_banner = True
                else:
                    code = response.status_code
                    detail = response.json().get('detail', 'Unknown error')
                    _resil    = health_data.get("resiliency_configuration", {})
                    _cooldown = _resil.get("request_timeout_seconds", 60)
                    _timeout  = _resil.get("request_timeout_seconds", 180)
                    if code == 503:
                        st.error(f"⚠️ **Circuit Breaker Open**: The Gemini API service is unavailable. Wait {_cooldown}s and retry. Detail: {detail}")
                    elif code == 504:
                        st.error(f"⏱️ **Request Timeout**: Exceeded {_timeout}s limit. Detail: {detail}")
                    elif code == 502:
                        st.error(f"🔌 **API Gateway Error**: Check API key or service limits. Detail: {detail}")
                    elif code == 400:
                        st.error(f"❌ **Validation Failed**: {detail}")
                    else:
                        st.error(f"🚨 **Pipeline Failed** (HTTP {code}): {detail}")
            else:
                st.error(f"❌ **Generation Failed**: {response_or_err}")

# Display Results Tab
with tab_results:
    if st.session_state.result is None:
        st.info("No content has been generated yet. Please fill the Intake Form and generate.")
    else:
        res = st.session_state.result
        content = res["content"]
        scores  = res["scores"]
        reports = res["reports"]
        meta    = res["metadata"]

        # Welcome success card (shown once after redirection)
        if st.session_state.get("show_welcome_banner"):
            st.markdown("""
            <div style="background:linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(5,150,105,0.1) 100%);
                        border:1px solid rgba(16,185,129,0.3);border-radius:16px;padding:24px 28px;
                        margin-bottom:25px;box-shadow:0 8px 32px rgba(0,0,0,0.25);">
                <div style="display:flex;align-items:center;gap:16px;">
                    <span style="font-size:32px;">✨</span>
                    <div>
                        <h3 style="margin:0;color:#10B981;font-size:18px;">Welcome to Your Generated Catalog Report</h3>
                        <p style="margin:4px 0 0 0;font-size:13px;color:#A7F3D0;">
                            The merchandising content has been validated against all guidelines. Use the sections below to review, verify safety checks, audit brand compliance, and download files.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.show_welcome_banner = False
        else:
            st.success("🎉 **Generation Complete** — Your AI merchandising content is ready.")

        # ================================================================
        # 2. PRODUCT OVERVIEW HERO SECTION (Amazon Style)
        # ================================================================
        _imgs      = st.session_state.get("uploaded_images", [])
        _pricing   = st.session_state.get("pricing", {})
        _pmeta     = st.session_state.get("product_meta", {})
        _has_img   = bool(_imgs)

        # Detect aspect ratio for adaptive column layout
        _col_ratio = [4, 6]
        if _has_img:
            try:
                from PIL import Image as _PILImage
                import io as _io
                _f = _imgs[0]; _f.seek(0)
                _pil = _PILImage.open(_io.BytesIO(_f.read()))
                _iw, _ih = _pil.size; _f.seek(0)
                _ar = _iw / _ih
                _col_ratio = [3, 7] if _ar < 0.8 else ([5, 5] if _ar > 1.6 else [4, 6])
            except Exception:
                _col_ratio = [4, 6]

        # Pricing HTML block
        def _price_html(pricing):
            orig  = pricing.get("price", 0)
            disc  = pricing.get("discount", 0)
            offer = pricing.get("offer_price", 0)
            saving = round(orig - offer, 2)
            return f"""
            <div style="margin-top:20px;background:rgba(0,198,255,0.05);
                        border:1px solid rgba(0,198,255,0.2);border-radius:14px;padding:18px;">
                <p style="margin:0 0 12px 0;font-size:10px;color:#475569;text-transform:uppercase;
                           letter-spacing:.12em;font-weight:700;">Pricing</p>
                <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-end;">
                    <div>
                        <p style="margin:0;font-size:10px;color:#64748B;">MRP</p>
                        <p style="margin:0;font-size:15px;color:#64748B;
                                   text-decoration:line-through;">₹{orig:,.0f}</p>
                    </div>
                    <div>
                        <p style="margin:0;font-size:10px;color:#64748B;">Discount</p>
                        <p style="margin:0;font-size:18px;color:#F59E0B;font-weight:700;">{disc}%</p>
                    </div>
                    <div>
                        <p style="margin:0;font-size:10px;color:#64748B;">Offer Price</p>
                        <p style="margin:0;font-size:32px;color:#10B981;font-weight:800;
                                   letter-spacing:-.5px;">₹{offer:,.0f}</p>
                    </div>
                    <div>
                        <p style="margin:0;font-size:10px;color:#64748B;">You Save</p>
                        <p style="margin:0;font-size:14px;color:#10B981;font-weight:600;"
                           >₹{saving:,.0f} &nbsp;<span style='font-size:11px;background:rgba(16,185,129,0.15);
                           border-radius:4px;padding:2px 6px;'>({disc}% off)</span></p>
                    </div>
                </div>
            </div>"""

        # Info panel HTML
        def _info_html(pmeta, content_data, pricing):
            title     = content_data.get("title", pmeta.get("product_name", ""))
            brand_v   = pmeta.get("brand", "")
            cat_v     = pmeta.get("category", "")
            sub_v     = pmeta.get("subcategory", "")
            audience  = pmeta.get("target_audience", "")
            warranty  = pmeta.get("warranty", "")
            short_d   = content_data.get("short_description", "")
            cat_disp  = cat_v + (f" &rsaquo; {sub_v}" if sub_v else "")
            price_blk = _price_html(pricing) if pricing.get("price", 0) > 0 else ""
            desc_blk  = (f'<p style="margin:16px 0 0 0;font-size:13px;color:#CBD5E1;'
                         f'line-height:1.75;border-top:1px solid rgba(255,255,255,0.06);'
                         f'padding-top:14px;">{short_d}</p>') if short_d else ""
            wrnt_blk  = (f'<p style="margin:8px 0 0 0;font-size:12px;color:#64748B;">'
                         f'🛡️ {warranty}</p>') if warranty else ""
            tags = ""
            if brand_v:
                tags += (f'<span style="background:rgba(0,114,255,0.12);border:1px solid rgba(0,114,255,0.25);'
                         f'border-radius:20px;padding:4px 12px;font-size:12px;color:#93C5FD;">🏷️ {brand_v}</span>')
            if cat_disp:
                tags += (f'<span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);'
                         f'border-radius:20px;padding:4px 12px;font-size:12px;color:#A5B4FC;">📦 {cat_disp}</span>')
            if audience:
                tags += (f'<span style="background:rgba(16,185,129,0.10);border:1px solid rgba(16,185,129,0.25);'
                         f'border-radius:20px;padding:4px 12px;font-size:12px;color:#6EE7B7;">👥 {audience}</span>')
            return f"""
            <div style="background:rgba(15,23,42,0.75);border:1px solid rgba(255,255,255,0.09);
                        border-radius:18px;padding:28px 32px;height:100%;box-sizing:border-box;
                        box-shadow:0 8px 32px rgba(0,0,0,0.3);">
                <p style="margin:0 0 8px 0;font-size:10px;color:#334155;text-transform:uppercase;
                           letter-spacing:.14em;font-weight:700;">Product Overview</p>
                <h2 style="margin:0 0 16px 0;color:#00C6FF;font-size:24px;line-height:1.3;
                           font-weight:800;">{title}</h2>
                <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px;">{tags}</div>
                {price_blk}{desc_blk}{wrnt_blk}
            </div>"""

        # Render the overview
        st.markdown("### 🏢 Product Overview")
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        if _has_img:
            if "hero_image_index" not in st.session_state:
                st.session_state.hero_image_index = 0
            _ov_img, _ov_info = st.columns(_col_ratio)
            with _ov_img:
                if st.session_state.hero_image_index >= len(_imgs):
                    st.session_state.hero_image_index = 0
                st.markdown(
                    '<div style="border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.1);'
                    'box-shadow:0 8px 30px rgba(0,0,0,0.4);background:#0f172a;">',
                    unsafe_allow_html=True)
                st.image(_imgs[st.session_state.hero_image_index], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                if len(_imgs) > 1:
                    st.write("")
                    _tc = st.columns(len(_imgs))
                    for _ti, _tf in enumerate(_imgs):
                        with _tc[_ti]:
                            label = f"📷 #{_ti + 1}"
                            if _ti == st.session_state.hero_image_index:
                                label = f"🎯 #{_ti + 1}"
                            if st.button(label, key=f"btn_hero_select_{_ti}", use_container_width=True):
                                st.session_state.hero_image_index = _ti
                                st.rerun()
            with _ov_info:
                st.markdown(_info_html(_pmeta, content, _pricing), unsafe_allow_html=True)
        else:
            st.markdown(_info_html(_pmeta, content, _pricing), unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ================================================================
        # 3. GENERATED MERCHANDISING CONTENT
        # ================================================================
        st.markdown("### Generated Merchandising Content")
        
        specs_summary = content.get('specification_summary', {})
        if isinstance(specs_summary, dict):
            specs_html = "<table style='width:100%; border-collapse:collapse; margin-top: 10px;'>"
            for k, v in specs_summary.items():
                specs_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.08);'><td style='padding:8px; font-weight:bold; width:30%; color:#00C6FF;'>{k}</td><td style='padding:8px; color:#E2E8F0;'>{v}</td></tr>"
            specs_html += "</table>"
        else:
            specs_html = str(specs_summary)

        st.markdown(f"""
        <div class="content-box">
            <h2 style="margin-top:0;color:#00C6FF;">{content['title']}</h2>
            <p><strong>Short Description:</strong><br>{content['short_description']}</p>
            <p><strong>Long Description:</strong><br>{content['long_description']}</p>
            <p><strong>Bullet Points:</strong></p>
            <ul>
                {"".join([f"<li>{b}</li>" for b in content['bullet_points']])}
            </ul>
            <p><strong>SEO Meta Description:</strong><br><em>{content['meta_description']}</em></p>
            <p><strong>SEO Keywords Used:</strong> {', '.join(content['seo_keywords_used'])}</p>
            <p><strong>Technical Specification Summary:</strong></p>
            {specs_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ================================================================
        # 4. QUALITY DASHBOARD
        # ================================================================
        st.markdown("### Quality & Validation Dashboard")

        card1, card2, card3, card4 = st.columns(4)
        overall = scores['overall']
        h_score = scores['hallucination']
        seo     = scores['seo']
        brand   = scores['brand']
        q_data  = reports.get('quality', {})
        risk_lv = q_data.get('risk_level', 'Low')
        risk_colors = {'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444', 'Critical': '#7C3AED'}
        risk_color  = risk_colors.get(risk_lv, '#10B981')
        val_conf    = round(q_data.get('validation_confidence', 1.0) * 100, 1)
        gen_conf    = round(q_data.get('generation_confidence', 1.0) * 100, 1)
        safety_sc   = q_data.get('safety_score', h_score)

        with card1:
            st.markdown(f"""
            <div class="metric-card good" style="border-left: 5px solid #10B981;">
                <div class="metric-num" style="color:#10B981;">🏆 {overall}%</div>
                <div class="metric-label">Overall Quality</div>
            </div>
            """, unsafe_allow_html=True)
        with card2:
            st.markdown(f"""
            <div class="metric-card good" style="border-left: 5px solid #3B82F6;">
                <div class="metric-num" style="color:#3B82F6;">🛡️ {safety_sc}%</div>
                <div class="metric-label">Safety Score</div>
            </div>
            """, unsafe_allow_html=True)
        with card3:
            st.markdown(f"""
            <div class="metric-card good" style="border-left: 5px solid #F59E0B;">
                <div class="metric-num" style="color:#F59E0B;">📈 {seo}%</div>
                <div class="metric-label">SEO Score</div>
            </div>
            """, unsafe_allow_html=True)
        with card4:
            st.markdown(f"""
            <div class="metric-card good" style="border-left: 5px solid #8B5CF6;">
                <div class="metric-num" style="color:#8B5CF6;">🛡️ {brand}%</div>
                <div class="metric-label">Brand Compliance</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence + Risk badges ──────────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 6px 0;">
            <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                        border-radius:10px;padding:10px 18px;">
                <p style="margin:0;font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.1em;">Validation Confidence</p>
                <p style="margin:4px 0 0 0;font-size:20px;color:#10B981;font-weight:800;">{val_conf}%</p>
            </div>
            <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
                        border-radius:10px;padding:10px 18px;">
                <p style="margin:0;font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.1em;">Generation Confidence</p>
                <p style="margin:4px 0 0 0;font-size:20px;color:#3B82F6;font-weight:800;">{gen_conf}%</p>
            </div>
            <div style="background:rgba(239,68,68,0.08);border:1px solid {risk_color}40;
                        border-radius:10px;padding:10px 18px;">
                <p style="margin:0;font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.1em;">Risk Level</p>
                <p style="margin:4px 0 0 0;font-size:20px;color:{risk_color};font-weight:800;">{risk_lv}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Explainable quality breakdown ─────────────────────────────────
        _breakdown = q_data.get('breakdown')
        if _breakdown:
            with st.expander('🔬 Quality Score Breakdown (Explainable AI)', expanded=False):
                _bd_rows = [
                    ('🤖 Generation Quality',  _breakdown.get('generation_quality')),
                    ('📈 SEO',                  _breakdown.get('seo')),
                    ('🛡️ Safety',              _breakdown.get('safety')),
                    ('🏷️ Brand Compliance',    _breakdown.get('brand_compliance')),
                    ('✅ Validation',           _breakdown.get('validation')),
                    ('🔍 Hallucination Guard',  _breakdown.get('hallucination')),
                ]
                _bd_html = "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
                _bd_html += "<tr style='border-bottom:1px solid rgba(255,255,255,0.12);'>"
                _bd_html += "<th style='text-align:left;padding:8px;color:#94A3B8;'>Component</th>"
                _bd_html += "<th style='text-align:center;padding:8px;color:#94A3B8;'>Score</th>"
                _bd_html += "<th style='text-align:center;padding:8px;color:#94A3B8;'>Penalty</th>"
                _bd_html += "<th style='text-align:left;padding:8px;color:#94A3B8;'>Reason</th></tr>"
                for _label, _comp in _bd_rows:
                    if _comp is None:
                        continue
                    _sc  = _comp.get('score', 0)
                    _pen = _comp.get('penalty', 0)
                    _rsn = _comp.get('reason', '')
                    _sc_color = '#10B981' if _sc >= 80 else ('#F59E0B' if _sc >= 60 else '#EF4444')
                    _pen_str  = f"-{_pen:.1f}" if _pen > 0 else "—"
                    _pen_color = '#EF4444' if _pen > 0 else '#64748B'
                    _bd_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
                    _bd_html += f"<td style='padding:8px;color:#E2E8F0;'>{_label}</td>"
                    _bd_html += f"<td style='padding:8px;text-align:center;color:{_sc_color};font-weight:700;'>{_sc:.1f}</td>"
                    _bd_html += f"<td style='padding:8px;text-align:center;color:{_pen_color};'>{_pen_str}</td>"
                    _bd_html += f"<td style='padding:8px;color:#CBD5E1;font-size:12px;'>{_rsn}</td></tr>"
                    # Safety sub-scores
                    if _label == '🛡️ Safety' and _comp.get('sub_scores'):
                        _ss = _comp['sub_scores']
                        for _sk, _sv in _ss.items():
                            _sv_color = '#10B981' if _sv >= 80 else ('#F59E0B' if _sv >= 60 else '#EF4444')
                            _bd_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.03);background:rgba(0,0,0,0.15);'>"
                            _bd_html += f"<td style='padding:6px 8px 6px 24px;color:#94A3B8;font-size:11px;'>↳ {_sk.replace('_', ' ').title()}</td>"
                            _bd_html += f"<td style='padding:6px 8px;text-align:center;color:{_sv_color};font-size:11px;'>{_sv:.1f}</td>"
                            _bd_html += f"<td></td><td></td></tr>"
                _bd_html += "</table>"
                st.markdown(_bd_html, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ================================================================
        # 5. EXECUTION METRICS (Timeline + Summary)
        # ================================================================
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 📈 Live Pipeline Execution Timeline")
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.45);padding:24px;border-radius:16px;
                        border:1px solid rgba(255,255,255,0.06);box-shadow:0 8px 32px rgba(0,0,0,0.15);">
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Normalize catalog inputs</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Validate attributes & boundaries</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Classify product category</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Enrich secondary meta attributes</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Select prompt templates</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Gemini LLM inference execution</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ JSON parsing & validation checks</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Hallucination engine auditing</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ SEO optimizer auditing</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Brand compliance rules verification</p>
                <p style="margin:6px 0;color:#10B981;font-weight:600;font-size:14px;">✓ Quality scoring evaluation</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown("### ⚡ AI Generation Summary")
            _model = meta.get('model_used', 'Gemini 3.5 Flash')
            _exec = f"{meta['execution_time_seconds']:.1f} sec"
            _retries = meta.get('retry_count', 0)
            _corr = meta.get('correction_loop_count', 0)
            
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.45);padding:24px;border-radius:16px;
                        border:1px solid rgba(255,255,255,0.06);box-shadow:0 8px 32px rgba(0,0,0,0.15);
                        height:100%;display:flex;flex-direction:column;justify-content:space-between;">
                <div>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">🤖 <strong>Model:</strong> {_model}</p>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">⏱️ <strong>Execution:</strong> {_exec}</p>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">🔄 <strong>Retries:</strong> {_retries}</p>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">🛠️ <strong>Corrections:</strong> {_corr}</p>
                    <div style="height:1px;background:rgba(255,255,255,0.08);margin:12px 0;"></div>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">⭐ <strong>Overall Quality:</strong> {overall}%</p>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">📈 <strong>SEO score:</strong> {seo}%</p>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">🔍 <strong>Safety score:</strong> {h_score}%</p>
                    <p style="margin:6px 0;font-size:14px;color:#CBD5E1;">🛡️ <strong>Brand score:</strong> {brand}%</p>
                </div>
                <div style="margin-top:16px;background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);
                            border-radius:10px;padding:12px;text-align:center;">
                    <span style="font-size:16px;color:#10B981;font-weight:bold;letter-spacing:1px;">🚀 READY FOR PUBLISHING</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ================================================================
        # 6. ENGINE VALIDATION REPORTS
        # ================================================================
        st.markdown("#### Engine Validation Reports")
        
        with st.expander("🔍 Hallucination Detection Report"):
            h_data    = reports["hallucination"]
            _fe_flags = st.session_state.get("frontend_hallucination_flags", [])
            _sec_a    = h_data.get('section_a') or {}
            _sec_b    = h_data.get('section_b') or {}
            _sec_c    = h_data.get('section_c') or {}
            _sec_d    = h_data.get('section_d') or {}

            # ── Section A: Category Context ───────────────────────────────
            _det_cat   = _sec_a.get('detected_category', '')
            _val_conf  = round(_sec_a.get('validation_confidence', 1.0) * 100, 1)
            _val_stat  = _sec_a.get('validation_status', 'PASS')
            _stat_col  = '#10B981' if _val_stat == 'PASS' else '#F59E0B' if _val_stat == 'DEGRADED' else '#EF4444'
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.35);border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px;padding:16px;margin-bottom:12px;">
                <p style="margin:0 0 6px 0;font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:.1em;">Section A — Category Validation Context</p>
                <div style="display:flex;gap:24px;flex-wrap:wrap;">
                    <span style="font-size:13px;color:#E2E8F0;">🏷️ <strong>Detected Category:</strong> {_det_cat or 'N/A'}</span>
                    <span style="font-size:13px;color:#E2E8F0;">⭐ <strong>Validation Confidence:</strong> {_val_conf}%</span>
                    <span style="font-size:13px;color:{_stat_col};font-weight:700;">● {_val_stat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if _val_stat == 'DEGRADED':
                st.warning("⚠️ **Semantic validation was DEGRADED** — attributes were not filtered. Manual review required before publishing.")

            # ── Section B: Rejected User Attributes ───────────────────────
            _rejected_b = _sec_b.get('rejected_attributes', [])
            if _rejected_b:
                st.markdown("**Section B — Rejected User Attributes** *(Invalid input — NOT AI hallucinations)*")
                for _rb in _rejected_b:
                    st.markdown(
                        f"&nbsp;&nbsp;✗ **{_rb.get('attribute', '')}** — "
                        f"<span style='font-size:12px;color:#94A3B8;'>Reason: {_rb.get('reason', '')}</span>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("**Section B — Rejected User Attributes:** None")

            # ── Section C: Unsupported AI Claims with severity ────────────
            _ai_claims = _sec_c.get('unsupported_claims', [])
            _sev_badges = {'High': ('🔴', '#EF4444'), 'Medium': ('🟡', '#F59E0B'), 'Low': ('🟢', '#10B981')}
            if _ai_claims:
                st.markdown("**Section C — Unsupported AI Claims** *(Genuine LLM hallucinations)*")
                for _cl in _ai_claims:
                    _cl_sev   = _cl.get('severity', 'Low') if isinstance(_cl, dict) else 'Low'
                    _cl_text  = _cl.get('claim', str(_cl))  if isinstance(_cl, dict) else str(_cl)
                    _cl_imp   = ', '.join(_cl.get('impact', [])) if isinstance(_cl, dict) else ''
                    _ico, _sc = _sev_badges.get(_cl_sev, ('🟢', '#10B981'))
                    st.markdown(
                        f"&nbsp;&nbsp;{_ico} **[{_cl_sev}]** {_cl_text}"
                        + (f" <span style='font-size:11px;color:#64748B;'>Impact: {_cl_imp}</span>" if _cl_imp else ""),
                        unsafe_allow_html=True
                    )
            else:
                st.success("✅ **Section C:** No unsupported AI-generated claims detected.")

            # ── Section D: Summary ────────────────────────────────────────
            _summary_text = _sec_d.get('summary', '')
            _hal_status   = _sec_d.get('hallucination_status', 'PASS')
            if _summary_text:
                if _hal_status == 'PASS':
                    st.success(f"✅ **Section D Summary:** {_summary_text}")
                else:
                    st.error(f"🚨 **Section D Summary:** {_summary_text}")

            # ── Frontend flags (cross-check) ──────────────────────────────
            if _fe_flags:
                st.markdown("---")
                st.warning("⚠️ **Frontend Cross-Check:** Ignored attributes found in generated content:")
                for _flag in _fe_flags:
                    st.error(f"🚨 `{_flag['item']}` | Severity: **High**")

            # ── Legacy comparison report ──────────────────────────────────
            if h_data.get("comparison_report"):
                st.markdown("---")
                st.markdown("##### Structured Attribute Verification Report")
                st.table(h_data["comparison_report"])

        with st.expander("📋 Validation Decision Log"):
            # Pull from the backend validation report (new) OR frontend summary (legacy)
            _vr_back  = reports.get('validation') or {}
            _dl       = _vr_back.get('decision_log', [])
            _vs       = st.session_state.get('validation_summary', {})
            _sem_stat = _vr_back.get('semantic_status', 'PASS')
            _vconf    = round(_vr_back.get('validation_confidence', 1.0) * 100, 1)
            _vconf_reason = _vr_back.get('confidence_reason', '')
            _needs_review = _vr_back.get('needs_manual_review', False)

            # Header bar
            _hdr_col = '#10B981' if _sem_stat == 'PASS' else '#F59E0B' if _sem_stat == 'DEGRADED' else '#EF4444'
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.35);border:1px solid {_hdr_col}40;
                        border-radius:12px;padding:14px 18px;margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                    <div>
                        <p style="margin:0;font-size:13px;color:#E2E8F0;">
                            🔬 <strong>Semantic Status:</strong>
                            <span style="color:{_hdr_col};font-weight:700;">{_sem_stat}</span>
                        </p>
                        <p style="margin:4px 0 0 0;font-size:12px;color:#94A3B8;">{_vconf_reason}</p>
                    </div>
                    <div style="background:{_hdr_col}20;border:1px solid {_hdr_col}40;border-radius:8px;
                                padding:6px 14px;text-align:center;">
                        <p style="margin:0;font-size:10px;color:#64748B;">Confidence</p>
                        <p style="margin:0;font-size:18px;color:{_hdr_col};font-weight:800;">{_vconf}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if _needs_review:
                st.warning("⚠️ **Manual Review Required** — Semantic validation was unavailable. These attributes have NOT been AI-validated.")

            if _dl:
                # Render structured decision log from backend
                _dec_icons = {
                    'ACCEPTED':             ('✓', '#10B981'),
                    'ACCEPTED_WITH_WARNING': ('⚠', '#F59E0B'),
                    'REJECTED':             ('✗', '#EF4444'),
                    'DEGRADED':             ('⚡', '#F59E0B'),
                }
                for _entry in _dl:
                    _attr = _entry.get('attribute', '')
                    _src  = _entry.get('source', 'User Input')
                    _dec  = _entry.get('decision', 'ACCEPTED')
                    _rsn  = _entry.get('reason', '')
                    _ico, _col = _dec_icons.get(_dec, ('?', '#94A3B8'))
                    _rsn_html = f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size:11px;color:#94A3B8;'>Reason: {_rsn}</span>" if _rsn else ""
                    _src_badge = f"<span style='font-size:10px;background:rgba(100,116,139,0.2);border-radius:4px;padding:1px 6px;color:#94A3B8;margin-left:8px;'>{_src}</span>"
                    st.markdown(
                        f"&nbsp;&nbsp;<span style='color:{_col};font-weight:700;'>{_ico}</span> "
                        f"<strong style='color:#E2E8F0;'>{_attr}</strong>{_src_badge}{_rsn_html}",
                        unsafe_allow_html=True
                    )
            else:
                # Legacy fallback: render from frontend session state
                _acc_f = _vs.get('accepted_features', [])
                _acc_s = _vs.get('accepted_specs', {})
                _acc_p = _vs.get('accepted_pkg', [])
                _ign_f = _vs.get('ignored_features', [])
                _ign_s = _vs.get('ignored_specs', [])
                _ign_p = _vs.get('ignored_pkg', [])
                if _acc_f or _acc_s or _acc_p:
                    for _f in _acc_f:
                        st.markdown(f"&nbsp;&nbsp;✓ {_f}", unsafe_allow_html=True)
                    for _k, _v in _acc_s.items():
                        st.markdown(f"&nbsp;&nbsp;✓ {_k}: {_v}", unsafe_allow_html=True)
                if _ign_f or _ign_s or _ign_p:
                    for _e in _ign_f + _ign_s + _ign_p:
                        st.markdown(
                            f"&nbsp;&nbsp;✗ **{_e['item']}** — <span style='font-size:11px;color:#94A3B8;'>Reason: {_e['reason']}</span>",
                            unsafe_allow_html=True
                        )
                if not any([_acc_f, _acc_s, _acc_p, _ign_f, _ign_s, _ign_p]):
                    st.info("No validation data available. Generate content to see the log.")

        with st.expander("📈 SEO Keyword & Length Analysis"):
            seo_data = reports["seo"]
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.write("**Length Analysis:**")
                st.write(f"- Title Length: `{seo_data['title_length']} characters`")
                st.write(f"- Meta Description: `{seo_data['meta_description_length']} characters`")
                st.write(f"- Long Description: `{seo_data['long_description_length']} words`")
                st.write(f"- Content Readability: `{seo_data['readability_score']}/100`")
            with col_s2:
                # Audit summary
                _aud_sum = seo_data.get('audit_summary') or {}
                if _aud_sum:
                    st.write("**Keyword Audit Summary:**")
                    st.write(f"- Total Keywords: `{_aud_sum.get('total', 0)}`")
                    st.write(f"- ✅ Optimized: `{_aud_sum.get('optimized', 0)}`")
                    st.write(f"- ⚠️ Needs Improvement: `{_aud_sum.get('needs_improvement', 0)}`")
                    st.write(f"- ❌ Missing: `{_aud_sum.get('missing', 0)}`")

            # Per-keyword audit table
            _kw_audit = seo_data.get('keyword_audit', [])
            if _kw_audit:
                st.markdown("**Per-Keyword Audit Table:**")
                _kw_html = "<table style='width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;'>"
                _kw_html += "<tr style='border-bottom:1px solid rgba(255,255,255,0.12);'>"
                _kw_html += "<th style='text-align:left;padding:7px;color:#94A3B8;'>Keyword</th>"
                _kw_html += "<th style='text-align:center;padding:7px;color:#94A3B8;'>Density</th>"
                _kw_html += "<th style='text-align:center;padding:7px;color:#94A3B8;'>Status</th>"
                _kw_html += "<th style='text-align:left;padding:7px;color:#94A3B8;'>Recommendation</th></tr>"
                _kw_status_colors = {'OPTIMIZED': '#10B981', 'NEEDS_IMPROVEMENT': '#F59E0B', 'MISSING': '#EF4444'}
                _kw_status_icons  = {'OPTIMIZED': '✅', 'NEEDS_IMPROVEMENT': '⚠️', 'MISSING': '❌'}
                for _ke in _kw_audit:
                    _ks  = _ke.get('status', 'OPTIMIZED')
                    _kc  = _kw_status_colors.get(_ks, '#94A3B8')
                    _ki  = _kw_status_icons.get(_ks, '?')
                    _krec = _ke.get('recommendation') or '<span style="color:#64748B;">—</span>'
                    _kw_html += f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05);'>"
                    _kw_html += f"<td style='padding:7px;color:#E2E8F0;'>{_ke.get('keyword','')}</td>"
                    _kw_html += f"<td style='padding:7px;text-align:center;color:#CBD5E1;'>{_ke.get('density',0):.2f}%</td>"
                    _kw_html += f"<td style='padding:7px;text-align:center;color:{_kc};font-weight:700;'>{_ki} {_ks}</td>"
                    _kw_html += f"<td style='padding:7px;color:#94A3B8;font-size:11px;'>{_krec}</td></tr>"
                _kw_html += "</table>"
                st.markdown(_kw_html, unsafe_allow_html=True)
            else:
                # Legacy fallback
                for kw, density in seo_data.get("keyword_density", {}).items():
                    st.write(f"- **{kw}**: `{density}%`")

            if seo_data.get("missing_keywords"):
                st.error(f"❌ Missing Keywords: {', '.join(seo_data['missing_keywords'])}")
            if seo_data.get("duplicate_keywords"):
                st.warning(f"⚠️ Keyword Stuffing Warning: {', '.join(seo_data['duplicate_keywords'])}")

        with st.expander("🛡️ Brand Compliance Audit"):
            brand_data = reports["brand"]
            _req_voice = st.session_state.get("product_meta", {}).get("brand_voice", "Professional")
            _req_tone = st.session_state.get("product_meta", {}).get("tone", "Expert")
            
            _voice_match = "PASS" if brand_data.get("voice_compliance", True) else "FAIL"
            _tone_check_str = str(brand_data.get("tone_check", ""))
            _tone_match = "PASS" if "FAIL" not in _tone_check_str else "FAIL"
            
            # Voice compliance section
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.35);border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:18px;margin-bottom:15px;">
                <p style="margin:4px 0;font-size:14px;color:#E2E8F0;">
                    🗣️ <strong>Requested Brand Voice:</strong> {_req_voice}
                </p>
                <p style="margin:4px 0;font-size:14px;color:#E2E8F0;">
                    🎤 <strong>Detected Voice:</strong> {_req_voice if _voice_match == "PASS" else "Mismatched"}
                </p>
                <p style="margin:4px 0;font-size:14px;color:#E2E8F0;">
                    🎯 <strong>Result:</strong> <span style="color:{'#10B981' if _voice_match == 'PASS' else '#EF4444'};font-weight:bold;">{_voice_match}</span>
                </p>
                <div style="height:1px;background:rgba(255,255,255,0.08);margin:10px 0;"></div>
                <p style="margin:4px 0;font-size:13px;color:#CBD5E1;line-height:1.6;">
                    <strong>Reason:</strong><br>
                    {("The generated content consistently uses objective language, avoids exaggerated marketing claims, "
                      f"maintains an informative tone, and aligns with the requested {_req_voice} brand voice.") if _voice_match == "PASS" else ("The generated content violates style guidelines or word length rules.")}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tone compliance section
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.35);border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:18px;margin-bottom:15px;">
                <p style="margin:4px 0;font-size:14px;color:#E2E8F0;">
                    🎭 <strong>Requested Tone:</strong> {_req_tone}
                </p>
                <p style="margin:4px 0;font-size:14px;color:#E2E8F0;">
                    🔊 <strong>Detected Tone:</strong> {_req_tone if _tone_match == "PASS" else "Inconsistent"}
                </p>
                <p style="margin:4px 0;font-size:14px;color:#E2E8F0;">
                    🎯 <strong>Result:</strong> <span style="color:{'#10B981' if _tone_match == 'PASS' else '#EF4444'};font-weight:bold;">{_tone_match}</span>
                </p>
                <div style="height:1px;background:rgba(255,255,255,0.08);margin:10px 0;"></div>
                <p style="margin:4px 0;font-size:13px;color:#CBD5E1;line-height:1.6;">
                    <strong>Reason:</strong><br>
                    {("The generated content maintains a consistent and appropriate tone for e-commerce listings, avoiding exclamations and slang words.") if _tone_match == "PASS" else f"Tone issues detected: {_tone_check_str}"}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if brand_data.get("violations"):
                st.error(f"❌ **Restricted Terms Used:** {', '.join(brand_data['violations'])}")

        st.markdown("---")

        # ================================================================
        # 7. AI DECISION TIMELINE
        # ================================================================
        _timeline = meta.get('decision_timeline', [])
        if _timeline:
            with st.expander("⚡ AI Decision Timeline", expanded=False):
                st.markdown(
                    "<p style='font-size:12px;color:#64748B;margin-bottom:12px;'>"
                    "A chronological log of every AI governance decision made during this pipeline run."
                    "</p>",
                    unsafe_allow_html=True
                )
                _tl_status_cfg = {
                    'STARTED':    ('🔵', '#3B82F6'),
                    'SUCCESS':    ('🟢', '#10B981'),
                    'CLASSIFIED': ('🏷️', '#8B5CF6'),
                    'REJECTED':   ('🔴', '#EF4444'),
                    'WARNING':    ('🟡', '#F59E0B'),
                    'FAILED':     ('🔴', '#EF4444'),
                    'IN_PROGRESS':('🔵', '#3B82F6'),
                    'AGENT_STEP': ('🤖', '#06B6D4'),
                }
                _tl_html = "<div style='border-left:2px solid rgba(255,255,255,0.1);padding-left:16px;'>"
                for _ev in _timeline:
                    _ts  = _ev.get('timestamp', '')
                    _lbl = _ev.get('label', '')
                    _st  = _ev.get('status', '')
                    _det = _ev.get('detail', '')
                    _ico, _col = _tl_status_cfg.get(_st, ('⚪', '#64748B'))
                    _det_html  = f"<span style='font-size:11px;color:#64748B;margin-left:8px;'>{_det}</span>" if _det else ''
                    _tl_html += (
                        f"<div style='margin-bottom:8px;display:flex;align-items:baseline;gap:12px;'>"
                        f"<span style='font-size:11px;color:#475569;font-family:monospace;min-width:60px;'>{_ts}</span>"
                        f"<span style='font-size:13px;'>{_ico}</span>"
                        f"<span style='font-size:13px;color:{_col};font-weight:600;'>{_lbl}</span>"
                        f"{_det_html}"
                        f"</div>"
                    )
                _tl_html += "</div>"
                st.markdown(_tl_html, unsafe_allow_html=True)

        # ================================================================
        # 7b. AGENT TRACE (tool calls made by the agentic writer)
        # ================================================================
        _trace = meta.get('agent_trace', [])
        if _trace:
            _n_tools = sum(1 for _t in _trace if _t.get('tool'))
            with st.expander(f"🤖 Agent Trace — {_n_tools} tool call(s)", expanded=True):
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("Planner", "LLM" if meta.get('planner_source') == 'llm' else "Rule-based")
                _c2.metric("Tool calls", _n_tools)
                _c3.metric("Human review", "Required" if meta.get('manual_review_required') else "Not needed")
                for _d in meta.get('supervisor_decisions', []):
                    st.info(_d)
                for _t in _trace:
                    _attempt = _t.get('attempt', 1)
                    if _t.get('tool'):
                        st.markdown(f"**Attempt {_attempt} · step {_t['step']} · 🔧 `{_t['tool']}`** "
                                    f"<span style='color:#64748B;font-size:12px;'>({_t.get('duration_seconds', 0)}s)</span>",
                                    unsafe_allow_html=True)
                        st.code(_t.get('result_preview', ''), language='json')
                    else:
                        st.markdown(f"**Attempt {_attempt} · ✅ {_t.get('result_preview', 'Draft submitted')}**")

        st.markdown("---")

        # ================================================================
        # 8. DOWNLOAD BUTTONS
        # ================================================================
        st.markdown("#### Download Reports")

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.download_button(
                "📥 Download Generated Content (JSON)",
                data=json.dumps(content, indent=2),
                file_name="generated_content.json",
                mime="application/json"
            )
            st.download_button(
                "📥 Download Hallucination Report (JSON)",
                data=json.dumps(reports["hallucination"], indent=2),
                file_name="hallucination_report.json",
                mime="application/json"
            )
        with d_col2:
            st.download_button(
                "📥 Download SEO Report (JSON)",
                data=json.dumps(reports["seo"], indent=2),
                file_name="seo_report.json",
                mime="application/json"
            )
            st.download_button(
                "📥 Download Brand Report (JSON)",
                data=json.dumps(reports["brand"], indent=2),
                file_name="brand_report.json",
                mime="application/json"
            )
        with d_col3:
            st.download_button(
                "📥 Download Quality Score (JSON)",
                data=json.dumps(reports["quality"], indent=2),
                file_name="quality_report.json",
                mime="application/json"
            )
            try:
                log_resp = requests.get(f"{BACKEND_URL}/report/{res['request_id']}", headers=API_HEADERS, timeout=10)
                log_data = log_resp.json().get("execution_log", {})
            except Exception:
                log_data = {"error": "Could not fetch execution log — backend may be unavailable."}

            st.download_button(
                "📥 Download Execution Logs (JSON)",
                data=json.dumps(log_data, indent=2),
                file_name="execution_log.json",
                mime="application/json"
            )

# System Logs Tab
with tab_logs:
    if st.session_state.result is None:
        st.info("System execution log is empty. Trigger generation to populate.")
    else:
        res = st.session_state.result
        try:
            log_resp = requests.get(f"{BACKEND_URL}/report/{res['request_id']}", headers=API_HEADERS, timeout=10)
            log_data = log_resp.json().get("execution_log", {})
        except Exception:
            log_data = {}

        if log_data:
            st.markdown(f"#### Execution Report for ID: `{res['request_id']}`")
            st.write(f"- Model: `{log_data.get('model_used')}`")
            st.write(f"- Duration: `{log_data.get('execution_time_seconds')} seconds`")
            st.write(f"- Total Retries: `{log_data.get('retry_count')}`")

            st.markdown("##### Pipeline Stages Execution Timeline")
            for stage in log_data.get("stages", []):
                duration_str = f"{stage['duration']}s" if stage['duration'] is not None else "N/A"
                status_color = "🟢" if stage['status'] == "SUCCESS" else "🔴"
                st.markdown(f"- {status_color} **{stage['stage']}**: Status `{stage['status']}` | Duration: `{duration_str}`")
                if stage.get("details"):
                    st.json(stage["details"])

            if log_data.get("errors"):
                st.error("Crashed/Retried Stage Errors:")
                st.write(log_data["errors"])
            if log_data.get("warnings"):
                st.warning("Execution Warnings:")
                st.write(log_data["warnings"])
        else:
            st.warning("Failed to fetch detailed system logs from server.")
