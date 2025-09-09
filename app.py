import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from collections import defaultdict

# Hugging Face CLIP
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------
# 1. Load CLIP Model (cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_clip_model_hf(device="cpu"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


# ---------------------------------------------------------
# 2. Candidate Prompts
# ---------------------------------------------------------
def build_candidate_prompts_hf():
    candidates = {
        "recyclable": [
            "a photo of a plastic bottle",
            "a photo of a metal can",
            "a photo of a glass jar",
            "a photo of recyclable packaging"
        ],
        "biodegradable": [
            "a photo of fruit peels",
            "a photo of vegetable waste",
            "a photo of leftover food",
            "a photo of garden leaves"
        ],
        "hazardous": [
            "a photo of a syringe",
            "a photo of a used battery",
            "a photo of a chemical container",
            "a photo of e-waste"
        ],
        "other": [
            "a photo of mixed waste",
            "a photo of general trash"
        ]
    }

    texts, label_index = [], []
    for cat, prompts in candidates.items():
        for p in prompts:
            texts.append(p)
            label_index.append(cat)
    return texts, label_index


# ---------------------------------------------------------
# 3. Classification Function
# ---------------------------------------------------------
def classify_image_clip_hf(image: Image.Image, model, processor, device="cpu"):
    texts, label_index = build_candidate_prompts_hf()

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # Aggregate scores
    cat_scores = defaultdict(list)
    for p, cat in zip(probs, label_index):
        cat_scores[cat].append(float(p))
    cat_mean_scores = {cat: float(np.mean(scores)) for cat, scores in cat_scores.items()}

    best_cat = max(cat_mean_scores, key=cat_mean_scores.get)
    best_conf = cat_mean_scores[best_cat]

    cat_pct = {cat: round(score * 100, 1) for cat, score in cat_mean_scores.items()}

    return best_cat, best_conf, cat_pct


# ---------------------------------------------------------
# 4. Streamlit App UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="CleanCity AI",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ CleanCity AI – Swachh Bharat Prototype")
st.markdown("### An AI-powered waste classifier & complaint system")

menu = st.sidebar.radio("Navigate", ["Home", "Scan Waste", "File Complaint", "Awareness Hub", "Leaderboard & Rewards"])

# ---------------- Home ----------------
if menu == "Home":
    st.header("Welcome to CleanCity AI")
    st.write("""
    This prototype supports **Swachh Bharat Mission** by helping citizens:
    - 🖼️ Classify waste using **AI (CLIP zero-shot)**
    - 📝 File complaints with auto-attached AI reports
    - 📚 Learn proper disposal practices
    - 🏆 Earn rewards for participation
    """)
    st.info("Use the sidebar to explore features.")

# ---------------- Scan Waste ----------------
elif menu == "Scan Waste":
    st.header("♻️ Waste Scanner (CLIP zero-shot)")

    uploaded_file = st.file_uploader("Upload a waste image (jpg, png)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with st.spinner("Loading CLIP model..."):
            clip_model, clip_processor = load_clip_model_hf(device=device)

        with st.spinner("Classifying with CLIP..."):
            best_cat, best_conf, per_cat_pct = classify_image_clip_hf(image, clip_model, clip_processor, device=device)

        st.success(f"Detected category: **{best_cat.upper()}**")
        st.write("Confidence breakdown:")
        for cat, pct in per_cat_pct.items():
            st.write(f"- {cat}: {pct}%")

        # Disposal tip
        if best_cat == "recyclable":
            st.info("Disposal tip: Put in Dry/Recyclable bin. Find nearest recycling center.")
        elif best_cat == "biodegradable":
            st.info("Disposal tip: Put in Wet/Organic bin or compost.")
        elif best_cat == "hazardous":
            st.warning("Disposal tip: Hazardous waste — handle carefully. Contact local disposal authority.")
        else:
            st.info("Disposal tip: Please check manually or file complaint if uncollected.")

        if st.button("📢 File complaint for this waste"):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            st.session_state["latest_image"] = image_bytes
            st.session_state["latest_category"] = best_cat
            st.session_state["latest_ai_label"] = f"CLIP (HF): {best_cat} ({per_cat_pct.get(best_cat,0)}%)"
            st.success("Ready to file complaint. Go to 'File Complaint' in the sidebar.")

# -------------  --- File Complaint ----------------
elif menu == "File Complaint":
    st.header("📢 File Complaint")

    if "latest_image" in st.session_state:
        st.image(st.session_state["latest_image"], caption="Waste image", width=250)
        st.write(f"AI-detected category: {st.session_state['latest_ai_label']}")

    name = st.text_input("Your Name")
    location = st.text_input("Location")
    complaint_text = st.text_area("Describe the issue")

    if st.button("Submit Complaint"):
        st.success("Complaint submitted successfully ✅ (demo mode)")
        st.info("In production, this would be sent to local authorities.")

# ---------------- Awareness Hub ----------------
elif menu == "Awareness Hub":
    st.header("📚 Awareness Hub")
    st.write("""
    Learn how to dispose waste properly:

    - **Recyclable** → ''_Clean & put in dry bin_''
             
             "Reduce waste today, recycle for a better tomorrow" 
             or 
             "it is the process of converting waste into new, useful products, which in turn reduces waste in landfills, conserves natural resources, saves energy, and creates jobs" 
    - **Biodegradable** → ''_Compost or wet bin_''  
             
             Biodegradable waste includes organic materials like food scraps, garden waste, and paper products that can decompose naturally.
    - **Hazardous** → ''_Batteries, chemicals, e-waste → special handling_''  
             
             "Your waste matters: Handle with heart, for a healthier Earth"
             or
             "Hazardous wastes are wastes with properties that make them dangerous or potentially harmful to human health or the environment. Hazardous wastes can be liquids, solids, contained gases, or sludges"
    - **Other** → ''_Avoid mixing, reduce at source_''  
             
             The Earth is not a bin, don't throw your waste within." Let's embrace waste segregation as a simple, yet powerful way to respect our planet and build a more sustainable future. 

    🌍 Every step helps keep Bharat clean!
    """)

# ---------------- Leaderboard ----------------
elif menu == "Leaderboard & Rewards":
    st.header("🏆 Leaderboard & Rewards (Prototype)")
    st.write("""
    Example leaderboard:
    - Rahul: 12 reports  
    - Priya: 10 reports
    - Aamir: 8 reports  

    ✅ Frequent reporters earn **points & recognition**
    """)
