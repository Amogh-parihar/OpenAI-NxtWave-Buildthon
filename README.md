# OpenAI-NxtWave-Buildthon
# CleanCity AI ♻️

An AI-powered prototype for **__Swachh Bharat Mission__**.

## 📌 Problem Statement  
Urban and rural areas in India often face waste management challenges. Despite the **Swachh Bharat Mission**, monitoring street-level cleanliness is difficult:  
- Manual inspections are **time-consuming and costly**.  
- Citizens have **no simple tool** to report cleanliness issues.  
- Municipalities lack **real-time data** for cleanliness drives.  

---

## 💡 Our Solution – CleanCity-AI  
CleanCity-AI is a **Gen-AI powered web app** that allows citizens to upload street/area photos and instantly get feedback on whether the area is clean or dirty.  


### 🔹 Key Features  
- Upload an image of any street/area.  
- AI model (OpenAI CLIP) classifies → **Clean ✅ / Dirty ❌**.  
- Awareness messages + motivational quotes (aligned with Swachh Bharat Abhiyan).  
- Can be extended for **municipal dashboards**.  

---

## 🤖 Tech Stack  
- **Frontend & Backend**: [Streamlit](https://streamlit.io/)  
- **AI Model**: OpenAI **CLIP (Zero-Shot Classifier)**  
- **Awareness Messages**: OpenAI GPT Models (optional)  
- **Deployment**: Streamlit Cloud + GitHub

-----------------------------

### Create a virtual environment:

python -m venv venv
source venv/Scripts/activate  

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

### DEMO 

👉 Deployed App: https://cleancity-ai.streamlit.app/
