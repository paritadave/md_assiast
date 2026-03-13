import os
import streamlit as st

def get_api_key():
    # 1. Check Streamlit Cloud Secrets (Production)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    
    # 2. Check Conda/System Environment Variables (Local Development)
    local_key = os.getenv("OPENAI_API_KEY")
    if local_key:
        return local_key
    
    return None

api_key = get_api_key()

if not api_key:
    st.error("OpenAI API Key not found. Please set it in your Conda env or Streamlit Secrets.")
#import streamlit as st
#import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="MediScribe AI Prototype", layout="wide", page_icon="🏥")
load_dotenv()

# CSS to make it look clinical and clean
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stHeader { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: Settings ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    model_choice = st.selectbox("LLM Engine", ["gpt-4o", "gpt-4-turbo"])
    st.info("This prototype simulates Biocode's FydoDx workflow for human clinical trials.")

# --- APP HEADER ---
st.title("🏥 MediScribe AI: Ambient Clinical Intelligence")
st.subheader("Transforming messy patient conversations into structured SOAP notes.")

# --- CORE LOGIC ---
def process_clinical_audio(audio_file):
    client = OpenAI(api_key=api_key)
    
    # 1. Transcription Layer
    with st.status("👂 Listening to audio (Whisper V3)..."):
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
    
    # 2. Reasoning Layer (SOAP Extraction)
    with st.status("🧠 Extracting Clinical Entities..."):
        llm = ChatOpenAI(model=model_choice, api_key=api_key, temperature=0)
        
        system_msg = """You are a professional Medical Scribe. 
        Convert the transcript into a formal SOAP note. 
        Use professional terminology (e.g., 'patient reports' instead of 'patient said')."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{transcript}")
        ])
        
        chain = prompt | llm
        soap_note = chain.invoke({"transcript": transcript})
    
    return transcript, soap_note.content

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.header("📂 1. Input")
    uploaded_file = st.file_uploader("Upload Clinic Audio (mp3, wav)", type=["mp3", "wav"])
    
    if uploaded_file and not api_key:
        st.warning("Please enter your API Key in the sidebar.")
    
    if uploaded_file and api_key:
        if st.button("🚀 Generate Clinical Note"):
            raw_text, structured_note = process_clinical_audio(uploaded_file)
            
            st.session_state['transcript'] = raw_text
            st.session_state['note'] = structured_note

with col2:
    st.header("📝 2. Output")
    if 'note' in st.session_state:
        st.success("Analysis Complete!")
        
        tab1, tab2 = st.tabs(["Structured SOAP Note", "Raw Transcript"])
        
        with tab1:
            st.markdown(f"### SOAP Document\n{st.session_state['note']}")
            st.download_button("Download Note", st.session_state['note'], file_name="clinic_note.txt")
        
        with tab2:
            st.text_area("Original Audio Text", st.session_state['transcript'], height=300)
    else:
        st.info("Awaiting input... Upload audio and click generate.")
