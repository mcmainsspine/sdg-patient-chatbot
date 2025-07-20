# SDG Spine Surgery Patient Assistant v3.2
# Final Deployment Code for Streamlit Community Cloud

import streamlit as st
import pandas as pd
import datetime
from groq import Groq
import gspread
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SDG Spine Surgery Patient Assistant",
    page_icon="⚕️",
    layout="centered"
)

# --- AUTHENTICATION & CLIENT SETUP ---
# Initialize Groq client from Streamlit Secrets
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    GROQ_API_AVAILABLE = True
except Exception as e:
    st.error("Groq API key is not configured correctly. Please check your app's Secrets settings.")
    GROQ_API_AVAILABLE = False

# Initialize Google Sheets client from Streamlit Secrets
try:
    # Read the JSON content as a single string from secrets
    creds_json_str = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    # Parse the string into a Python dictionary
    creds_dict = json.loads(creds_json_str)
    
    # Manually format the private key to handle newline characters correctly
    creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    
    gc = gspread.service_account_from_dict(creds_dict)
    # Open the Google Sheet by its name
    log_sheet = gc.open("SDG_Chatbot_Log").sheet1
    GSHEETS_AVAILABLE = True
except Exception as e:
    st.error(f"Google Sheets connection failed. Please check your app's Secrets settings and sheet name. Error: {e}")
    GSHEETS_AVAILABLE = False

# --- DATA LOADING ---
@st.cache_data
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['Alternate_Questions'] = df['Alternate_Questions'].fillna('')
        return df
    except FileNotFoundError:
        st.error(f"The protocol file ('combined_protocols.csv') was not found in the GitHub repository.")
        return None

master_df = load_data("combined_protocols.csv")

# --- CORE LOGIC FUNCTIONS ---
def find_relevant_info(user_question, dataframe):
    stop_words = set(['a', 'about', 'an', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'i', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with', 'the', 'my', 'can', 'should', 'do', 'me', 'your'])
    dataframe['Search_Text'] = dataframe['Question'] + ' ' + dataframe['Alternate_Questions']
    query_words = set(user_question.lower().split()) - stop_words
    num_keywords = len(query_words)
    if num_keywords == 0: return None
    best_match_score = 0
    best_match_index = -1
    for index in range(len(dataframe)):
        row = dataframe.iloc[index]
        protocol_words = set(row['Search_Text'].lower().split()) - stop_words
        common_words = query_words.intersection(protocol_words)
        score = len(common_words)
        if score > best_match_score:
            best_match_score = score
            best_match_index = index
    is_match = False
    if num_keywords <= 2 and best_match_score == num_keywords: is_match = True
    elif num_keywords > 2 and best_match_score >= 2: is_match = True
    if is_match:
        relevant_row = dataframe.iloc[best_match_index]
        context = f"--- RELEVANT PROTOCOL INFO ---\nQuestion: {relevant_row['Question']}\nAnswer: {relevant_row['Answer']}\n--- END OF PROTOCOL INFO ---\n"
        return context
    else:
        return None

def create_protocol_prompt(user_question, context):
    return f"You are a helpful, polite, and safe AI assistant for the OrthoIndy spine surgery practice. Your role is to answer patient questions about their upcoming surgery. You must adhere to the following rules STRICTLY:\n1. Base your answer ONLY on the information provided in the 'RELEVANT PROTOCOL INFO' section.\n2. Do NOT use any of your general medical knowledge.\n3. Begin your answer in a friendly and reassuring tone.\n\nPATIENT QUESTION: \"{user_question}\"\n{context}\nPlease provide your answer now."

def create_general_prompt(user_question):
    return f"You are a helpful AI assistant with deep medical knowledge. A patient from the OrthoIndy spine surgery practice has asked a general medical question that is not covered by their surgeon's specific post-operative protocols. Your task is to answer the following question clearly and accurately for a patient.\nCRITICAL RULE: After providing your answer, you MUST include the following disclaimer verbatim (exactly as written) at the end of your response, separated by a line.\n---\n*Disclaimer: This is general medical information and not a substitute for direct medical advice regarding your specific condition. This information is not part of Dr. [Your Name]'s official protocol. For any questions about your personal care plan, please contact the OrthoIndy office directly.*\n\nPATIENT QUESTION: \"{user_question}\"\nPlease provide your answer now, followed by the mandatory disclaimer."

def log_unanswered_question(user_question, surgery_type):
    if not GSHEETS_AVAILABLE: return
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_sheet.append_row([timestamp, surgery_type, user_question])
        st.info("This question has been logged for review.", icon="📝")
    except Exception as e:
        st.warning(f"Could not write to the log file. Error: {e}")

def get_model_response(prompt_text):
    if not GROQ_API_AVAILABLE: return "The AI model is currently unavailable."
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while contacting the AI model: {e}"

# --- STREAMLIT UI ---

st.title("SDG Spine Surgery Patient Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "surgery_type" not in st.session_state:
    st.session_state.surgery_type = None

if st.session_state.surgery_type is None:
    st.info("Welcome! To provide the most accurate information, please select your surgery type below.")
    
    if master_df is not None:
        surgery_options = list(master_df['SurgeryType'].unique())
        selected_surgery = st.selectbox("Select your surgery:", [""] + surgery_options)

        if selected_surgery:
            st.session_state.surgery_type = selected_surgery
            st.session_state.session_df = master_df[master_df['SurgeryType'] == selected_surgery].copy().reset_index(drop=True)
            st.rerun()
    else:
        st.error("Protocol data could not be loaded. The app cannot continue.")

else:
    st.sidebar.title("Options")
    if st.sidebar.button("Change Surgery / Start Over"):
        st.session_state.surgery_type = None
        st.session_state.messages = []
        st.rerun()

    st.success(f"Protocol for **{st.session_state.surgery_type.upper()}** is loaded. How can I help you?")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your surgery..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                protocol_context = find_relevant_info(prompt, st.session_state.session_df)
                
                final_prompt = ""
                if protocol_context:
                    final_prompt = create_protocol_prompt(prompt, protocol_context)
                else:
                    log_unanswered_question(prompt, st.session_state.surgery_type)
                    final_prompt = create_general_prompt(prompt)
                
                response = get_model_response(final_prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
