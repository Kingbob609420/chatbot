import streamlit as st
from huggingface_hub import InferenceClient

# Initialize session state variables
def init_session_state():
    if "image_prompt" not in st.session_state:
        st.session_state.image_prompt = ""
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "answer" not in st.session_state:
        st.session_state.answer = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

# Initialize clients
@st.cache_resource
def load_clients():
    return {
        "image_client": InferenceClient(
            provider="hf-inference",
            api_key="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"
        ),
        "chat_client": InferenceClient(
            provider="fireworks-ai",
            api_key="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"
        )
    }

# Function to generate image
def generate_image(prompt, client):
    try:
        image = client.text_to_image(prompt)
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# Function to ask a question to the chatbot
def ask_question(question, conversation_history, client):
    conversation_history.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model="CombinHorizon/zetasepic-abliteratedV2-Qwen2.5-32B-Inst-BaseMerge-TIES",
        messages=conversation_history,
        max_tokens=500
    )
    
    assistant_reply = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, conversation_history

# UI Components and main app (same as before)
# ... [rest of your code remains the same, just remove references to models/tokenizers]
