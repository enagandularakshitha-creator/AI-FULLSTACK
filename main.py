import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Campus Event AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Campus Event Recommendation System")
st.write("Find campus events based on your interests using AI.")

# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# ChromaDB
# -----------------------------
@st.cache_resource
def create_database():
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="campus_events"
    )

    return collection

collection = create_database()

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Add PDF to Database
# -----------------------------
def add_pdf_to_database(text):

    # Split text into smaller chunks
    chunks = []

    words = text.split()

    chunk_size = 150

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        collection.upsert(
            ids=[f"event_{i}"],
            documents=[chunk],
            embeddings=[embedding]
        )

    return len(chunks)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📂 Event Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Campus Event PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.sidebar.button("Add Event Data"):

        with st.spinner("Processing PDF..."):

            text = extract_pdf_text(
                uploaded_file
            )

            if text.strip():

                count = add_pdf_to_database(text)

                st.success(
                    f"Added {count} event information chunks!"
                )

            else:

                st.error(
                    "Could not extract text from this PDF."
                )


# -----------------------------
# Student Preferences
# -----------------------------
st.header("👩‍🎓 Student Preferences")

interest = st.text_input(
    "What are you interested in?",
    placeholder="Example: AI, Data Science, Coding, Robotics"
)

department = st.selectbox(
    "Select your department",
    [
        "CSE",
        "CSD",
        "CSM",
        "ECE",
        "EEE",
        "Other"
    ]
)

year = st.selectbox(
    "Select your year",
    [
        "1st Year",
        "2nd Year",
        "3rd Year",
        "4th Year"
    ]
)


# -----------------------------
# AI Recommendation
# -----------------------------
if st.button("🔍 Recommend Events"):

    if not interest:

        st.warning(
            "Please enter your interests."
        )

    else:

        with st.spinner(
            "AI is finding suitable events..."
        ):

            query = f"""
            Student interests: {interest}
            Department: {department}
            Year: {year}
            """

            query_embedding = embedding_model.encode(
                query
            ).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            documents = results.get(
                "documents",
                [[]]
            )[0]

            if documents:

                context = "\n\n".join(
                    documents
                )

                prompt = f"""
                You are an AI campus event recommendation assistant.

                Student information:
                {query}

                Available campus event information:
                {context}

                Recommend the most suitable events for this student.

                For each recommendation provide:
                1. Event name
                2. Why it is suitable
                3. Skills they can learn
                4. Who should attend

                Give a simple and clear answer.
                """

                try:

                    response = ollama.chat(
                        model="llama3.2",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = response["message"]["content"]

                    st.subheader(
                        "🤖 AI Recommendations"
                    )

                    st.write(answer)

                except Exception as e:

                    st.error(
                        "Could not connect to Ollama."
                    )

                    st.info(
                        "Make sure Ollama is running "
                        "and the llama3.2 model is installed."
                    )

            else:

                st.warning(
                    "No event data found. "
                    "Please upload event information first."
                )


# -----------------------------
# Database Information
# -----------------------------
st.divider()

st.subheader("📊 System Information")

try:

    count = collection.count()

    st.write(
        f"Event information stored: **{count} chunks**"
    )

except:

    st.write(
        "No event data available yet."
    )