from config import *

from agents.llm_config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from openai import OpenAI

###############################################################
# Groq OpenAI-compatible Client
###############################################################

# Groq exposes an OpenAI-compatible API, so the existing RAG code can
# use the same chat-completions interface without a second provider.
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    timeout=60.0,
)

###############################################################
# Load Embedding Model
###############################################################

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

###############################################################
# Load FAISS Vector DB
###############################################################

vector_db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

###############################################################
# Retrieve Context
###############################################################

def retrieve(question):

    docs = vector_db.similarity_search(
        question,
        k=2
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context, docs

###############################################################
# Build Prompt
###############################################################

def build_prompt(context, question):

    return f"""
You are an AI Hospital Assistant.

Answer ONLY using the document excerpts below.

Do NOT use your own medical knowledge.

If the answer cannot be found in the document excerpts, reply exactly:

"I couldn't find this information in the hospital documentation."

-----------------------------
DOCUMENT EXCERPTS
-----------------------------

{context}

-----------------------------
USER QUESTION
-----------------------------

{question}

-----------------------------
ANSWER
-----------------------------
"""

###############################################################
# Call LLM
###############################################################

def ask_llm(prompt):

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a document question answering assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        print("\nLLM ERROR\n")
        print(e)

        return "Unable to contact the LLM."

###############################################################
# Test
###############################################################

###############################################################
# Chat Loop
###############################################################

if __name__ == "__main__":

    print("=" * 60)
    print("Hospital Assistant")
    print("=" * 60)

    while True:

        question = input("\nYou : ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        # Retrieve relevant context
        context, docs = retrieve(question)

        # Build prompt
        prompt = build_prompt(context, question)

        # Get LLM response
        answer = ask_llm(prompt)

        # Display only the answer
        print("\nAssistant:\n")
        print(answer)