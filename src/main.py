""" FastAPI server handling user interactions and LLM processing. """

import requests
import json
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer 
from langchain.schema import Document
from src.bot_llm import LlamaInstruct

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = LlamaInstruct()

VECTOR_DB_API_URL = "http://34.27.92.10:8000/query"  # Remote vector DB API


def retrieve_relevant_docs(query: str, top_k: int = 5):
    """
    Queries the remote vector database for the most relevant documents.
    Ensures the request body is correctly formatted.
    """
    query_embedding = embedding_model.encode(query).tolist()  # Convert to list of floats

    payload = {
        "query_vector": query_embedding,  # Make sure this is a list of floats
        "top_k": top_k 
    }

    print("Sending request to VectorDB API with payload:", json.dumps(payload, indent=2))

    try:
        response = requests.post(VECTOR_DB_API_URL, json=payload)
        response.raise_for_status() 
        data = response.json()
        return [{"url": doc} for doc in data.get("results", [])]
    
    except requests.exceptions.RequestException as e:
        print(f"Error querying vector database: {e}")
        return []


@app.get("/", response_class=HTMLResponse)
async def server_form():
    """ Serve the user interface form. """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ask a Question</title>
    </head>
    <body>
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                let textarea = document.getElementById("question");
                let form = document.getElementById("questionForm");

                textarea.addEventListener("keypress", function(event) {
                    if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();  
                        form.submit(); 
                    }
                });
            });
        </script>

        <h1>Ask Scrappy a Question! </h1>
        <form id="questionForm" action="/ask" method="post">  <!-- Added ID -->
            <label for="question">Your Question:</label><br><br>
            <textarea id="question" name="question" rows="4" cols="50"></textarea><br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/ask")
async def model_response(question: str = Form(...)):
    """ Queries the vector DB and serves response to user. """

    query_vector = embedding_model.encode(question).tolist()  
    query_payload = {"query_vector": query_vector, "top_k": 3}

    print("\n========= SENDING QUERY TO VECTOR DB =========")
    print(query_payload)
    print("==============================================\n")

    try:
        response = requests.post(VECTOR_DB_API_URL, json=query_payload, timeout=120)  # 🔹 Add timeout
        print("\n========= VECTOR DB RESPONSE RECEIVED =========")
        print(f"Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        print("==============================================\n")

    except requests.exceptions.Timeout:
        print("\n⏳ Timeout: The request to the VectorDB took too long and was aborted.\n")
        return "The request took too long. Please try again later."

    except requests.exceptions.ConnectionError:
        print("\n🚨 Error: Could not connect to the VectorDB API. Is it running?\n")
        return "VectorDB API is unavailable."

    except requests.exceptions.RequestException as e:
        print(f"\n🚨 Unexpected Error: {e}\n")
        return "An error occurred while querying the VectorDB."

    if response.status_code != 200:
        return f"Error querying vector database: {response.text}"

    results = response.json()

    relevant_docs = [Document(page_content=doc["url"]) for doc in results.get("results", [])]

    scrappy_response = llm.generate(question, relevant_docs)
    scrappy_answer = scrappy_response.get("answer")

    # ✅ Ensure correct HTML rendering for code snippets
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Response</title>
        <style>
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            code {{
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <h1>Your Question:</h1>
        <p>{question}</p>

        <h1>LLM's Answer:</h1>
        <p>{scrappy_answer}</p>

        <h1>Relevant Documents:</h1>
        <ul>
            {"".join([f'<li><a href="{doc.page_content}" target="_blank">{doc.page_content}</a></li>' for doc in relevant_docs])}
        </ul>

        <a href="/">Ask Another Question</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

