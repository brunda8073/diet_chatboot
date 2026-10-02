import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pymongo import MongoClient


# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")


# Check API keys
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing in .env file")

if not mongo_uri:
    raise ValueError("MONGODB_URI is missing in .env file")


# Connect to MongoDB
client = MongoClient(mongo_uri)

db = client["chat"]
collection = db["users"]


# Create FastAPI app
app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class ChatRequest(BaseModel):
    user_id: str
    question: str


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a diet specialist, give me the output accordingly."
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "{question}"
        ),
    ]
)


# Groq model
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

chain = prompt | llm


# Get previous chat history
def get_history(user_id: str):
    chats = collection.find(
        {"user_id": user_id}
    ).sort("timestamp", 1)

    history = []

    for chat in chats:
        if chat["role"] == "user":
            history.append(("human", chat["message"]))

        elif chat["role"] == "assistant":
            history.append(("ai", chat["message"]))

    return history


# Home route
@app.get("/")
def home():
    return {
        "message": "Welcome to the Diet Specialist Chatbot API!"
    }


# Chat route
@app.post("/chat")
def chat(request: ChatRequest):

    # Get previous conversation
    history = get_history(request.user_id)

    # Generate response
    response = chain.invoke(
        {
            "history": history,
            "question": request.question
        }
    )

    # Save user message
    collection.insert_one(
        {
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": datetime.now(timezone.utc)
        }
    )

    # Save assistant response
    collection.insert_one(
        {
            "user_id": request.user_id,
            "role": "assistant",
            "message": response.content,
            "timestamp": datetime.now(timezone.utc)
        }
    )

    return {
        "response": response.content
    }