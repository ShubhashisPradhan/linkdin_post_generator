from dotenv import load_dotenv
from langchain_groq import ChatGroq 
import os

load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model="meta-llama/llama-4-scout-17b-16e-instruct")


if __name__ == "__main__":
    print(llm.invoke("What is the capital of Odisha?").content)

