import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA

load_dotenv()

df = pd.read_csv(r"C:\Users\emrek\Desktop\project\London_hotel_reviews.csv", encoding="latin-1")
df = df.dropna(subset=["Review Text", "Property Name"])
df = df.sample(500, random_state=42)


docs = []
for _, row in df.iterrows():
    content = f"Hotel: {row['Property Name']}\nRating: {row['Review Rating']}\nReview: {row['Review Text']}"
    docs.append(Document(page_content=content))

print(f"Loaded {len(docs)} reviews")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

print("\nLondon Hotel Assistant - ask me anything! (type 'quit' to exit)\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = qa_chain.invoke(question)
    print(f"\nBot: {answer['result']}\n")
