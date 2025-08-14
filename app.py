from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import initialize_llm, setup_qa_chain, create_vector_db
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

app=Flask(__name__)
CORS(app)

llm=initialize_llm()
db_path=r"C:\Users\hp\Desktop\ai chatbot\chroma_db"

if not os.path.exists(db_path):
    vector_db=create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db=Chroma(persist_directory=db_path, embedding_function=embeddings)
qa_chain=setup_qa_chain(vector_db,llm)
@app.route("/chat",methods=["POST"])
def chat():
    user_input=request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a message"}),400
    response=qa_chain.invoke(user_input)
    return jsonify({"response": response})
if __name__ == "__main__":
    app.run(debug=True)
    