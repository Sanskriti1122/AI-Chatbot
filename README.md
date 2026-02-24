# 🧠 Mental Health AI Chatbot (RAG-Based)

An intelligent Mental Health Support Chatbot built using Retrieval-Augmented Generation (RAG), LangChain, HuggingFace embeddings, ChromaDB, and Groq LLM API.

This system provides context-aware, empathetic responses by grounding LLM outputs in relevant knowledge retrieved from a vector database.

---

## 📌 Project Overview

This chatbot is designed to:

- Provide emotionally supportive responses
- Retrieve relevant contextual knowledge before generating answers
- Reduce hallucination using RAG architecture
- Deliver low-latency responses using Groq API

Unlike standard chatbots, this system does not rely purely on a language model — it first retrieves relevant information from a vector database and then generates a response grounded in that data.

---

## 🏗️ System Architecture

1. User enters a query  
2. Query is converted into embeddings (HuggingFace)  
3. Embedding is compared with stored document embeddings in ChromaDB  
4. Relevant context is retrieved  
5. Retrieved context + user query are sent to Groq LLM  
6. LLM generates context-aware response  
7. Response is displayed via web interface  

This architecture ensures higher factual reliability and better response quality.

---

## 🛠️ Tech Stack

- Python
- LangChain
- HuggingFace Embeddings
- ChromaDB (Vector Database)
- Groq LLM API
- Flask / Streamlit (depending on your frontend setup)
- HTML/CSS (if web-based UI)

---

## 🧠 Core Concepts Used

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Similarity Search
- Prompt Engineering
- Context Grounding
- LLM Integration

---

## 🎯 Key Features

- Context-aware mental health responses
- Reduced hallucination compared to standalone LLM
- Modular backend architecture
- Scalable document retrieval system
- Fast inference using Groq API
- Simple and clean UI interface

---

## ⚠️ Limitations

- Not a substitute for professional mental health care
- No crisis detection module (yet)
- Limited personalization memory
- Performance depends on quality of embedded documents

---

## 🚀 Future Improvements

- Add sentiment analysis layer
- Implement crisis detection trigger system
- Add long-term user memory
- Improve personalization
- Deploy using cloud services (AWS/GCP)
- Add secure authentication system

---

## 📂 Project Structure

```
app.py          # Main web application
chatbot.py      # RAG logic and LLM integration
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/mental-health-chatbot.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Add your API keys in environment variables:

```
GROQ_API_KEY=your_api_key
```

4. Run the application:

```
python app.py
```

5. Open in browser:
```
http://localhost:5000
```

---

## 💡 Why RAG Instead of Fine-Tuning?

- Reduces hallucination
- Allows dynamic knowledge updates
- No need for expensive retraining
- Better scalability
- Easier document management

---

## 🧠 Learning Outcomes

- Built full-stack AI system integrating LLM + Vector DB
- Understood embedding generation and similarity search
- Designed scalable AI architecture
- Implemented real-world AI application

---

## 👩‍💻 Author

Sanskriti Sharma  
Computer Science & Data Science Student  
Interested in AI Systems, NLP, and Intelligent Assistive Technologies
