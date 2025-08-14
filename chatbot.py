from langchain_groq import ChatGroq  # if you have this working
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def initialize_llm():
    llm=ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
)
    return llm
def create_vector_db():
    loader = DirectoryLoader(r"C:\Users\hp\Desktop\ai chatbot\data", glob='*.pdf', loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
    texts=text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db=Chroma.from_documents(texts,embeddings,persist_directory='./chroma_db')
    vector_db.persist()

    print("ChromaDB created and data saved")

    return vector_db
def setup_qa_chain(vector_db,llm):
    retriever=vector_db.as_retriever()
    prompt_template="""You are a compassionate mental health chatbot. Respond thoughtfully to the following question 
     {context}
     User : {question}
     Chatbot """
    PROMPT=PromptTemplate(template=prompt_template, input_variables=['context','questions'])

    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt":PROMPT}

    )
    return qa_chain
def main():
    print("Initializing chatbot.....")
    llm=initialize_llm()

    db_path = r"C:\Users\hp\Desktop\ai chatbot\chroma_db"


    if not os.path.exists(db_path):
        vector_db = create_vector_db()
    else:
         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
         vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    qa_chain=setup_qa_chain(vector_db, llm)

    while True:
        query=input("\nHuman: ")
        if query.lower() =="exit":
            print("Chatbot: Take Care of Yourself, GoodBye!")
            break
        response = qa_chain.invoke(query)

        print(f"Chatbot: {response}")
    
if __name__ == "__main__":
    main()