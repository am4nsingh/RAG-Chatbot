import streamlit as st
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

st.set_page_config(page_title="Conversational RAG Chatbot")
st.header("Conversational :blue[RAG Chatbot]")
st.chat_message("ai").write("Hello! I'm here to answer your questions")

persist_directory = r"C:\Users\amans\Desktop\Chatbot_project\faq_vdb"
embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en-v1.5")

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

chat_model = ChatOllama(model="llava:7b",
                     base_url="http://127.0.0.1:11434",
                     temperature=0,
                     num_predict=512
                     )

prompt = ChatPromptTemplate([
    ("system", "Strictly only use the following context to answer the user's question. "
               "If the answer is not in the context, say 'I don't know based on the provided document. "
               "Check the type a context belongs to and match the ques with it to get answer from right context"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("system", "Context:\n{context}")])

retriever = vectordb.as_retriever(search_kwargs = {"k":1})

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
chat_history = st.session_state["chat_history"]

for c in chat_history:
    if "HumanMessage" in str(type(c)):
        st.chat_message("human").write(c.content)
    elif "AIMessage" in str(type(c)):
        st.chat_message("ai").write(c.content)
def run():
    question = st.chat_input("Type your query here...")
   
    # st.chat_message("user").write("How are you ?")
    context = ""
    if question:
        st.chat_message("human").write(question)
        chat_history.append(HumanMessage(content=question))
        results = retriever.invoke(question)
        for i, r in enumerate(results):
            section = r.metadata["type"]
            content = r.page_content
            context += f"TYPE {i+1}: {section} \n\n CONTEXT {i+1}:{content} \n\n"
        chain = prompt | chat_model | StrOutputParser()
        response = chain.invoke({"question": question, "context": context, "chat_history": chat_history})
        st.chat_message("ai").write(response)
        chat_history.append(AIMessage(content=response))
if __name__ == "__main__":
    run()