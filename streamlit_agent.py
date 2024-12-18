import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import Qt
from langchain import document_loaders as dl
from langchain_core.documents.base import Document
from langchain import text_splitter as ts
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
import torch
import faiss
from typing import Annotated, Sequence, Literal, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables.history import RunnableWithMessageHistory
import numexpr
import math
import logging
import time
import pandas as pd
from clustering import cluster_documents, label_clusters, plot_clusters


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦙")
st.title("🦙 LLama Chat with documents")

if "messages" not in st.session_state:
    st.session_state.messages = []


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)

st.session_state.references = None


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


avatars = {"human": "user", "ai": "assistant"}


def get_file_path():
    # Create a temporary main window to act as the parent
    app = QApplication([])
    temp_window = QMainWindow()
    temp_window.setWindowFlags(
        temp_window.windowFlags() | Qt.WindowStaysOnTopHint
    )  # Ensure it stays on top
    temp_window.show()  # Make sure the parent window is initialized

    # Open the QFileDialog with the temporary main window as parent
    file_name, _ = QFileDialog.getOpenFileName(temp_window, "Select File")
    temp_window.close()  # Close the temporary window after selection

    # Display the selected file
    st.text_input("Selected File", file_name)

    return file_name


def get_dir_path():
    # Create a temporary main window to act as the parent
    app = QApplication([])
    temp_window = QMainWindow()
    temp_window.setWindowFlags(
        temp_window.windowFlags() | Qt.WindowStaysOnTopHint
    )  # Ensure it stays on top
    temp_window.show()  # Make sure the parent window is initialized

    # Open the QFileDialog with the temporary main window as parent
    dir_name = QFileDialog.getExistingDirectory(temp_window, "Select Folder")
    temp_window.close()  # Close the temporary window after selection

    # Display the selected folder
    st.text_input("Selected Folder", dir_name)

    return dir_name


def read_file(file):
    loader = dl.PyPDFLoader(file)

    doc = loader.load()
    text_splitter = ts.RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    )

    all_splits = text_splitter.split_documents(doc)

    return all_splits


def read_folder(folder):
    loader = dl.DirectoryLoader(
        folder, glob="*.pdf", loader_cls=dl.PyPDFLoader, show_progress=True
    )
    doc = loader.load()

    text_splitter = ts.RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )

    all_splits = text_splitter.split_documents(doc)

    return all_splits


def start_retriever(docs=None):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("Hello world!")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    if docs is not None:
        _ = vector_store.add_documents(docs)

    retriever = vector_store.as_retriever(search_kwargs={"k": st.session_state.k})

    return retriever


with st.sidebar:
    st.session_state.k = st.slider("Number of documents to retrieve", 1, 10, 5)

if "retriever" not in st.session_state:
    st.session_state.retriever = start_retriever()


click_folder = st.sidebar.button("Select Folder")
click_file = st.sidebar.button("Select File")
click_cluster = st.sidebar.button("Cluster Documents")

if click_folder:
    with st.spinner("I'm reading your documents, this may take a while..."):
        folder = get_dir_path()
        docs = read_folder(folder)
        st.session_state.retriever = start_retriever(docs)

if click_file:
    with st.spinner("I'm reading your document, this may take a while..."):
        file = get_file_path()
        docs = read_file(file)
        st.session_state.retriever = start_retriever(docs)

if click_cluster:
    folder_path = get_dir_path()
    st.chat_message("assistant").write(
        "I'm embedding your documents, this may take a while..."
    )
    df = cluster_documents(folder_path)
    st.chat_message("assistant").write(
        "I'm reading the clustered documents to assign a label..."
    )
    df = label_clusters(df)
    st.chat_message("assistant").write("Time to display a nice plot")
    fig = plot_clusters(df)

    st.pyplot(fig)


@tool
def retriever(query: str):
    """Retrieve information related to a query.

    Args:
        query: The query to search for.

    """
    retrieved_docs = st.session_state.retriever.invoke(query)

    return retrieved_docs


@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**5" for "37593^5"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )


tools = [retriever, calculator]


def call_tools(msg):
    tool_map = {tool.name: tool for tool in tools}

    tool_calls = msg.tool_calls.copy()

    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])

    return tool_calls


def agent(state):

    print("---AGENT INVOKED---")

    model = ChatOllama(
        model="llama3-groq-tool-use:8b-fp16",
        temperature=0,
        streaming=True,
    )
    model_with_tools = model.bind_tools(tools)

    messages = state["messages"]

    user_question = state["messages"][-1].content

    out = model_with_tools.invoke(messages)

    out.tool_calls = call_tools(out)

    return {"messages": [out], "user_question": user_question}


def router(state):
    print("---ROUTER INVOKED---")

    message = state["messages"][-1]

    if message.tool_calls:
        name = message.tool_calls[0]["name"]

        if name == "retriever":
            return f"{name}_node"
        else:
            return END
    else:
        return END


def retriever_node(state):
    print("---RETRIEVE NODE INVOKED---")

    model = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0, streaming=True)

    user_question = state["user_question"]
    messages = state["messages"]
    context = state["messages"][-1].tool_calls[0]["output"]

    if not context:
        out = """ Looks like you didn't provide any documents to retrieve information from. \n
        Please, provide some documents using the buttons on the right."""
        return {
            "messages": [AIMessage(content=out)],
            "user_question": user_question,
            "context": context,
        }

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    formatted_docs = format_docs(context)

    prompt = PromptTemplate(
        template="""You are an helpful assistant that can provide information to the user
        based on the retrieved documents. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the conversation so far: {question} \n """,
        # Provide a concise answer to the question using no more than 3 sentences. \n""",
        # If context is empty, ask the user to provide you the needed information. \n""",
        input_variables=["context", "question"],
    )

    chain = prompt | model

    out = chain.invoke({"context": formatted_docs, "question": messages})

    references = {}

    for doc in context:
        title = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_number = doc.metadata.get("page", "Unknown")

        if title in list(references.keys()):
            references[title] += f", {page_number}"
        else:
            references[title] = f"{page_number}"

    ref_df = pd.DataFrame.from_dict(references, orient="index", columns=["Pages"])
    st.table(ref_df)

    return {"messages": [out], "user_question": user_question, "context": context}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: List[Document]
    user_question: str


workflow = StateGraph(AgentState)

workflow.add_node("retriever_node", retriever_node)
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router)
workflow.add_edge("retriever_node", END)

graph = workflow.compile()


def stream_chat(graph, messages, config):

    try:

        out = graph.invoke(messages, config=config)
        reply = out["messages"][-1].content

        response = ""
        response_placeholder = st.empty()

        for r in [reply]:
            response += r + "\n"

            response_placeholder.write(response)

        logging.info(
            f"Model: {graph}, Messages: {messages}, Response: {response}, Full Response: {out}"
        )
        return response

    except Exception as e:
        logging.error(f"Error: {e}")
        return "I'm sorry, I cannot respond to that."


if st.session_state.references is not None:
    with st.sidebar:
        st.table(st.session_state.references)


def main():

    with st.chat_message("assistant"):
        st.write("How may I assist you today?")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Writing..."):
                msgs = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                messages = {
                    "messages": msgs,
                }
                config = {
                    "configurable": {"thread_id": "abc123", "session_id": "any"},
                    # "callbacks": [stream_handler],
                }
                response_message = stream_chat(graph, messages, config)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                )


if __name__ == "__main__":
    main()
