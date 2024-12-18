import streamlit as st
from transformers import pipeline
import logging
import torch
from transformers import AutoTokenizer
import time
import os
import tkinter as tk
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import Qt
from langchain import document_loaders as dl
from langchain_core.documents.base import Document
from langchain import embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import text_splitter as ts
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
import torch
import faiss
from typing import Annotated, Sequence, Literal, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Image, display
import os
from langchain import hub
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, create_react_agent
import json
import numexpr
import pprint
import math
from clustering import cluster_documents, label_clusters, plot_clusters

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)

if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


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
        chunk_size=1000, chunk_overlap=200
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


def main(workflow, memory, config):

    st.title("LLama Chatbot")
    logging.info("Starting LLama Chatbot")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    click_folder = st.sidebar.button("Select Folder")
    click_file = st.sidebar.button("Select File")
    click_cluster = st.sidebar.button("Cluster Documents")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("Hello world!")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    splits = None

    if click_folder:
        dir_path = get_dir_path()
        splits = read_folder(dir_path)
        print("qua")

    if click_file:
        file_path = get_file_path()
        splits = read_file(file_path)
        print("qui")

    if click_cluster:
        folder_path = get_dir_path()
        df = cluster_documents(folder_path)
        df = label_clusters(df)
        fig = plot_clusters(df)

        st.pyplot(fig)

    if splits is not None:
        _ = vector_store.add_documents(splits)
        print("ecchime")

    ret = vector_store.as_retriever(search_kwargs={"k": 5})

    print(ret.invoke("Hello world!"))

    # TOOLS

    @tool
    def retriever(query: str):
        """Retrieve information related to a query.

        Args:
            query: The query to search for.

        """
        retrieved_docs = ret.invoke(query)

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

    # AGENT

    def call_tools(msg):
        tool_map = {tool.name: tool for tool in tools}

        tool_calls = msg.tool_calls.copy()

        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])

        return tool_calls

    def agent(state):

        print("---AGENT INVOKED---")

        model = ChatOllama(model="llama3-groq-tool-use:8b-fp16", temperature=0)
        model_with_tools = model.bind_tools(tools)

        user_question = state["messages"][-1].content

        out = model_with_tools.invoke(user_question)

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

        model = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0)

        user_question = state["user_question"]
        context = state["messages"][-1].tool_calls[0]["output"]

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        formatted_docs = format_docs(context)

        prompt = PromptTemplate(
            template="""You are an helpful assistant that can provide information to the user
            based on the retrieved documents. \n
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            Provide a concise answer to the question using no more than 3 sentences. \n
            If context is empty, ask the user to provide some documents. \n""",
            input_variables=["context", "question"],
        )

        chain = prompt | model

        out = chain.invoke({"context": formatted_docs, "question": user_question})

        return {"messages": [out], "user_question": user_question, "context": context}

    # GRAPH

    workflow.add_node("retriever_node", retriever_node)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router)
    workflow.add_edge("retriever_node", END)

    graph = workflow.compile(checkpointer=memory)

    with st.chat_message("assistant"):
        st.write("How may I assist you today?")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):

                start = time.time()
                logging.info("Generating response...")

                with st.spinner("Writing..."):

                    try:
                        messages = {
                            "messages": [
                                HumanMessage(
                                    content=st.session_state.messages[-1]["content"]
                                ),
                            ]
                        }
                        response_message = stream_chat(graph, messages, config)
                        duration = time.time() - start
                        response_message_with_duration = f"{response_message}\n\nResponse time: {duration:.2f} seconds"
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_message_with_duration,
                            }
                        )
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(
                            f"Response: {response_message}, Duration: {duration:.2f} seconds"
                        )

                    except Exception as e:
                        print(e)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "I'm sorry, I cannot respond to that.",
                            }
                        )
                        logging.error(f"Error: {e}")


if __name__ == "__main__":

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        context: List[Document]
        user_question: str

    workflow = StateGraph(AgentState)

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "abc123"}}
    main(workflow=workflow, memory=memory, config=config)
