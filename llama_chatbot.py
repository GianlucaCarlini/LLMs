import streamlit as st
from transformers import pipeline
import logging
import torch
from transformers import AutoTokenizer
import time
from llama_index.core import SimpleDirectoryReader
import os
import tkinter as tk
from tkinter import filedialog
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import Qt

logging.basicConfig(level=logging.INFO)


if "messages" not in st.session_state:
    st.session_state.messages = []


def stream_chat(model, messages):

    try:

        out = model(
            messages,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=1024,
            truncation=True,
        )
        reply = out[0]["generated_text"][-1]

        response = ""
        response_placeholder = st.empty()

        for r in [reply]:
            response += r["content"] + "\n"

            response_placeholder.write(response)

        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response

    except Exception as e:
        logging.error(f"Error: {e}")
        return "I'm sorry, I cannot respond to that."


available_models = {
    "llama3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3-8B-ita": "DeepMount00/Llama-3-8b-Ita",
}


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


def read_file(file):
    reader = SimpleDirectoryReader(file)
    docs = reader.read()
    return docs


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


def main():
    st.title("LLAMA Chatbot")
    logging.info("App started")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    model_id = st.sidebar.selectbox(
        "Choose a model",
        ["llama3.2-1B", "llama3.2-3B", "llama3-8B-ita"],
    )

    model_id = available_models[model_id]

    llama = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    st.sidebar.title("Read Files")
    click_folder = st.sidebar.button("Select Folder")
    click_file = st.sidebar.button("Select File")

    # if clicked execute the function
    if click_folder:
        dir_path = get_dir_path()

    if click_file:
        file_path = get_file_path()

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
                        messages = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages
                        ]
                        response_message = stream_chat(llama, messages)
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
    main()
