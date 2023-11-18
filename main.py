import os
import os.path
import openai
import gradio as gr
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from theme import CustomTheme

def initialize():
    if not os.path.exists("./storage"):
        # load the documents and create the index
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist()

def response(message, history):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    answer = query_engine.query(message)

    return str(answer)


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    initialize()

    custom_theme = CustomTheme()

    chatbot = gr.ChatInterface(
        fn=response,
        retry_btn=None,
        undo_btn=None,
        theme=custom_theme,
    )

    chatbot.launch(inbrowser=True)


if __name__ == "__main__":
    main()
