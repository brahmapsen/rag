#Project 1 - cloning Chat GPT UI  #chainlit is like streamlit but for LLM applications
# set HNSWLIB_NO_NATIVE=1

print("START Project #3, PDF Splitter")
import chainlit as cl
import os, sys
import openai

# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from chainlit.types import AskFileResponse
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document

os.environ['OPENAI_API_KEY'] = 'sk-9'
openai.api_key = 'sk-'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to chainlit PDF QA demo! 1. Upload pdf file 2. Ask a question"""

## Have all the chunks and label them as documents
# def process_file(file: AskFileResponse):
def process_file(file):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == 'application/pdf':
        print("PDF FILE:", file.name)
        Loader = PyPDFLoader

    loader = Loader(file.path)
    documents = loader.load()

    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs

# def get_docsearch(file: AskFileResponse):
def get_docsearch(file):
    docs = process_file(file)
    #save data in User session
    cl.user_session.set("docs", docs)
    # Create a unique namespace for the file
    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch

# template = """Question: {question} Answer: Let's think step by step."""
@cl.on_chat_start
async def start():
    #Sending an image with local file path
    await cl.Message(content="You can chat with Pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content = welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb = 20,
            timeout = 180,
        ).send()

    file = files[0]
   
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()