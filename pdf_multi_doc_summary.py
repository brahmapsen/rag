print("START multi PDF Summarizer using Hugging Face Embedding")

# https://github.com/samwit/langchain-tutorials/blob/main/embeddings/
# YT_HF_Instructor_Embeddings_Chroma_DB_Multi_Doc_Retriever_LangChain_Part2.ipynb

import chainlit as cl
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
# from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from InstructorEmbedding import INSTRUCTOR

from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

## Cite sources
import textwrap

os.environ["OPENAI_API_KEY"] = "sk-"

documents = []
texts = ''
turbo_llm = None

embedding = OpenAIEmbeddings()
# embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
#                                                   model_kwargs={"device": "cuda"})

# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

def loadFiles():
    loader = DirectoryLoader('./pdfs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print("No of documents read:", len(documents))

    # splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print("Length of Texts:", len(texts))

    print("Persist text in Vector DB local folder")
    vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    vectordb.persist()
    print("Saved to DB")

def createDbFromLocal():
    # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)
    cl.user_session.set("vectordb", vectordb)

## RETRIEVER
def setChainQA():
    vectordb = cl.user_session.get("vectordb")

    print("Set Retriever")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents("Which states are successful with VBC?")
    print("Relevant docs", len(docs))
    print("Search Type:", retriever.search_type)

    print("creating chain to answer questions")

    # Set up the turbo LLM
    # turbo_llm = ChatOpenAI(
    #     temperature=0,
    #     model_name='gpt-3.5-turbo'
    # )

    # create the chain to answer questions 
    # chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
    #                               chain_type="stuff", 
    #                               retriever=retriever, 
    #                               return_source_documents=True)

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
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)
     
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

@cl.action_callback("read directory")
async def on_action(action):
    print("Read Directory")
    loadFiles()
    await cl.Message(content=f" {action.name} completed").send()
    # Optionally remove the action button from the chatbot user interface
    # await action.remove()

@cl.action_callback("read db")
async def on_action(action):
    print("Read DB")
    createDbFromLocal()
    setChainQA()
    await cl.Message(content=f" {action.name} completed").send()
    await cl.Message(content=f" Start asking Questions!!!!").send()
    
@cl.action_callback("set context")
async def on_action(action):
    print("Set context for Retriver")
    await cl.Message(content=f" {action.name} completed").send()

@cl.on_chat_start
async def start():
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="read directory", value="example_value", description="Read directory!"),
        cl.Action(name="read db", value="example_value", description="Read Local Vector DB!"),
        cl.Action(name="set context", value="example_value", description="Set Retriever with Context!")
    ]
    await cl.Message(content="Load files from directory and read DB for question answering:", actions=actions).send()

    text_content = "Hello, this is a text element."
    elements = [
        cl.Text(name="simple_text", content=text_content, display="inline")
    ]

    await cl.Message(
        content="Check out this text element!",
        elements=elements,
    ).send()

@cl.on_message
async def main(message: str):
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