#Project 1 - cloning Chat GPT UI  #chainlit is like streamlit but for LLM applications
# set HNSWLIB_NO_NATIVE=1
print("START Project #3, PDF Splitter")
import chainlit as cl
import os, sys
import openai

# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from chainlit.types import AskFileResponse

os.environ['OPENAI_API_KEY'] = 'sk-'
openai.api_key = 'sk-'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to chainlit PDF QA demo! 1. Upload pdf file 2. Ask a question"""

## Have all teh chunks and label them as documents
# def process_file(file: AskFileResponse):
def process_file(file):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == 'application/pdf':
        print("PDF FILE:", file.name)
        Loader = PyPDFLoader

    print("File path:", file.path)

    with tempfile.NamedTemporaryFile() as tempfile:
        # tempfile.write(file.path)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>about to load tempfile")
        # loader = Loader(tempfile.name)
        loader = Loader(file.path)
        documents = loader.load()

        print("#########About to split doc")
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs

    # with open('filename.ext', 'wb') as file_output:
    #     file_output.write(file.content)
    #     loader = Loader('filename.ext')
    #     documents = loader.load()
    #     docs = text_splitter.split_documents(documents)
    #     for i, doc in enumerate(docs):
    #         doc.metadata["source"] = f"source_{i}"
    #     return docs
    
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
   
    msg = cl.Message(content=f"Processing  `{file.name}`....")
    await msg.send()

    print("Call get_docsearch")
    docsearch = await cl.make_async(get_docsearch)(file)

    #Chains unite model and prompt together, 
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4897),
    )
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()
    # Save the chain with User Session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    #Provide citation of what is happening
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    #get docs from user session
    docs = cl.user_session.get("docs")
    metadatas = [docs.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]   

    # if sources:
    #     found_sources = []

    #     for source in sources.split(","):
    #         sources_name = source.strip().replace(".","")
    #         try:
    #             index = all_sources.index(sources_name)
    #         except ValueError:
    #             continue
    #         text = docs[index].page_content
    #         found_sources.append(source_name)
    #         source_elements.append(cl.Text(content=text, name=source_name))
        
    #     if found_sources:
    #         answer += f"\nSources: {', '.join(found_source)}"
    #     else:
    #         answer += "\n No sources found"
    
    answer += "\n No sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream_elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
#"END Project #2, PDF Splitter"