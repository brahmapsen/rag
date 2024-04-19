#Project 1 - cloning Chat GPT UI #chainlit is like streamlit but for LLM applications
print("START Project #1, Cloning Chat GPT UI")
import chainlit as cl
import openai
import os

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
os.environ['OPENAI_API_KEY'] = 'sk-'
openai.api_key = 'sk-'

template = """Question: {question} Answer: Let's think step by step."""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template = template, input_variables =["question"])
    llm_chain = LLMChain(   ###LLM Chain connects prompt with LLM
        prompt = prompt,
        llm = OpenAI(temperature = 1, streaming = True),
        verbose = True,
    )
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks = [cl.AsyncLangchainCallbackHandler()])
    await cl.Message(res["text"]).send()

    # response = openai.ChatCompletion.create(
    #     model= 'gpt-3.5-turbo',
    #     messages = [
    #         {"role": "assistant", "content": "you are a helpful assistant."},
    #         {"role": "user", "content": message}
    #     ],
    #     temperature = 1,
    # )
    # await cl.Message(content=f"{response['choices'][0]['message']['content']}", ).send()
