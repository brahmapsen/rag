# Agent Search in UI
# Project 3 - GPT Researcher using arxiv - ReACT architecture -Reasoning and ACT - Browse internet
print("START Project #3, Agent Search in UI")
import chainlit as cl
import os
import openai

from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor, load_tools, AgentType
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'sk-'
openai.api_key = 'sk-'

@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    
    tools = load_tools(
        ["arxiv"]
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=3,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  ### IMPORTANT
    )

    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])