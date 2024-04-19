# Project 3 - GPT Researcher using arxiv - ReACT architecture -Reasoning and ACT - Browse internet

print("START Project #3, GPT Researcher - arxiv")
import chainlit as cl
import os
import openai

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
os.environ['OPENAI_API_KEY'] = 'sk-'
openai.api_key = 'sk-'

llm = ChatOpenAI(temperature=0.3)
tools = load_tools(
    ["arxiv"]
)

# This is a Agent chain - unites prompts and LLMs and algorithms
agent_chain = initialize_agent(
    tools,
    llm,
    max_iterations=5, #allows not to get stuck in a loop
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  ## this algorithm allows the LLM to think
    verbose=True,
    handle_parsing_errors=True, ##this is important
)

agent_chain.run(
    "what is RLHF?",
)


