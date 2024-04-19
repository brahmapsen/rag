#https://python.langchain.com/docs/modules/agents/tools/multi_input_tool
#install numpy package in my environment

from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor, load_tools, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import ShellTool
import chainlit as cl
import os

os.environ['OPENAI_API_KEY'] = 'sk-'

shell_tool = ShellTool()

@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0)

    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")

    agent = initialize_agent(
        [shell_tool],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors = True
    )
    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])