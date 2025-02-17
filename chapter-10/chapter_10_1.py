import os
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchResults,
    ArxivQueryRun,
)

# Set the USER_AGENT environment variable to avoid warnings
os.environ["USER_AGENT"] = "Chapter_10_Agent/1.0"

# Instantiate the Wikipedia API wrapper and create the tool
wiki_api = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# Define the tools with clear descriptions.
tools = [
    wiki_tool,
    DuckDuckGoSearchResults(),
    ArxivQueryRun(),
    Tool(
        name="Python REPL",
        func=PythonREPL().run,
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    ),
    YahooFinanceNewsTool(),
]

# Define the system prompt with clues on tool selection
system_message = (
    "You are Charlie, a helpful intelligent agent that selects the most appropriate tool for each query and provides the best response."
    "If the query asks for financial analysis (especially stock-related news), use the Yahoo Finance News tool. "
    "For general information, use Wikipedia or DuckDuckGo. "
    "If the query is about scientific papers or research (for example, papers on LLMs), use arXiv. "
    "For executing code or calculations, use the PythonREPL tool. "
    "Always decide based on the context of the user's request."
)

# Initialize the language model.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the agent, passing the system message as part of the agent's configuration.
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_kwargs={"system_message": system_message},
)

# 1. A financial news query:
print("=== Financial News Query ===")
result = agent.invoke("What happened today with Microsoft stocks?")
print("Final Answer:", result)

# 2. A scientific research query:
print("\n=== Scientific Research Query ===")
result = agent.invoke("Find recent research papers on large language models.")
print("Final Answer:", result)

# 3. A code execution query:
print("\n=== Code Execution Query ===")
result = agent.invoke("Run a quick Python snippet to compute 2+2.")
print("Final Answer:", result)
