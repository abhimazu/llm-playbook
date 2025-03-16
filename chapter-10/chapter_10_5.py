import os
import requests
import shutil

from langchain.agents import initialize_agent, Tool
# For demonstration, we’re using OpenAI as a stand-in.
# Replace with the appropriate Together-AI LLM integration if available.
from langchain.llms import OpenAI

# Set your Together-AI API key
os.environ["TOGETHER_AI_API_KEY"] = "YOUR_TOGETHER_AI_API_KEY_HERE"

# -------------------------
# Define the custom tool functions
# -------------------------

def wikipedia_query(query: str) -> str:
    """
    Tool 1: Query the Wikipedia API.
    Given a topic (query), fetch its summary from Wikipedia.
    """
    # Wikipedia expects the title to be URL-friendly
    title = query.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "No summary available.")
    else:
        return f"Error: Unable to retrieve data for '{query}' from Wikipedia."

def arxiv_query(query: str) -> str:
    """
    Tool 2: Query the ArXiv API.
    Returns raw XML results for research papers matching the query.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 3  # limiting results for brevity
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: Unable to query ArXiv for '{query}'."

def celsius_to_fahrenheit(celsius: float) -> str:
    """
    Tool 3: Convert Celsius to Fahrenheit.
    """
    fahrenheit = (celsius * 9/5) + 32
    return f"{celsius}°C is {fahrenheit}°F."

def save_text_to_file(params: str) -> str:
    """
    Tool 4: Save text to a file.
    Expects input in the format: file_name||content
    """
    try:
        file_name, content = params.split("||", 1)
        with open(file_name.strip(), "w", encoding="utf-8") as f:
            f.write(content.strip())
        return f"Content saved to {file_name.strip()}."
    except Exception as e:
        return f"Error saving file: {str(e)}"

def copy_file(params: str) -> str:
    """
    Tool 5: Copy a file.
    Expects input in the format: source_path||destination_path
    """
    try:
        source_path, destination_path = params.split("||", 1)
        shutil.copyfile(source_path.strip(), destination_path.strip())
        return f"File copied from {source_path.strip()} to {destination_path.strip()}."
    except Exception as e:
        return f"Error copying file: {str(e)}"

# -------------------------
# Wrap the functions as LangChain Tools
# -------------------------

tools = [
    Tool(
        name="wikipedia_query",
        func=wikipedia_query,
        description=(
            "Queries the Wikipedia API to retrieve article summaries. "
            "Input should be the topic title, e.g., 'Artificial Intelligence'."
        )
    ),
    Tool(
        name="arxiv_query",
        func=arxiv_query,
        description=(
            "Queries the ArXiv API to retrieve research papers. "
            "Input should be a search term, e.g., 'machine learning optimization'."
        )
    ),
    Tool(
        name="celsius_to_fahrenheit",
        func=celsius_to_fahrenheit,
        description=(
            "Converts a temperature from Celsius to Fahrenheit. "
            "Input should be a number representing Celsius, e.g., 25."
        )
    ),
    Tool(
        name="save_text_to_file",
        func=save_text_to_file,
        description=(
            "Saves provided text to a file. "
            "Input should be in the format: 'file_name||content'. "
            "Example: 'greeting.txt||Hello, world!'."
        )
    ),
    Tool(
        name="copy_file",
        func=copy_file,
        description=(
            "Copies a file from a source to a destination. "
            "Input should be in the format: 'source_path||destination_path'. "
            "Example: 'report.doc||backup_report.doc'."
        )
    )
]

# -------------------------
# Create the LLM agent
# -------------------------

# Initialize the LLM. In this example, we use OpenAI as a placeholder.
llm = OpenAI(api_key=os.environ["TOGETHER_AI_API_KEY"], temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -------------------------
# Example Queries
# -------------------------

queries = [
    "What is the summary of the Wikipedia page for 'Artificial Intelligence'?",
    "Show me the Wikipedia article about 'Quantum Computing'.",
    "Search for research papers on 'machine learning optimization' on Arxiv.",
    "Retrieve recent papers on 'graph neural networks' from Arxiv.",
    "Convert 25 Celsius to Fahrenheit.",
    "What is 100°C in Fahrenheit?",
    "Save the text 'Hello, world!' to a file called greeting.txt.",
    "Write my meeting notes to a file named notes.txt.",
    "Make a copy of the file report.doc to backup_report.doc.",
    "Duplicate the file data.csv to data_backup.csv.",
    "Get the Wikipedia overview of 'Climate Change'.",
    "Find me Arxiv papers about 'reinforcement learning'.",
    "Convert -40 Celsius to Fahrenheit.",
    "Please save this recipe to a file named recipe.txt.",
    "Copy the file image.png to image_copy.png."
]

# -------------------------
# Run the queries through the agent
# -------------------------

for i, query in enumerate(queries, start=1):
    print(f"Query {i}: {query}")
    try:
        # The agent's run method will decide which tool to invoke based on the query.
        result = agent.run(query)
    except Exception as e:
        result = f"Agent error: {str(e)}"
    print(f"Result: {result}\n{'-'*60}\n")
