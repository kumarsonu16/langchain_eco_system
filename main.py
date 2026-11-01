import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv(override=True)

# We can integrate the Tavily package directly, but I think it would be better to use Tavily’s official LangChain integration. 
# Since Tavily developed this package specifically for LangChain, they may have provided better descriptions, optimizations, and compatibility for seamless integration


# from tavily import TavilyClient
# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

from langchain_tavily import TavilySearch

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# tools = [search] # we don't need to define our search function once imported TavilySearch from langchain_tavily
tools =[TavilySearch(max_results=5)]
agent = create_agent(model=llm, tools=tools)


def main():
    print("first search aganet using langchain")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in UAE on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
