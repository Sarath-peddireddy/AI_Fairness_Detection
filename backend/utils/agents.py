from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ..config import OPENAI_API_KEY, SERPAPI_API_KEY
import importlib
import sys

def initialize_agents(ensemble_retriever):
    """Initialize RAG and websearch agents."""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",  # Using a more widely available model
            temperature=0,
            streaming=False,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        
        # Initialize RAG agent
        rag_agent = initialize_agent(
            tools=[Tool(
                name="Retrieval",
                func=ensemble_retriever.get_relevant_documents,
                description="Useful for answering questions about the documents."
            )],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
        # Check if SerpAPI is available
        try:
            # Try to import SerpAPIWrapper
            from langchain_community.utilities import SerpAPIWrapper
            # Initialize websearch agent
            search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
            websearch_agent = initialize_agent(
                tools=[Tool(
                    name="Google Search",
                    func=search.run,
                    description="Useful for when you need to answer questions about current events or the broader world."
                )],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                max_iterations=10,
                handle_parsing_errors=True
            )
        except ImportError:
            print("SerpAPI not available, using dummy web search agent")
            websearch_agent = DummyAgent("Web search is not available")
        
        return rag_agent, websearch_agent
    except Exception as e:
        print(f"Error initializing agents: {str(e)}")
        # Return dummy agents that return a message about the error
        return DummyAgent(f"RAG agent initialization failed: {str(e)}"), DummyAgent(f"Web search agent initialization failed: {str(e)}")

def combined_agent_query(rag_agent, websearch_agent, user_query):
    """Execute combined agent query."""
    try:
        rag_result = rag_agent.invoke({"input": user_query})
        websearch_result = websearch_agent.invoke({"input": user_query})
        
        final_rag = rag_result.get("output", "No RAG result found")
        final_web = websearch_result.get("output", "No Web result found")
        
        return f"{final_rag}\n{final_web}"
    except Exception as e:
        print(f"Error in combined agent query: {str(e)}")
        return f"Error retrieving information: {str(e)}"

class DummyAgent:
    """A dummy agent that returns a fixed message."""
    def __init__(self, message):
        self.message = message
    
    def invoke(self, query):
        return {"output": self.message}