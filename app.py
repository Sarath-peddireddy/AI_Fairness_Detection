import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from dotenv import load_dotenv
import openai
import requests
import traceback
import onnxruntime

# Import functions from your modules
from backend.utils.agents import initialize_agents, combined_agent_query
from backend.utils.pdf_processing import load_pdfs, chunk_documents, get_embeddings, embed_chunks
from backend.utils.retrieval import initialize_retrievers
from backend.utils.mitigation import get_mitigation_strategies
from backend.utils.bias_detection import infer_bias_type, detect_bias

# Load environment variables
load_dotenv()

# Set API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define PDF directory (adjust as needed)
PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), "backend", "data", "pdfs")

# Initialize document processing pipeline
try:
    # Load PDFs
    documents = load_pdfs(pdf_directory=PDF_DIRECTORY)
    
    # Chunk documents
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
    
    # Get embeddings
    embeddings = get_embeddings()
    
    # Embed chunks
    embedded_chunks = embed_chunks(chunks)
    
    # Initialize retrievers
    retrievers = initialize_retrievers(chunks, embedded_chunks)
    
    # Initialize agents
    agents = initialize_agents(retrievers)
    
    # Extract agents - assuming agents is a tuple with two elements (rag_agent, websearch_agent)
    # This is the fix for the 'tuple' object has no attribute 'get' error
    if isinstance(agents, tuple) and len(agents) >= 2:
        rag_agent = agents[0]
        websearch_agent = agents[1]
    else:
        print("Unexpected format for agents, falling back to simple search")
        rag_agent = None
        websearch_agent = None
    
    print("Successfully initialized document processing pipeline")
except Exception as e:
    print(f"Error initializing document processing pipeline: {str(e)}")
    print("Falling back to simple document search")
    # Fallback to simple document search
    rag_agent = None
    websearch_agent = None

# Simple in-memory database for demo purposes (fallback)
fallback_documents = [
    "Gender bias refers to unfair discrimination based on a person's gender.",
    "Racial bias involves prejudice or discrimination based on race.",
    "Age bias is discrimination against individuals based on their age.",
    "Confirmation bias is the tendency to search for information that confirms one's preexisting beliefs.",
    "Algorithmic bias occurs when an algorithm produces systematically prejudiced results."
]

def search_documents(query):
    """Simple document search."""
    results = []
    for doc in fallback_documents:
        if query.lower() in doc.lower():
            results.append(doc)
    return results

def search_web(query):
    """Search the web using SerpAPI."""
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 3
        }
        response = requests.get(url, params=params)
        results = response.json()
        
        # Extract organic results
        snippets = []
        for result in results.get("organic_results", []):
            if "snippet" in result:
                snippets.append(result["snippet"])
                
        return "\n".join(snippets)
    except Exception as e:
        print(f"Error searching web: {str(e)}")
        return f"Error searching web: {str(e)}"

def summarize_info(info):
    """Summarize information using OpenAI."""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes information."},
                {"role": "user", "content": f"Summarize the following information: {info}"}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing info: {str(e)}")
        return info

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_OPTIONS(self):
        self._set_headers()
        
    def do_GET(self):
        if self.path == '/':
            self._set_headers()
            response = {'message': 'AI Fairness Detection API is running'}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/status':
            self._set_headers()
            response = {'status': 'operational'}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == '/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                if 'text' not in data:
                    self.send_response(400)
                    self.end_headers()
                    response = {'error': 'Missing text field in request'}
                    self.wfile.write(json.dumps(response).encode())
                    return
                    
                query_text = data['text']
                
                # Step 1: Bias Detection
                bias_label, bias_score, _, explanation = detect_bias(query_text)
                
                # Step 2: Infer Bias Type
                bias_type = infer_bias_type(explanation)
                
                # Step 3: Mitigation Strategies
                mitigation_strategies = get_mitigation_strategies(bias_type)
                
                # Step 4: Document Search
                if rag_agent and websearch_agent:
                    try:
                        doc_results = combined_agent_query(rag_agent, websearch_agent, query_text)
                    except Exception as e:
                        print(f"Error in agent query: {str(e)}")
                        doc_results = search_documents(query_text)
                else:
                    doc_results = search_documents(query_text)
                
                # Step 5: Web Search
                web_results = search_web(query_text)
                
                # Step 6: Summarize
                if isinstance(doc_results, list):
                    doc_results_text = "\n".join(doc_results)
                else:
                    doc_results_text = str(doc_results)
                    
                combined_info = doc_results_text + "\n" + web_results
                summary = summarize_info(combined_info)
                
                # Prepare response
                self._set_headers()
                response = {
                    "bias_label": bias_label,
                    "bias_score": bias_score,
                    "bias_type": bias_type,
                    "explanation": explanation,
                    "mitigation_strategies": mitigation_strategies,
                    "rag_results": combined_info,
                    "summary": summary
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                traceback.print_exc()
                response = {'error': str(e)}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 