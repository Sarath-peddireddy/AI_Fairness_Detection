# AI Fairness Detection System 🚀

## Introduction
The AI Fairness Detection System is a comprehensive solution designed to detect and mitigate bias in text using state-of-the-art AI techniques. As AI systems increasingly impact decision-making processes, ensuring fairness and reducing bias is essential. This project addresses bias detection, classification, and mitigation while offering real-time feedback and insightful analysis.

## Key Features 🌟
- **Real-time Bias Detection and Scoring:** Automatically identify biased content with precision (0-1 scale).
- **Bias Type Classification:** Categorize detected bias into specific types for nuanced analysis.
- **Detailed Explanations:** Understand the context and reasoning behind detected biases.
- **Mitigation Strategies:** Get actionable and context-aware recommendations to reduce bias.
- **Hybrid Search with BM25 and Cross-Encoders:** Efficiently retrieve and rank relevant documents.
- **Document Search with RAG (Retrieval Augmented Generation):** Retrieve contextual information accurately.
- **Web Search Integration:** Stay up-to-date with real-time, relevant data.
- **Responsive and Dynamic UI:** Interactive interface with real-time feedback and visualization.

## Tech Stack 🛠️
### Backend
- Python 3.9
- OpenAI GPT-4 for bias analysis
- LangChain for RAG implementation
- ChromaDB for vector storage
- FastAPI/HTTP Server for API endpoints
- ONNX Runtime for optimized inference
- BM25 for hybrid search
- Cross-Encoders for reranking

### Frontend
- HTML5/CSS3 for responsive design
- Vanilla JavaScript for dynamic interactions
- Real-time bias visualization with dynamic color coding

## Integrations 🔗
- OpenAI API for bias detection
- SerpAPI for web search integration
- ChromaDB for document embedding
- BM25 and Cross-Encoders for hybrid search and reranking

## Installation & Setup 🛠️
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-fairness-detection.git
   cd env-name
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables for API keys:
   ```bash
   export HUGGINGFACE_API_KEY=your_huggingface_key
   export OPENAI_API_KEY=your_openai_key
   ```
4. Run the server:
   ```bash
   uvicorn app:app --reload
   ```

## Usage 🚀
1. Open the UI in your browser:
   ```
http://localhost:8000
   ```
2. Upload a document or enter text to analyze for bias.
3. View real-time results, including bias type, score, and mitigation suggestions.

## Contributing 🤝
Contributions are welcome! Feel free to open issues or submit pull requests to enhance features or fix bugs.

## License 📄
This project is licensed under the MIT License.

## Acknowledgements 🙌
Special thanks to the developers and communities behind OpenAI, LangChain, ChromaDB, and other libraries used in this project.

