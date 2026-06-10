# Botanical Chatbot

An intelligent botanical AI assistant powered by Azure OpenAI and the Semantic Kernel framework. Cinny combines retrieval-augmented generation (RAG), first-order logic reasoning, fuzzy logic, and computer vision to provide expert plant care guidance, species identification, and botanical information.

## Overview

This chatbot demonstrates enterprise-grade AI architecture by integrating multiple AI/ML techniques to deliver accurate, context-aware botanical assistance. It features hybrid search capabilities, multi-modal reasoning, and graceful fallback mechanisms for offline operation.

**Key Technologies:**
- Azure OpenAI (o4-mini) for natural language generation
- Azure AI Search with vector and semantic search
- Microsoft Semantic Kernel v1.x for agent orchestration
- Sentence-transformers for semantic embeddings
- NLTK for first-order logic reasoning
- YOLOv8 and ResNet50 for plant classification and flower detection

## Features

### Core Capabilities

- **Botanical Q&A**: Answer plant care questions using a hybrid RAG pipeline combining keyword search, vector similarity, and semantic reranking
- **Plant Classification**: Identify plants from images using local ResNet50 model
- **Flower Detection**: Detect and locate flowers in images using YOLOv8
- **Logic-Based Reasoning**: Verify plant facts using first-order logic inference
- **Fuzzy Logic Analysis**: Assess plant suitability based on environmental conditions
- **Multi-Language Support**: Translate responses to 10+ languages
- **Text-to-Speech**: Generate audio responses in supported languages
- **Species Lookup**: Query iNaturalist API for detailed botanical information

### Plugin Architecture

| Plugin | Purpose |
|--------|---------|
| **LogicPlugin** | First-order logic proving and fact verification |
| **FuzzyPlugin** | Fuzzy logic inference for environmental suitability |
| **VisionPlugin** | ResNet50 plant classification and YOLOv8 flower detection |
| **BotanyPlugin** | iNaturalist API integration for species data |
| **RAGPlugin** | Hybrid vector search with local TF-IDF fallback |

## Requirements

Python 3.10 or higher

Core Dependencies:
- openai >= 2.37.0
- semantic-kernel >= 1.42.0
- azure-search-documents >= 12.0.0
- sentence-transformers >= 5.5.0
- nltk >= 3.9.0
- ultralytics >= 8.0.0 (YOLOv8)
- torch and torchvision (for ResNet50)
- deep-translator >= 1.11.0
- gTTS >= 2.5.0

All dependencies are listed in `requirements.txt` with pinned versions.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YahyaLimbo/Botanical-chatbot.git
cd Botanical-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the `.env.example` file and configure with your Azure credentials:

```bash
cp .env.example .env
```

Edit `.env` with:
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_SEARCH_API_KEY`: Your Azure Search API key
- `AZURE_SEARCH_ENDPOINT`: Your Azure Search endpoint URL

**Note:** The chatbot will run in offline fallback mode if Azure credentials are not configured, using local models only.

## Project Structure

```
Botanical-chatbot/
├── src/
│   ├── cinny-ai.py                    # Entry point
│   ├── cinny_agent_cli.py             # Interactive CLI interface
│   ├── botany_kb.csv                  # Plant Q&A knowledge base (33 entries)
│   ├── logical_kb.csv                 # First-order logic rules
│   ├── fuzzy_kb.json                  # Fuzzy logic plant parameters
│   ├── agents/
│   │   ├── semantic_kernel_agent.py   # SK agent with RAG pipeline
│   │   └── plugins.py                 # 5 native plugins
│   ├── azure_services/
│   │   └── search_indexer.py          # Azure AI Search index management
│   ├── config/
│   │   └── settings.py                # Environment-based configuration
│   └── utils/
│       ├── logger.py                  # Logging utilities
│       └── test_agent.py              # Pipeline verification
├── requirements.txt                    # Python dependencies
├── SETUP_GUIDE.md                     # Detailed setup documentation
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

## Usage

### Start the Chatbot

```bash
python src/cinny-ai.py
```

This launches the interactive CLI where you can ask botanical questions, request image analysis, and use various commands.

### Index the Knowledge Base (First Time Setup)

```bash
# Create and populate the Azure AI Search index
python -m src.azure_services.search_indexer

# Check indexing status
python -m src.azure_services.search_indexer --status
```

### Verify the Agent Pipeline

```bash
python src/utils/test_agent.py
```

This runs a verification test that confirms all plugins are properly loaded and functional.

### CLI Commands

Inside the chatbot interface, use these commands:

- `/help` - Display available commands
- `/Language [Name]` - Change language (e.g., `/Language Spanish`)
- `/Voice ON|OFF` - Toggle text-to-speech output
- `/IMAGE` - Enter image classification or flower detection mode

## Architecture

### RAG Pipeline

The chatbot answers botanical questions through a multi-stage retrieval-augmented generation pipeline:

1. **Embedding**: User query is encoded to 384-dimensional vector using sentence-transformers (all-MiniLM-L6-v2)
2. **Hybrid Search**: Azure AI Search performs dual search:
   - BM25 keyword matching
   - Vector similarity search (HNSW index)
   - Semantic reranking for relevance
3. **Context Retrieval**: Top 3 most relevant knowledge base results are retrieved
4. **Generation**: Azure OpenAI generates natural language response grounded in retrieved context
5. **Fallback**: If Azure is unavailable, local TF-IDF retrieval provides offline responses

### Azure Services

| Service | Purpose | Configuration |
|---------|---------|----------------|
| Azure OpenAI (o4-mini) | LLM for response generation | Deployed model: o4-mini |
| Azure AI Search (F0) | Vector and semantic search | 3 indexes, 50MB storage, 10K docs |

Both services offer free tier eligibility for students and trial users.

## Development

### Project Dependencies

- **Semantic Kernel v1.x**: Modern Microsoft framework for LLM application development
- **Pydantic v2**: Data validation and settings management
- **Azure SDK**: Azure OpenAI and Search service clients
- **NLTK**: Natural Language Toolkit for logic inference
- **PyTorch & Torchvision**: Deep learning for vision models

### Logging

Logs are written to `logs/cinny_ai.log` and the console. Configure log level in `.env`:

```
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Configuration

All application settings are centralized in `src/config/settings.py` and loaded from environment variables via `.env`.

## Use Cases

- Plant identification and care guidance
- Indoor and outdoor gardening advice
- Plant species lookup and botanical information
- Multi-language plant care instructions
- Image-based plant detection and classification
- Off-topic filtering with graceful fallbacks

## References

- [Microsoft Semantic Kernel Documentation](https://microsoft.github.io/semantic-kernel/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/)
- [Sentence Transformers](https://www.sbert.net/)
- [YOLOv8 Documentation](https://github.com/ultralytics/ultralytics)

## License

This project is provided as-is for educational and research purposes.

## Author

Yahya Limbo - YahyaLimbo

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
