# Cinny-AI Setup Guide

## Environment Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/YahyaLimbo/Cinny-AI_v.1.git
   cd Cinny-AI_v.1
   git checkout feature/azure-ai103-extensions
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

## Project Structure

```
Cinny-AI_v.1/
├── src/
│   ├── cinny-ai.py                  # Entry point
│   ├── cinny_agent_cli.py           # Interactive CLI (voice, translation, image modes)
│   ├── botany_kb.csv                # Plant Q&A knowledge base (33 entries)
│   ├── logical_kb.csv               # First-order logic rules
│   ├── fuzzy_kb.json                # Fuzzy logic plant parameters
│   ├── agents/
│   │   ├── semantic_kernel_agent.py # SK agent with RAG pipeline
│   │   └── plugins.py              # 5 native plugins (Logic, Fuzzy, Vision, Botany, RAG)
│   ├── azure_services/
│   │   └── search_indexer.py        # Azure AI Search index manager
│   ├── config/
│   │   └── settings.py              # Centralized env-based configuration
│   └── utils/
│       ├── logger.py                # Logging utility
│       └── test_agent.py            # Pipeline verification script
├── requirements.txt
├── .env.example
└── .gitignore
```

## Core Architecture

### RAG Pipeline (Retrieval-Augmented Generation)
```
User question
  ├── Embed query (sentence-transformers, all-MiniLM-L6-v2)
  ├── Hybrid search: BM25 keyword + vector similarity + semantic reranking
  │   (Azure AI Search with HNSW vector index)
  ├── Top 3 KB results retrieved as context
  └── Azure OpenAI (o4-mini) generates natural answer grounded in context
      Falls back to local TF-IDF if Azure is unreachable
```

### Semantic Kernel Plugins

| Plugin | Capability |
|---|---|
| **LogicPlugin** | First-order logic proving (NLTK Resolution Prover) |
| **FuzzyPlugin** | Fuzzy logic water/sunlight needs, desert suitability scoring |
| **VisionPlugin** | ResNet50 plant classification, YOLOv8 flower detection (local) |
| **BotanyPlugin** | iNaturalist API species lookup, IP-based local plant discovery |
| **RAGPlugin** | Hybrid vector search (Azure AI Search) with local TF-IDF fallback |

## Azure Services Used

| Service | Purpose | Free Tier |
|---|---|---|
| Azure OpenAI (o4-mini) | RAG answer generation | Student/free credits |
| Azure AI Search (F0) | Vector + semantic search index | 3 indexes, 50MB, 10K docs |

## Quick Start

### Run the Chatbot
```bash
cd ~/Cinny-AI_v.1
python src/cinny-ai.py
```

### Index the Knowledge Base (first time / after KB updates)
```bash
python -m src.azure_services.search_indexer          # Create index + upload docs
python -m src.azure_services.search_indexer --status  # Check document count
```

### Verify the Agent Pipeline
```bash
python src/utils/test_agent.py
```

### CLI Commands (inside chatbot)
- `/help` — Show available commands
- `/Language [Name]` — Change language (e.g. `/Language Spanish`)
- `/Voice ON/OFF` — Toggle text-to-speech
- `/IMAGE` — Open image classification / YOLO detection mode

## AI-103 Exam Domain Coverage

1. **Plan and manage an AI solution** — Architecture, service integration design
2. **Implement decision support solutions** — Semantic Kernel agent, multi-step reasoning
3. **Implement Computer Vision solutions** — Local ResNet50/YOLOv8 classification
4. **Implement NLP solutions** — Knowledge base Q&A, semantic search
5. **Implement knowledge mining solutions** — Azure AI Search with vector + semantic ranking
6. **Implement generative AI solutions** — RAG pipeline with Azure OpenAI
7. **Implement responsible AI** — Off-topic filtering, graceful fallbacks
