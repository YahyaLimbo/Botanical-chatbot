import csv
import sys
import hashlib
import argparse
import logging
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import settings

logger = logging.getLogger(__name__)

KB_PATH = ROOT_DIR / "src" / "botany_kb.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSIONS = 384
_embedding_model = None


def _get_embedding_model():
    """Lazily load the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a text string."""
    model = _get_embedding_model()
    return model.encode(text).tolist()


def _get_index_client() -> SearchIndexClient:
    """Create an authenticated SearchIndexClient for index management."""
    credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
    return SearchIndexClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        credential=credential,
    )


def _get_search_client() -> SearchClient:
    """Create an authenticated SearchClient for document operations."""
    credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
    return SearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )


def _build_index_schema() -> SearchIndex:
    """Define the search index schema with semantic + vector configuration.

    Fields:
        id              - unique document key (hash of question)
        Question        - searchable, semantic title field
        Answer          - searchable, semantic content field
        Category        - filterable tag for faceted search
        QuestionVector  - 384-dim embedding of Question (for vector search)
    """
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="Question",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SearchableField(
            name="Answer",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SimpleField(
            name="Category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="QuestionVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="botany-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="botany-hnsw"),
        ],
        profiles=[
            VectorSearchProfile(
                name="botany-vector-profile",
                algorithm_configuration_name="botany-hnsw",
            ),
        ],
    )

    
    semantic_config = SemanticConfiguration(
        name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="Question"),
            content_fields=[SemanticField(field_name="Answer")],
        ),
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    return SearchIndex(
        name=settings.AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def _categorize_question(question: str) -> str:
    """Auto-assign a category based on simple keyword heuristics."""
    q = question.lower()
    if any(w in q for w in ("water", "moisture", "humidity", "soak")):
        return "watering"
    if any(w in q for w in ("sunlight", "light", "led", "sun")):
        return "lighting"
    if any(w in q for w in ("pest", "mite", "gnat", "aphid", "infestation")):
        return "pests"
    if any(w in q for w in ("disease", "mildew", "fungus", "yellow", "brown", "powdery")):
        return "disease"
    if any(w in q for w in ("soil", "repot", "pot", "compost", "mulch")):
        return "soil"
    if any(w in q for w in ("propagat", "cutting", "node")):
        return "propagation"
    if any(w in q for w in ("forest", "deforestation", "native", "ecosystem")):
        return "ecology"
    if any(w in q for w in ("nutrient", "fertiliz", "nitrogen", "phosphorus")):
        return "nutrition"
    return "general"


def _load_documents_from_csv() -> list[dict]:
    """Read botany_kb.csv, embed each question, and convert to Azure Search documents."""
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {KB_PATH}")

    
    rows = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["Question"].strip(), row["Answer"].strip()))

    
    model = _get_embedding_model()
    questions = [q for q, a in rows]
    print(f"Embedding {len(questions)} questions...")
    vectors = model.encode(questions).tolist()

    documents = []
    for (question, answer), vector in zip(rows, vectors):
        doc_id = hashlib.md5(question.encode()).hexdigest()[:16]
        documents.append({
            "id": doc_id,
            "Question": question,
            "Answer": answer,
            "Category": _categorize_question(question),
            "QuestionVector": vector,
        })

    return documents


def create_or_update_index() -> str:
    """Create the search index (or update if it already exists)."""
    client = _get_index_client()
    index = _build_index_schema()
    result = client.create_or_update_index(index)
    return f"Index '{result.name}' ready (fields: {len(result.fields)})"


def upload_documents() -> str:
    """Upload all KB documents to the search index."""
    documents = _load_documents_from_csv()
    client = _get_search_client()
    result = client.upload_documents(documents=documents)

    succeeded = sum(1 for r in result if r.succeeded)
    failed = sum(1 for r in result if not r.succeeded)
    return f"Uploaded {succeeded} documents ({failed} failed) to '{settings.AZURE_SEARCH_INDEX_NAME}'"


def get_index_status() -> str:
    """Check current index document count and status."""
    try:
        client = _get_index_client()
        stats = client.get_index_statistics(settings.AZURE_SEARCH_INDEX_NAME)
        return (
            f"Index '{settings.AZURE_SEARCH_INDEX_NAME}': "
            f"{stats.document_count} documents, "
            f"{stats.storage_size:,} bytes"
        )
    except ResourceNotFoundError:
        return f"Index '{settings.AZURE_SEARCH_INDEX_NAME}' does not exist yet."


def delete_index() -> str:
    """Delete the search index entirely."""
    client = _get_index_client()
    client.delete_index(settings.AZURE_SEARCH_INDEX_NAME)
    return f"Index '{settings.AZURE_SEARCH_INDEX_NAME}' deleted."


def main():
    parser = argparse.ArgumentParser(description="Cinny-AI Azure Search Index Manager")
    parser.add_argument("--status", action="store_true", help="Show index status")
    parser.add_argument("--delete", action="store_true", help="Delete the index")
    args = parser.parse_args()

    if not settings.AZURE_SEARCH_API_KEY or not settings.AZURE_SEARCH_ENDPOINT:
        print("ERROR: AZURE_SEARCH_API_KEY and AZURE_SEARCH_ENDPOINT must be set in .env")
        sys.exit(1)

    if args.status:
        print(get_index_status())
    elif args.delete:
        confirm = input(f"Delete index '{settings.AZURE_SEARCH_INDEX_NAME}'? (y/N): ")
        if confirm.lower() == "y":
            print(delete_index())
    else:
        print("Creating/updating index...")
        print(create_or_update_index())
        print("\nUploading documents from botany_kb.csv...")
        print(upload_documents())
        print("\nDone! " + get_index_status())


if __name__ == "__main__":
    main()
