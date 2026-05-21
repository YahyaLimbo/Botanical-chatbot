import os
import csv
import json
import logging
import requests
import wikipedia
import threading
from typing import Dict, List, Tuple
from pathlib import Path

from semantic_kernel.functions.kernel_function_decorator import kernel_function
from src.config.settings import settings

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
QA_PAIRS_PATH = SCRIPT_DIR / "botany_kb.csv"
LOGICAL_KB_PATH = SCRIPT_DIR / "logical_kb.csv"
FUZZY_KB_PATH = SCRIPT_DIR / "fuzzy_kb.json"

torch = None
models = None
transforms = None
Image = None
YOLO = None
cv2 = None
ResolutionProver = None
Expression = None


def _lazy_load_dl():
    """Load PyTorch, torchvision, and Pillow lazily"""
    global torch, models, transforms, Image
    if torch is None:
        import torch as t
        import torchvision.transforms as tf
        from torchvision import models as m
        from PIL import Image as img

        torch = t
        transforms = tf
        models = m
        Image = img


def _lazy_load_yolo():
    """Load ultralytics YOLO and opencv lazily"""
    global YOLO, cv2
    if YOLO is None:
        from ultralytics import YOLO as y
        import cv2 as c

        YOLO = y
        cv2 = c


def _lazy_load_logic():
    """Load NLTK logical inference components lazily"""
    global ResolutionProver, Expression
    if ResolutionProver is None:
        import nltk
        from nltk.inference import ResolutionProver as rp
        from nltk.sem import Expression as expr

        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        ResolutionProver = rp
        Expression = expr

-
class LogicPlugin:
    """Plugin for deterministic First Order Logic proving and consistency verification"""

    def __init__(self):
        _lazy_load_logic()
        self.kb = []
        self.lock = threading.Lock()
        self.prover_timeout = settings.PROVER_TIMEOUT_SECONDS
        self._load_kb()

    def _load_kb(self):
        try:
            if not LOGICAL_KB_PATH.exists():
                logger.warning(f"Logical KB not found at {LOGICAL_KB_PATH}. Initializing empty.")
                return

            with open(LOGICAL_KB_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            expr = Expression.fromstring(line)
                            self.kb.append(expr)
                        except Exception as e:
                            logger.error(f"Error parsing logic fact '{line}': {e}")
            logger.info(f"Loaded {len(self.kb)} logical rules and facts.")
        except Exception as e:
            logger.error(f"Failed to load FOL knowledge base: {e}")

    def _clean_input(self, text: str) -> str:
        text = text.strip().lower()
        words = [w for w in text.split() if w not in ("a", "an", "the")]
        return "_".join(words)

    def _prove_with_timeout(self, goal, assumptions) -> bool:
        result = [False]

        def worker():
            try:
                prover = ResolutionProver()
                result[0] = prover.prove(goal, assumptions)
            except Exception:
                result[0] = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(self.prover_timeout)
        return result[0]

    @kernel_function(
        name="check_fact",
        description="Verify if a logical relationship is True, False, or Unknown in the KB (e.g. check if cactus is a plant)"
    )
    def check_fact(self, subject: str, predicate: str) -> str:
        """Verify if a logical statement is true or false"""
        with self.lock:
            sub = self._clean_input(subject)
            pred = self._clean_input(predicate)
            goal_str = f"{pred}({sub})"

            try:
                goal = Expression.fromstring(goal_str)
                negation = Expression.fromstring(f"-{goal_str}")

        
                if self._prove_with_timeout(goal, self.kb):
                    return "Correct"
                if self._prove_with_timeout(negation, self.kb):
                    return "Incorrect"
                return "Unknown"
            except Exception as e:
                return f"Logic resolution error: {e}"

    @kernel_function(
        name="add_fact",
        description="Add a logical rule or relationship to the KB (e.g. state that Rose is a plant)"
    )
    def add_fact(self, subject: str, predicate: str) -> str:
        """Add a factual logical statement if consistent"""
        with self.lock:
            sub = self._clean_input(subject)
            pred = self._clean_input(predicate)
            fact_str = f"{pred}({sub})"

            try:
                fact = Expression.fromstring(fact_str)
                negation = Expression.fromstring(f"-{fact_str}")

                
                if self._prove_with_timeout(negation, self.kb):
                    return f"Contradiction! I cannot believe that because it contradicts my existing knowledge of {subject}."

                test_kb = self.kb + [fact]
                if self._prove_with_timeout(negation, test_kb):
                    return f"Adding this fact would make my logical knowledge base inconsistent."

                if fact not in self.kb:
                    self.kb.append(fact)
            
                    with open(LOGICAL_KB_PATH, "w", encoding="utf-8") as f:
                        f.write("\n".join(str(expr) for expr in self.kb) + "\n")
                    return f"OK, I will remember that {subject} is {predicate}."
                return f"I already know that {subject} is {predicate}."
            except Exception as e:
                return f"Logic insertion error: {e}"

    @kernel_function(
        name="remove_fact",
        description="Remove a logical statement from the knowledge base"
    )
    def remove_fact(self, subject: str, predicate: str) -> str:
        """Forget/delete a logical statement"""
        with self.lock:
            sub = self._clean_input(subject)
            pred = self._clean_input(predicate)
            fact_str = f"{pred}({sub})"

            try:
                fact = Expression.fromstring(fact_str)
                if fact in self.kb:
                    self.kb.remove(fact)
                    with open(LOGICAL_KB_PATH, "w", encoding="utf-8") as f:
                        f.write("\n".join(str(expr) for expr in self.kb) + "\n")
                    return f"OK, I have forgotten that {subject} is {predicate}."
                return f"I don't have the fact '{fact_str}' in my knowledge base."
            except Exception as e:
                return f"Logic deletion error: {e}"

class FuzzyPlugin:
    """Plugin for managing and querying water/sunlight needs using Fuzzy Logic metrics"""

    def __init__(self, logic_plugin: LogicPlugin):
        self.logic_plugin = logic_plugin
        self.fuzzy_kb = {}
        self.lock = threading.Lock()
        self._load_fuzzy_kb()

    def _load_fuzzy_kb(self):
        try:
            if not FUZZY_KB_PATH.exists():
                return
            with open(FUZZY_KB_PATH, "r") as f:
                self.fuzzy_kb = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load fuzzy KB: {e}")

    def _save_fuzzy_kb(self):
        try:
            with open(FUZZY_KB_PATH, "w") as f:
                json.dump(self.fuzzy_kb, f)
        except Exception as e:
            logger.error(f"Failed to save fuzzy KB: {e}")

    @kernel_function(
        name="add_fuzzy_need",
        description="Record a fuzzy value (0.0 to 1.0) for a property of a plant (e.g. record water need of rose as high or 0.8)"
    )
    def add_fuzzy_need(self, subject: str, property_name: str, value: str) -> str:
        """Assign a fuzzy score (0.0 to 1.0) or semantic tag to a plant requirement"""
        with self.lock:
            sub = self.logic_plugin._clean_input(subject)
            prop = property_name.strip().lower()

        
            word_map = {
                "none": 0.0, "very low": 0.1, "low": 0.2, "medium low": 0.3,
                "medium": 0.5, "moderate": 0.5,
                "medium high": 0.7, "high": 0.8, "very high": 0.9, "extreme": 1.0,
            }

            val_str = value.strip().lower()
            if val_str in word_map:
                val = word_map[val_str]
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    return f"Invalid fuzzy value. Please specify a number between 0.0 and 1.0, or use tags like: low, medium, high."

            if not (0.0 <= val <= 1.0):
                return "Fuzzy values must be between 0.0 and 1.0."

            level = "Low" if val <= 0.3 else "Moderate" if val <= 0.6 else "High"

           
            if prop == "water" and val > 0.5:
                
                logic_res = self.logic_plugin.check_fact(subject, "succulent")
                if logic_res == "Correct":
                    return f"Contradiction! {subject.title()} is logically proven to be a succulent. Succulents require low water. Cannot record a water need of {val} ({level})."

            if sub not in self.fuzzy_kb:
                self.fuzzy_kb[sub] = {}
            self.fuzzy_kb[sub][prop] = val
            self._save_fuzzy_kb()

            return f"Successfully recorded {prop} requirement of {subject} as {val} ({level})."

    @kernel_function(
        name="check_fuzzy_need",
        description="Query the fuzzy value level of a plant property (e.g. check how much water a cactus needs)"
    )
    def check_fuzzy_need(self, subject: str, property_name: str) -> str:
        """Evaluate and retrieve fuzzy parameters for a plant"""
        sub = self.logic_plugin._clean_input(subject)
        prop = property_name.strip().lower()

        if sub in self.fuzzy_kb and prop in self.fuzzy_kb[sub]:
            val = self.fuzzy_kb[sub][prop]
            level = "Low" if val <= 0.3 else "Moderate" if val <= 0.6 else "High"
            return f"Based on fuzzy membership rules ({val}), {subject.title()} has a {level} {prop} requirement."
        return f"I have no fuzzy data recorded for {prop} needs of {subject}."

    @kernel_function(
        name="check_desert_suitability",
        description="Compute a desert suitability score (0.0-1.0) using fuzzy logic based on sunlight and low-water needs"
    )
    def check_desert_suitability(self, subject: str) -> str:
        """Evaluate desert compatibility score using a fuzzy t-norm (min intersection)"""
        sub = self.logic_plugin._clean_input(subject)

        if sub in self.fuzzy_kb and "water" in self.fuzzy_kb[sub] and "sunlight" in self.fuzzy_kb[sub]:
            w_need = self.fuzzy_kb[sub]["water"]
            s_need = self.fuzzy_kb[sub]["sunlight"]

            suitability_score = min(s_need, 1.0 - w_need)
            suit = "highly" if suitability_score > 0.7 else "moderately" if suitability_score > 0.4 else "not very"

            return (
                f"Fuzzy Desert Suitability Score for {subject.title()}: {suitability_score:.2f}.\n"
                f"It is evaluated as {suit} suitable for arid/desert environments."
            )
        return f"I need both 'water' and 'sunlight' fuzzy parameters recorded to calculate desert suitability for {subject}."


class VisionPlugin:
    """Plugin for PyTorch ResNet50 plant classification and YOLOv8 flower species detection"""

    def __init__(self):
        self.device = None
        self.transform = None
        self.flower_classes = []
        self.flower_model = None
        self.yolo_model = None

    def _init_resnet(self) -> bool:
        _lazy_load_dl()
        if self.flower_model is not None:
            return True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        paths = [
            SCRIPT_DIR / "best_plant_classifier.pt",
            SCRIPT_DIR / "src" / "best_plant_classifier.pt"
        ]
        model_path = next((p for p in paths if p.exists()), None)

        if not model_path:
            logger.warning("best_plant_classifier.pt not found.")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.flower_classes = checkpoint["class_names"]
            self.flower_model = models.resnet50(weights=None)
            num_ftrs = self.flower_model.fc.in_features
            self.flower_model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(num_ftrs, len(self.flower_classes)),
            )
            self.flower_model.load_state_dict(checkpoint["model_state_dict"])
            self.flower_model.to(self.device)
            self.flower_model.eval()
            logger.info(f"ResNet50 Flower Classifier loaded. Classes: {self.flower_classes}")
            return True
        except Exception as e:
            logger.error(f"Error loading ResNet50 classifier: {e}")
            return False

    def _init_yolo(self) -> bool:
        _lazy_load_yolo()
        if self.yolo_model is not None:
            return True

        paths = [
            SCRIPT_DIR / "flower_yolov8.pt",
            SCRIPT_DIR / "src" / "flower_yolov8.pt",
            SCRIPT_DIR / "plants_yolov8n.pt",
            SCRIPT_DIR / "src" / "plants_yolov8n.pt"
        ]
        model_path = next((p for p in paths if p.exists()), None)

        if not model_path:
            logger.warning("YOLOv8 weights not found.")
            return False

        try:
            self.yolo_model = YOLO(str(model_path))
            logger.info(f"YOLOv8 Detector loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLOv8: {e}")
            return False

    @kernel_function(
        name="classify_plant_photo",
        description="Identify the species of a single plant in a photo (using local PyTorch ResNet50)"
    )
    def classify_plant_photo(self, image_path: str) -> str:
        """Classify a single plant image using local ResNet50 model"""
        if not self._init_resnet():
            return "Local PyTorch plant classifier is not currently loaded or model weights are missing."

        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.flower_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probs, 0)
                label = self.flower_classes[predicted.item()].title()

                if confidence.item() < 0.4:
                    return f"I'm not highly confident, but it resembles a {label} ({confidence.item()*100:.1f}%). Try a clearer picture!"
                return f"I am {confidence.item()*100:.1f}% confident that this is a {label}."
        except Exception as e:
            return f"Error classifying image: {e}"

    @kernel_function(
        name="detect_flower_objects",
        description="Detect and list multiple flower occurrences in a single photo (using local YOLOv8)"
    )
    def detect_flower_objects(self, image_path: str) -> str:
        """Run multi-object YOLOv8 bounding-box detection on an image"""
        from collections import Counter
        if not self._init_yolo():
            return "YOLOv8 detection model is not loaded or weights are missing."

        try:
            res = self.yolo_model(image_path, conf=0.25, iou=0.45, imgsz=640)
            detected_names = []
            img_draw = cv2.imread(image_path)

            species_colors = {
                "Chamomile": (255, 255, 255), "Daffodil": (0, 255, 255),
                "Dandelion": (0, 200, 255), "Crocus": (255, 100, 150),
                "Iris": (255, 0, 128), "Pansy": (200, 50, 200),
                "Rose": (147, 20, 255), "Sunflower": (0, 215, 255),
                "Tigerlily": (0, 128, 255), "Tulip": (255, 0, 127)
            }
            default_color = (0, 255, 0)

            if len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.yolo_model.names[class_id]

                    detected_names.append(class_name)

                    color = species_colors.get(class_name, default_color)
                    label = f"{class_name}: {confidence:.2f}"

                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_draw, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
                    cv2.putText(img_draw, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            o_path = str(SCRIPT_DIR / "src" / "detected_output.jpg")
            cv2.imwrite(o_path, img_draw)

            if detected_names:
                count_dict = Counter(detected_names)
                result_parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in count_dict.items()]
                return f"Flower detection complete! Bounding boxes saved to detected_output.jpg. Found: {', '.join(result_parts)}"
            return "No recognisable flower species detected in the image."
        except Exception as e:
            return f"Error executing object detection: {e}"


class BotanyPlugin:
    """Plugin for live web integration with iNaturalist taxon APIs and Wikipedia summaries"""

    @kernel_function(
        name="get_botanical_info",
        description="Query external APIs to fetch species identification, description, and details for a plant name"
    )
    def get_botanical_info(self, plant_name: str) -> str:
        """Fetch real-time data from iNaturalist and Wikipedia"""
        query = plant_name.strip()
        try:
            url = f"https://api.inaturalist.org/v1/taxa?q={query}&rank=species,genus&iconic_taxa=Plantae"
            res = requests.get(url, timeout=5).json()

            if res.get("results"):
                best = res["results"][0]
                common = best.get("preferred_common_name", "No common name")
                sci = best.get("name", "Unknown")
                wiki_url = best.get("wikipedia_url")
                threat = best.get("conservation_status", {}).get("status_name", "Least Concern")
                observations = best.get("observations_count", 0)

                summary = ""
                if wiki_url:
                    try:
                        title = wiki_url.split("/")[-1]
                        summary = wikipedia.summary(title, sentences=2, auto_suggest=False)
                    except Exception:
                        pass

                info = (
                    f"Result: {common} ({sci})\n"
                    f"- Status: {threat}\n"
                    f"- Global Observations: {observations:,}\n"
                )
                if summary:
                    info += f"- Description: {summary}\n"
                if wiki_url:
                    info += f"- Wikipedia Link: {wiki_url}"
                return info

        except Exception as e:
            logger.error(f"Error querying botany APIs: {e}")

        return f"I couldn't fetch live botanical information online for '{query}'."

    @kernel_function(
        name="get_local_plants",
        description="Determine the user's approximate city using their IP and fetch the top 5 most commonly observed local plants"
    )
    def get_local_plants(self) -> str:
        """Fetch top observed native plants based on user IP geolocation"""
        try:
            res = requests.get("http://ip-api.com/json/", timeout=5).json()
            if res.get("status") == "success":
                city = res["city"]
                lat = res["lat"]
                lon = res["lon"]

                url = f"https://api.inaturalist.org/v1/observations/species_counts?lat={lat}&lng={lon}&radius=3&iconic_taxa=Plantae"
                plant_data = requests.get(url, timeout=5).json()

                if plant_data.get("results"):
                    names = []
                    for r in plant_data["results"][:5]:
                        taxon = r["taxon"]
                        name = taxon.get("preferred_common_name", taxon["name"])
                        names.append(name)
                    return f"Within 3km of {city}, the most commonly observed plant species are: {', '.join(names)}."
        except Exception as e:
            logger.error(f"Error getting local plants: {e}")

        return "I was unable to determine your local region or fetch nearby plant observations."

class RAGPlugin:
    """Plugin for Retrieval-Augmented Generation semantic search on local QA corpus or Azure AI Search"""

    def __init__(self):
        self.qa_pairs = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.lemmatizer = None
        self._load_local_kb()

    def _load_local_kb(self):
        """Load and vectorize botany_kb.csv for local semantic matching"""
        try:
            if not QA_PAIRS_PATH.exists():
                logger.warning(f"Botany Q&A corpus not found at {QA_PAIRS_PATH}")
                return

            with open(QA_PAIRS_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.qa_pairs.append((row["Question"], row["Answer"]))

            
            from sklearn.feature_extraction.text import TfidfVectorizer
            import nltk
            from nltk.stem import WordNetLemmatizer

            self.lemmatizer = WordNetLemmatizer()

            def lemmatize_text(text):
                tokens = nltk.word_tokenize(text.lower())
                return " ".join([self.lemmatizer.lemmatize(t) for t in tokens])

            questions = [lemmatize_text(q) for q, a in self.qa_pairs]
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(questions)

            logger.info(f"Loaded and indexed {len(self.qa_pairs)} botany Q&A pairs locally.")
        except Exception as e:
            logger.error(f"Failed to load or index local botany Q&A: {e}")

    def _azure_search_available(self) -> bool:
        """Check if Azure AI Search credentials are configured (not placeholder values)."""
        key = settings.AZURE_SEARCH_API_KEY
        endpoint = settings.AZURE_SEARCH_ENDPOINT
        return bool(
            key and endpoint
            and not key.startswith("your_")
            and not endpoint.startswith("https://your-")
        )

    def _get_query_vector(self, query: str) -> list[float]:
        """Embed a query string using the local sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_embed_model"):
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model.encode(query).tolist()

    def _search_azure(self, query: str, category: str = None) -> str | None:
        """Execute hybrid search (keyword + vector + semantic reranking).

        Combines three search signals for best results:
        1. BM25 keyword matching on Question/Answer text
        2. Vector similarity on QuestionVector embeddings
        3. Semantic reranking for final ordering
        """
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.models import QueryType, VectorizedQuery

            credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
            client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=settings.AZURE_SEARCH_INDEX_NAME,
                credential=credential,
            )
            
            filter_expr = f"Category eq '{category}'" if category else None

            query_vector = VectorizedQuery(
                vector=self._get_query_vector(query),
                k_nearest_neighbors=3,
                fields="QuestionVector",
            )

            results = client.search(
                search_text=query,
                vector_queries=[query_vector],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
                select=["Question", "Answer", "Category"],
                filter=filter_expr,
                top=3,
            )

            hits = []
            for doc in results:
                q = doc.get("Question", "")
                a = doc.get("Answer", "")
                cat = doc.get("Category", "")
                caption = ""
                if hasattr(doc, "captions") and doc.captions:
                    caption = doc.captions[0].text
                entry = f"Q: {q}\nA: {a}"
                if caption:
                    entry += f"\nHighlight: {caption}"
                if cat:
                    entry += f"\n[{cat}]"
                hits.append(entry)

            if hits:
                logger.info(f"Azure AI Search returned {len(hits)} hybrid results.")
                return "\n\n---\n\n".join(hits)

        except Exception as e:
            logger.warning(f"Azure Search query failed (falling back to local index): {e}")

        return None

    @kernel_function(
        name="search_botany_kb",
        description="Query the botanical Q&A knowledge base (RAG) to find explanations for plant symptoms or care instructions. Optionally filter by category: watering, lighting, pests, disease, soil, propagation, ecology, nutrition, general."
    )
    def search_botany_kb(self, query: str, category: str = None) -> str:
        """Perform semantic retrieval: Azure AI Search with semantic ranking, or local TF-IDF fallback."""
        if self._azure_search_available():
            result = self._search_azure(query, category)
            if result:
                return result

        if not self.qa_pairs or not self.vectorizer:
            return "No plant care database is currently loaded."

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import nltk

            def lemmatize_text(text):
                tokens = nltk.word_tokenize(text.lower())
                return " ".join([self.lemmatizer.lemmatize(t) for t in tokens])

            query_vector = self.vectorizer.transform([lemmatize_text(query)])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]

            threshold = settings.SIMILARITY_THRESHOLD
            if best_score >= threshold:
                q, a = self.qa_pairs[best_idx]
                return f"[Local KB Match - Confidence: {best_score:.2f}]\nQ: {q}\nA: {a}"

        except Exception as e:
            logger.error(f"Local semantic search error: {e}")

        return "I could not find any specific matching instructions or data in the Q&A knowledge base for your query."
