import logging
from typing import Callable, Any, Optional, List
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from openai import AzureOpenAI

from src.config.settings import settings
from src.agents.plugins import (
    LogicPlugin,
    FuzzyPlugin,
    VisionPlugin,
    BotanyPlugin,
    RAGPlugin,
)

logger = logging.getLogger(__name__)


class SemanticKernelAgent:
    """
    AI Agent using Microsoft Semantic Kernel modern v1.x framework.
    
    Features:
    - Native Plugin registration for tool use
    - Memory and context management
    - LLM-driven planning and reasoning
    - Extensible tool ecosystem
    """

    def __init__(self):
        """Initialize the Semantic Kernel agent"""
        self.kernel = Kernel()
        self.openai_client = None
        self._setup_ai_service()
        self._register_plugins()

    def _setup_ai_service(self) -> None:
        """Configure Azure OpenAI service in the kernel"""
        key = settings.AZURE_OPENAI_API_KEY
        endpoint = settings.AZURE_OPENAI_ENDPOINT
        if not key or key.startswith("your_"):
            logger.warning(
                "AZURE_OPENAI_API_KEY is not set in environment or .env. "
                "Agent running in Mock Offline Mode (Offline fallback responses will be simulated)."
            )
            return

        try:
          
            self.openai_client = AzureOpenAI(
                api_key=key,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=endpoint,
            )
            logger.info("Azure OpenAI client configured successfully for RAG generation.")
        except Exception as e:
            logger.error(f"Failed to configure Azure OpenAI: {e}")
            raise

    def _register_plugins(self) -> None:
        """Register custom native plugins in the kernel"""
        try:
            
            self.logic_plugin = LogicPlugin()
            self.fuzzy_plugin = FuzzyPlugin(self.logic_plugin)
            self.vision_plugin = VisionPlugin()
            self.botany_plugin = BotanyPlugin()
            self.rag_plugin = RAGPlugin()
            self.kernel.add_plugin(self.logic_plugin, plugin_name="LogicPlugin")
            self.kernel.add_plugin(self.fuzzy_plugin, plugin_name="FuzzyPlugin")
            self.kernel.add_plugin(self.vision_plugin, plugin_name="VisionPlugin")
            self.kernel.add_plugin(self.botany_plugin, plugin_name="BotanyPlugin")
            self.kernel.add_plugin(self.rag_plugin, plugin_name="RAGPlugin")

            logger.info("All local reasoning, fuzzy, vision, and RAG plugins registered in Semantic Kernel v1.x.")
        except Exception as e:
            logger.error(f"Failed to register custom native plugins: {e}")
            raise

    def _retrieve_rag_context(self, query: str) -> str:
        """Retrieve relevant context from Azure AI Search (vector + semantic hybrid).

        Returns formatted context string, or empty string if no results.
        """
        try:
            result = self.rag_plugin.search_botany_kb(query)
            if "could not find" not in result.lower() and "no plant care" not in result.lower():
                return result
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
        return ""

    def _generate_rag_response(self, query: str, context: str) -> str:
        """Send the user query + retrieved context to Azure OpenAI for natural answer generation."""
        system_prompt = (
            "You are Cinny, a friendly and knowledgeable botanical AI assistant. "
            "Your expertise is plants, gardening, botany, and ecology. "
            "Answer the user's question using the provided knowledge base context when relevant. "
            "If the context is relevant, ground your answer in it. "
            "If the context doesn't cover the question but it's still about plants, provide general botanical advice. "
            "If the question is completely unrelated to plants or botany (e.g. coding, math, history), "
            "politely decline and redirect: explain that you specialize in plant care and botany, "
            "and invite them to ask a plant-related question instead. "
            "Keep answers concise (2-4 sentences), conversational, and helpful."
        )

        user_message = f"Question: {query}"
        if context:
            user_message = (
                f"Knowledge Base Context:\n{context}\n\n"
                f"---\n\n"
                f"User Question: {query}\n\n"
                f"Use the context above to answer the question naturally."
            )

        response = self.openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=500,
        )
        return response.choices[0].message.content

    async def plan_and_execute(
        self,
        goal: str,
    ) -> str:
        """
        RAG pipeline: Retrieve context from Azure AI Search, then generate
        a natural answer with Azure OpenAI. Falls back to local tools if
        Azure services are unavailable.
        """
        if not self.openai_client:
            return self._execute_local_fallback(goal)

        try:
            
            context = self._retrieve_rag_context(goal)
            logger.info(f"RAG context retrieved ({len(context)} chars)")
            answer = self._generate_rag_response(goal, context)
            if answer:
                return answer

        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
        return self._execute_local_fallback(goal)

    def _execute_local_fallback(self, goal: str) -> str:
        """
        Offline fallback matching algorithm.
        Executes local tools directly depending on simple keyword patterns.
        """
        query = goal.strip().lower().rstrip("?!.")

        greetings = {"hello", "hi", "hey", "hola", "greetings", "howdy", "good morning", "good afternoon", "good evening"}
        if query in greetings or any(query.startswith(g + " ") for g in greetings):
            return "Hello! I am Cinny, your botanical assistant. I am running in local fallback mode. Ask me about plant care, species identification, desert suitability, or try `/IMAGE` mode!"

        if "classify" in query or "identify" in query or "image" in query or "photo" in query:
            return "Please select 'Image Mode' from the main CLI loop to process pictures using local Vision models."

        if "is a" in query or "is an" in query:
            parts = query.split("is a")
            if len(parts) < 2:
                parts = query.split("is an")
            if len(parts) >= 2:
                sub = parts[0].strip().replace("is ", "")
                pred = parts[1].strip()
                return self.logic_plugin.check_fact(sub, pred)

        if "water" in query or "sunlight" in query:
            words = query.split()
            plant = next((w for w in words if w not in ("how", "much", "water", "sunlight", "does", "need", "a", "an", "the")), None)
            if plant:
                prop = "water" if "water" in query else "sunlight"
                return self.fuzzy_plugin.check_fuzzy_need(plant, prop)

        if "desert" in query or "suitability" in query:
            words = query.split()
            plant = next((w for w in words if w not in ("how", "suitable", "is", "for", "a", "desert", "an", "the")), None)
            if plant:
                return self.fuzzy_plugin.check_desert_suitability(plant)

        if "near me" in query or "my area" in query or "local" in query:
            return self.botany_plugin.get_local_plants()

        rag_res = self.rag_plugin.search_botany_kb(goal)
        if "could not find" not in rag_res:
            return rag_res

        botany_res = self.botany_plugin.get_botanical_info(goal)
        if "couldn't fetch" not in botany_res:
            return botany_res

        return "I am running in offline mode. Please define Azure OpenAI credentials in your .env file to enable advanced chat reasoning."
