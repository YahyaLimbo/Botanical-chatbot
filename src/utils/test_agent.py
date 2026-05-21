import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.logger import get_logger
from src.agents.semantic_kernel_agent import SemanticKernelAgent

logger = get_logger("verification")


def verify_agentic_pipeline():
    logger.info("Starting verification of the Agentic pipeline...")
    
    try:
        logger.info("Initializing SemanticKernelAgent and loading native plugins...")
        agent = SemanticKernelAgent()
        logger.info("Successfully instantiated SemanticKernelAgent!")

        logger.info("Testing LogicPlugin check_fact locally...")
        res = agent.logic_plugin.check_fact("cactus", "plant")
        logger.info(f"LogicPlugin fact check result: {res}")

        logger.info("Testing FuzzyPlugin check_fuzzy_need locally...")
        fuzzy_res = agent.fuzzy_plugin.check_fuzzy_need("cactus", "water")
        logger.info(f"FuzzyPlugin check result: {fuzzy_res}")

        logger.info("Testing RAGPlugin search_botany_kb fallback locally...")
        rag_res = agent.rag_plugin.search_botany_kb("monstera propagation")
        logger.info(f"RAGPlugin query result: {rag_res[:150]}...")

        logger.info("Testing BotanyPlugin iNaturalist queries locally...")
        botany_res = agent.botany_plugin.get_botanical_info("aloe vera")
        logger.info(f"BotanyPlugin query result: {botany_res[:150]}...")

        logger.info("--- PIPELINE VERIFICATION SUCCESSFUL! ---")
        return True
    except Exception as e:
        logger.error(f"Pipeline verification FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_agentic_pipeline()
    sys.exit(0 if success else 1)
