import os
import re
import sys
import asyncio
import logging
import platform
import warnings
from pathlib import Path

from deep_translator import GoogleTranslator
from gtts import gTTS

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
logs_dir = ROOT_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.WARNING,
    filename=str(logs_dir / "cinny_agent.log"),
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

for logger_name in ("semantic_kernel", "azure", "urllib3", "openai", "src"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

logger = logging.getLogger("cinny_agent")

tk = None
filedialog = None


def _lazy_load_gui():
    """Load tkinter components lazily"""
    global tk, filedialog
    if tk is None:
        try:
            import tkinter as t
            from tkinter import filedialog as fd
            tk = t
            filedialog = fd
        except ImportError:
            pass


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

from src.config.settings import settings
from src.agents.semantic_kernel_agent import SemanticKernelAgent

session_lang = "en"
voice_enabled = False
agent = None

SUPPORTED_LANGS = {
    "english": "en", "italian": "it", "spanish": "es", "french": "fr",
    "german": "de", "portuguese": "pt", "chinese": "zh-cn", "japanese": "ja",
    "russian": "ru", "arabic": "ar"
}


def print_banner():
    """Render the classic premium green Cinny terminal banner"""
    GREEN = "\033[92m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    lines = [
        f"{GREEN}           ███",
        "          ▒▒▒                                   " + WHITE,
        "  ██████  ████  ████████   ████████   █████ ████",
        " ███▒▒███▒▒███ ▒▒███▒▒███ ▒▒███▒▒███ ▒▒███ ▒███ ",
        "▒███ ▒▒▒  ▒███  ▒███ ▒███  ▒███ ▒███  ▒███ ▒███ ",
        "▒███  ███▒▒███  ▒███ ▒███  ▒███ ▒███  ▒███ ▒███ ",
        "▒▒██████  █████ ████ █████ ████ █████ ▒▒███████ ",
        " ▒▒▒▒▒▒  ▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒   ▒▒▒▒▒███ ",
        "                                       ███ ▒███ ",
        "                                       ▒▒██████  ",
        "                                        ▒▒▒▒▒▒   " + RESET,
    ]
    print("\n".join(lines))
    print(f"{GREEN}AI-PLANT AGENTIC ASSISTANT (SEMANTIC KERNEL EDITION){RESET}\n")


def speak_text(text: str, lang_code: str):
    """Synthesize response speech and play locally"""
    if not voice_enabled or not text.strip():
        return
    try:
        tts = gTTS(text=text, lang=lang_code)
        filename = SCRIPT_DIR / "response.mp3"
        tts.save(str(filename))

        if platform.system() == "Windows":
            os.system(f"start /min {filename}")
        else:
            os.system(f"mpg123 -q {filename} > /dev/null 2>&1")
            if filename.exists():
                filename.unlink()
    except Exception:
        pass


async def handle_image_mode():
    """Trigger visual analysis using local VisionPlugin tools"""
    _lazy_load_gui()
    if tk is None:
        print("Error: tkinter GUI libraries are not installed or available.")
        return

    print("\n=== IMAGE MODE ===")
    print("1. Identify single plant species (Classification)")
    print("2. Detect multiple flowers in image (YOLOv8 Detection)")
    
    try:
        choice = input("Select mode (1 or 2): ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        f_path = filedialog.askopenfilename(
            title="Select an Image for Vision Analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        root.destroy()

        if not f_path:
            print("No file selected.")
            return

        print(f"Loading local Vision model to process: {Path(f_path).name}...")
        loop = asyncio.get_running_loop()
        if choice == "2":
            ans = await loop.run_in_executor(None, agent.vision_plugin.detect_flower_objects, f_path)
        else:
            ans = await loop.run_in_executor(None, agent.vision_plugin.classify_plant_photo, f_path)

        print(ans)
        speak_text(ans, session_lang)

    except Exception as e:
        print(f"Vision GUI Error: {e}")
        f_p = input("Enter absolute file path to image: ").strip()
        if os.path.exists(f_p):
            loop = asyncio.get_running_loop()
            ans = await loop.run_in_executor(None, agent.vision_plugin.classify_plant_photo, f_p)
            print(ans)
            speak_text(ans, session_lang)


async def main():
    global session_lang, voice_enabled, agent

    print_banner()
    print("Initializing Semantic Kernel Agent, local PyTorch classifiers, and vector search RAG...")
    
    try:
        loop = asyncio.get_running_loop()
        agent = await loop.run_in_executor(None, SemanticKernelAgent)
        print("\nInitialization Complete!")
        print("Hello, I am Cinny! Your AI agent plant expert. How can I help you today?")
    except Exception as e:
        print(f"Critical error initializing agent: {e}")
        return

    while True:
        try:
   
            user_input_raw = await loop.run_in_executor(None, lambda: input("> "))
        except (EOFError, KeyboardInterrupt):
            print("\nBye! Happy gardening!")
            break

        if not user_input_raw.strip():
            continue

        if user_input_raw.lower() == "/help":
            print("\nCommands Available:")
            print("- /Language [Name] : Change speaking language (e.g. /Language Spanish)")
            print("- /Voice ON/OFF    : Toggle synthetic voice responses")
            print("- /IMAGE           : Open dialog to classify plant photo / run YOLO detection")
            print("- /help            : Show this list")
            print("- Exit (Ctrl+C)    : Exit assistant\n")
            continue

        if user_input_raw.upper().startswith("/VOICE"):
            voice_enabled = "ON" in user_input_raw.upper()
            print(f"Voice Output: {'ENABLED' if voice_enabled else 'DISABLED'}")
            continue

        if user_input_raw.startswith("/Language"):
            parts = user_input_raw.split()
            if len(parts) > 1 and parts[1].lower() in SUPPORTED_LANGS:
                session_lang = SUPPORTED_LANGS[parts[1].lower()]
                print(f"Language session changed to: {parts[1].lower()}")
            else:
                print("Supported: " + ", ".join(SUPPORTED_LANGS.keys()))
            continue

        if user_input_raw.upper() == "/IMAGE":
            await handle_image_mode()
            continue

        user_prompt = user_input_raw
        if session_lang != "en":
            try:
                user_prompt = GoogleTranslator(source=session_lang, target="en").translate(user_input_raw)
            except Exception:
                user_prompt = user_input_raw

        print("Cinny-Agent thinking...")
        final_answer = await agent.plan_and_execute(user_prompt)


        output_answer = final_answer
        if session_lang != "en":
            try:
                output_answer = GoogleTranslator(source="en", target=session_lang).translate(final_answer)
            except Exception:
                output_answer = final_answer

        print(output_answer)
        speak_text(output_answer, session_lang)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession Terminated.")
