
from dotenv import load_dotenv
import os
from models import Models
from memory import Memory
load_dotenv()


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price: float,
    output_price: float,
) -> float:
    """Estimate USD cost from token counts and per-million-token prices."""
    return (
        (prompt_tokens / 1_000_000) * input_price
        + (completion_tokens / 1_000_000) * output_price
    )


class Config:
    tavily_api_key: str | None = None
    is_dev: bool = True
    model: Models = Models.CHAT 
    max_history_messages: int = 50
    autonomous_risk: bool = False
    input_price_per_mtok: float = 0.50
    output_price_per_mtok: float = 1.50
    api_key:str | None = None

    def __init__(self):
        auth_path = Memory.root / "auth.json"
        auth_json = Memory.read_from_json(auth_path) or {}

        credentials = auth_json.get("credentials") or {}
        llm_settings = auth_json.get("llm_settings") or {}

        api_key = credentials.get("api_key") or os.getenv("LLM_KEY")
       
        self.api_key = api_key

        self.provider = (
            llm_settings.get("provider")
            or os.getenv("LLM_PROVIDER", "mistral")
        ).lower()

        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        env_mode = os.getenv("ENVIRONMENT") or os.getenv("enviroment")
        self.is_dev = env_mode.lower() in ("dev", "development", "true", "1") if env_mode else True

        max_hist = os.getenv("MAX_HISTORY_MESSAGES")
        if max_hist and max_hist.isdigit():
            self.max_history_messages = int(max_hist)

        auto_risk = os.getenv("AUTONOMOUS_RISK")
        if auto_risk:
            self.autonomous_risk = auto_risk.lower() in ("true", "1", "yes")

        try:
            self.input_price_per_mtok = float(os.getenv("INPUT_PRICE_PER_MTOK", "0.50"))
        except ValueError:
            self.input_price_per_mtok = 0.50

        try:
            self.output_price_per_mtok = float(os.getenv("OUTPUT_PRICE_PER_MTOK", "1.50"))
        except ValueError:
            self.output_price_per_mtok = 1.50