
from dotenv import load_dotenv
import os
from models import Models
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
    llm_api_key: str | None 
    tavily_api_key: str | None = None
    is_dev: bool = True
    model: Models = Models.CHAT 
    max_history_messages: int = 50
    autonomous_risk: bool = False
    input_price_per_mtok: float = 0.50
    output_price_per_mtok: float = 1.50

    def __init__(self):
        api_key = os.getenv("LLM_KEY")
        if not api_key:
            raise RuntimeError("LLM API key is Required!")
        is_dev = os.getenv("enviroment") 
        self.llm_api_key = api_key
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.is_dev = True if is_dev else False
        
        max_hist = os.getenv("MAX_HISTORY_MESSAGES")
        if max_hist and max_hist.isdigit():
            self.max_history_messages = int(max_hist)
            
        auto_risk = os.getenv("AUTONOMOUS_RISK")
        if auto_risk and auto_risk.lower() in ("true", "1", "yes"):
            self.autonomous_risk = True

        self.input_price_per_mtok = float(os.getenv("INPUT_PRICE_PER_MTOK", "0.50"))
        self.output_price_per_mtok = float(os.getenv("OUTPUT_PRICE_PER_MTOK", "1.50"))
