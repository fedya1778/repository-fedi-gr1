import logging
import os
from fastapi import FastAPI
from src.models.predictor import ModelPredictor


def _read_log_level_from_config(path: str = "configs/config.yaml") -> str:
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
            lvl = cfg.get("app", {}).get("log_level") if isinstance(cfg, dict) else None
            if lvl:
                return str(lvl)
    except Exception:
        pass

    try:
        with open(path, "r") as f:
            for line in f:
                if "log_level" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
    except Exception:
        pass

    return os.environ.get("LOG_LEVEL", "INFO")


_level_name = _read_log_level_from_config()
logging.basicConfig(
    level=getattr(logging, _level_name.upper(), logging.INFO),
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api_service")

app = FastAPI(title="BNPL Risk Scoring Service")
predictor = ModelPredictor()

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint was pinged")
    return {
        "status": "healthy",
        "model_loaded": getattr(predictor, "model", None) is not None
    }

@app.post("/predict")
async def get_risk_score(client_data: dict):
    logger.info(f"Received prediction request for client. Age: {client_data.get('Customer_Age')}")
    
    try:
        result = predictor.predict(client_data)
        logger.info(f"Prediction successful. Status: {result['status']}, Score: {result['credit_score']}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}", exc_info=True)
        raise e

@app.get("/")
def read_root():
    return {"message": "Скорринговая модель готова к работе!"}