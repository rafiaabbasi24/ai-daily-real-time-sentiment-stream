import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from river import compose, feature_extraction, naive_bayes, metrics, preprocessing
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("sentiment_analyzer")

# --------------------------------------------------------------------------- #
# Pydantic models for request/response validation
# --------------------------------------------------------------------------- #


class PredictRequest(BaseModel):
    """Schema for a single prediction request."""

    text: str = Field(..., min_length=1, description="The tweet or text to analyze.")


class PredictResponse(BaseModel):
    """Schema for the prediction response."""

    transformer_label: str
    transformer_score: float
    river_label: Optional[str] = None
    river_score: Optional[float] = None


# --------------------------------------------------------------------------- #
# Core sentiment analysis component
# --------------------------------------------------------------------------- #


class SentimentModel:
    """
    Wrapper around a multilingual transformer pipeline and an online River model.

    The transformer provides high‑quality multilingual sentiment predictions,
    while the River model is updated in real‑time with user feedback to adapt
    to domain‑specific language.
    """

    def __init__(self) -> None:
        # Load a multilingual sentiment model from HuggingFace.
        # The chosen model returns labels like "1 star", "2 stars", … "5 stars".
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        logger.info("Loading transformer model %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.transformer_pipe = pipeline(
            "sentiment-analysis",
            model=self.transformer,
            tokenizer=self.tokenizer,
            return_all_scores=False,
        )

        # Initialise an online River pipeline.
        # Bag‑of‑words → Naïve Bayes (multinomial) with a running accuracy metric.
        logger.info("Initialising River online model")
        self.river_pipe = compose.Pipeline(
            feature_extraction.BagOfWords(),
            naive_bayes.MultinomialNB(),
        )
        self.river_metric = metrics.Accuracy()

    # ------------------------------------------------------------------- #
    # Transformer inference
    # ------------------------------------------------------------------- #
    def _transformer_predict(self, text: str) -> Dict[str, Any]:
        """
        Run the transformer pipeline on ``text`` and return the top label and its score.
        """
        try:
            result = self.transformer_pipe(text)[0]  # type: ignore[index]
            label = result["label"]
            score = float(result["score"])
            logger.debug("Transformer prediction: %s (%.4f)", label, score)
            return {"label": label, "score": score}
        except Exception as exc:
            logger.exception("Transformer inference failed")
            raise RuntimeError("Transformer inference error") from exc

    # ------------------------------------------------------------------- #
    # River inference
    # ------------------------------------------------------------------- #
    def _river_predict(self, text: str) -> Dict[str, Any]:
        """
        Predict with the online River model. Returns the most likely label and its
        probability. If the model has not been trained yet, returns ``None`` values.
        """
        # River works with dictionaries of features.
        features = {"text": text}
        try:
            proba = self.river_pipe.predict_proba_one(features)
            if not proba:
                # Model has not seen any data.
                logger.debug("River model untrained – returning None")
                return {"label": None, "score": None}
            # Choose the label with highest probability.
            label, score = max(proba.items(), key=lambda item: item[1])
            logger.debug("River prediction: %s (%.4f)", label, score)
            return {"label": label, "score": float(score)}
        except Exception as exc:
            logger.exception("River inference failed")
            raise RuntimeError("River inference error") from exc

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    def predict(self, text: str) -> PredictResponse:
        """
        Produce a combined prediction from both the transformer and the River model.
        """
        transformer_res = self._transformer_predict(text)
        river_res = self._river_predict(text)

        return PredictResponse(
            transformer_label=transformer_res["label"],
            transformer_score=transformer_res["score"],
            river_label=river_res["label"],
            river_score=river_res["score"],
        )

    def update(self, text: str, label: str) -> None:
        """
        Update the online River model with a new (text, label) pair.
        ``label`` must be a string compatible with the River model's label space.
        """
        try:
            self.river_pipe.learn_one({"text": text}, label)
            # Update running metric for monitoring (optional).
            pred = self.river_pipe.predict_one({"text": text})
            self.river_metric.update(label, pred)
            logger.debug(
                "River model updated with label=%s; running accuracy=%.4f",
                label,
                self.river_metric.get(),
            )
        except Exception as exc:
            logger.exception("Failed to update River model")
            raise RuntimeError("River model update error") from exc


# --------------------------------------------------------------------------- #
# FastAPI application setup
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Real‑Time Adaptive Multilingual Sentiment Analyzer",
    version="1.0.0",
    description=(
        "A FastAPI service that streams tweets, predicts sentiment using a "
        "multilingual transformer model, and adapts online with River."
    ),
)

# Allow calls from browsers or other origins during development / production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise the shared sentiment model (singleton pattern).
sentiment_model = SentimentModel()


# --------------------------------------------------------------------------- #
# Health check endpoint
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["Utility"])
async def health_check() -> Dict[str, str]:
    """Simple health‑check used by orchestration tools."""
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# HTTP prediction endpoint
# --------------------------------------------------------------------------- #
@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """
    Return a sentiment prediction for a single piece of text.

    The endpoint runs both the transformer and the online River model.
    """
    try:
        prediction = sentiment_model.predict(request.text)
        return prediction
    except Exception as exc:
        logger.exception("Prediction endpoint error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# --------------------------------------------------------------------------- #
# WebSocket streaming endpoint
# --------------------------------------------------------------------------- #
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    Bi‑directional WebSocket that receives raw tweet texts and streams back
    sentiment predictions in real time.

    Expected client message format (JSON):
    ``{"text": "some tweet"}``

    Server response format (JSON) follows :class:`PredictResponse`.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                payload = json.loads(raw_msg)
                text = payload.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Missing or empty 'text' field")
            except (json.JSONDecodeError, ValueError) as exc:
                error_msg = {"error": "Invalid payload: must be JSON with a non‑empty 'text' field"}
                await websocket.send_json(error_msg)
                logger.warning("Invalid WebSocket payload: %s", exc)
                continue

            # Perform prediction.
            try:
                prediction = sentiment_model.predict(text)
                await websocket.send_json(prediction.dict())
                logger.debug("Sent prediction for text: %s", text[:30])
            except Exception as exc:
                error_msg = {"error": f"Prediction failed: {str(exc)}"}
                await websocket.send_json(error_msg)
                logger.exception("Prediction failed for incoming WebSocket message")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("Unexpected error in WebSocket handling")
        await websocket.close(code=1011)  # Internal error


# --------------------------------------------------------------------------- #
# Optional endpoint to provide feedback and adapt the River model
# --------------------------------------------------------------------------- #
class FeedbackRequest(BaseModel):
    """Schema for user feedback used to adapt the online model."""

    text: str = Field(..., min_length=1, description="Original tweet text.")
    label: str = Field(..., description="Correct sentiment label (e.g., '1 star').")


@app.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Feedback"],
)
async def feedback_endpoint(feedback: FeedbackRequest) -> None:
    """
    Accept a corrected label for a previously analysed tweet.
    The online River model is updated immediately to improve future predictions.
    """
    try:
        sentiment_model.update(feedback.text, feedback.label)
    except Exception as exc:
        logger.exception("Feedback processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# --------------------------------------------------------------------------- #
# Uvicorn entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Run the application with hot‑reload disabled (production) and a sensible port.
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1,
    )