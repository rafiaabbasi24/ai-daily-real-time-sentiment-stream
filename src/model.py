import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from river import compose, linear_model, metrics, preprocessing, optim

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AdaptiveSentimentModel:
    """
    Incremental multilingual sentiment analyzer.

    The model consists of two parts:

    1. **Embedding extractor** – a pretrained multilingual transformer from
       HuggingFace (default: ``sentence-transformers/paraphrase-multilingual-
       MiniLM-L12-v2``). The transformer is used in inference mode only; the
       extracted sentence embeddings are fed to the online learner.

    2. **Online learner** – a River ``LogisticRegression`` model wrapped in a
       small pipeline that normalises the features. The learner can be updated
       with ``learn`` (partial‑fit) and queried with ``predict`` in real time.

    The class provides utilities for batch evaluation, persistence and
    convenient conversion between raw text and the feature representation
    required by River.
    """

    #: Keys used when serialising the model directory.
    _CONFIG_FILE = "config.json"
    _RIVER_MODEL_FILE = "river_model.joblib"

    def __init__(
        self,
        transformer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        learning_rate: float = 0.1,
        l2: float = 1e-5,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Initialise the adaptive sentiment model.

        Parameters
        ----------
        transformer_name: str
            HuggingFace model identifier for the multilingual encoder.
        learning_rate: float
            Learning rate for the River ``LogisticRegression`` optimiser.
        l2: float
            L2 regularisation strength for the River model.
        device: str | torch.device
            Device on which the transformer runs (``'cpu'`` or ``'cuda'``).
        """
        self.transformer_name = transformer_name
        self.device = torch.device(device)

        # Load transformer and tokenizer
        logger.info("Loading transformer %s on %s", transformer_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.transformer = AutoModel.from_pretrained(transformer_name).to(self.device)
        self.transformer.eval()

        # River online learner pipeline
        # The pipeline expects a mapping of feature_name -> value.
        # We generate feature names ``f0``, ``f1`` … from the embedding vector.
        self.river_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(
                optimizer=optim.SGD(learning_rate=learning_rate),
                l2=l2,
                intercept_init=0.0,
                intercept_lr=learning_rate,
            ),
        )
        # Metric for optional evaluation
        self.metric = metrics.Accuracy()

    # --------------------------------------------------------------------- #
    # Embedding utilities
    # --------------------------------------------------------------------- #
    def _embed(self, text: str) -> np.ndarray:
        """
        Convert a single piece of text into a fixed‑size embedding.

        The implementation follows the usual mean‑pooling strategy used by
        sentence‑transformers.

        Parameters
        ----------
        text: str
            Input sentence/document.

        Returns
        -------
        np.ndarray
            1‑D array with the embedding.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            model_output = self.transformer(**inputs)

        # Mean pooling (ignore padding tokens)
        token_embeddings = model_output.last_hidden_state  # (1, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.squeeze(0).cpu().numpy()

    def _vector_to_dict(self, vec: np.ndarray) -> Dict[str, float]:
        """
        Transform a dense vector into the ``feature_name -> value`` mapping
        required by River.

        Parameters
        ----------
        vec: np.ndarray
            1‑D embedding vector.

        Returns
        -------
        dict[str, float]
            Mapping where keys are ``f0``, ``f1`` ….
        """
        return {f"f{i}": float(v) for i, v in enumerate(vec)}

    # --------------------------------------------------------------------- #
    # Public API – online learning & inference
    # --------------------------------------------------------------------- #
    def predict_proba(self, text: str) -> Tuple[float, float]:
        """
        Return the probability for the negative and positive class.

        The underlying River model produces a probability for the positive
        class (label ``1``). The negative probability is ``1 - p_pos``.

        Parameters
        ----------
        text: str
            Input sentence.

        Returns
        -------
        Tuple[float, float]
            ``(p_negative, p_positive)``.
        """
        embedding = self._embed(text)
        features = self._vector_to_dict(embedding)
        p_pos = self.river_model.predict_proba_one(features)[1]
        return 1.0 - p_pos, p_pos

    def predict(self, text: str) -> int:
        """
        Predict the sentiment label for a single text.

        Returns ``1`` for positive sentiment and ``0`` for negative.

        Parameters
        ----------
        text: str
            Input sentence.

        Returns
        -------
        int
            Predicted label.
        """
        p_neg, p_pos = self.predict_proba(text)
        return int(p_pos >= 0.5)

    def learn(self, text: str, label: int) -> None:
        """
        Incrementally update the model with a new labelled example.

        Parameters
        ----------
        text: str
            Input sentence.
        label: int
            Ground‑truth sentiment (``0`` or ``1``).
        """
        if label not in (0, 1):
            raise ValueError("label must be 0 (negative) or 1 (positive)")

        embedding = self._embed(text)
        features = self._vector_to_dict(embedding)
        self.river_model.learn_one(features, label)
        # Update metric for optional monitoring
        self.metric.update(label, self.river_model.predict_one(features))

    # --------------------------------------------------------------------- #
    # Batch utilities
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a DataFrame and return common metrics.

        The method processes the data in batches to avoid excessive GPU memory
        usage. After each batch the model is *not* updated – this function is
        purely evaluative.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame containing at least ``text_column`` and ``label_column``.
        text_column: str
            Name of the column with raw text.
        label_column: str
            Name of the column with integer labels (0/1).
        batch_size: int
            Number of rows processed simultaneously.

        Returns
        -------
        dict[str, float]
            Dictionary with ``accuracy`` and ``balanced_accuracy``.
        """
        if text_column not in data.columns or label_column not in data.columns:
            raise KeyError(f"DataFrame must contain columns {text_column} and {label_column}")

        accuracy_metric = metrics.Accuracy()
        bal_acc_metric = metrics.BalancedAccuracy()

        for start in range(0, len(data), batch_size):
            batch = data.iloc[start : start + batch_size]
            texts = batch[text_column].astype(str).tolist()
            labels = batch[label_column].astype(int).tolist()

            embeddings = self._batch_embed(texts)
            for emb, y in zip(embeddings, labels):
                feats = self._vector_to_dict(emb)
                y_pred = self.river_model.predict_one(feats)
                accuracy_metric.update(y, y_pred)
                bal_acc_metric.update(y, y_pred)

        return {
            "accuracy": accuracy_metric.get(),
            "balanced_accuracy": bal_acc_metric.get(),
        }

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Efficiently embed a list of texts using the transformer in a single
        forward pass (where possible). The function respects the maximum token
        length of the underlying model.

        Parameters
        ----------
        texts: List[str]
            Sentences to embed.

        Returns
        -------
        List[np.ndarray]
            List of embedding vectors (same order as ``texts``).
        """
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            model_output = self.transformer(**inputs)

        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)  # (batch, hidden_dim)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        mean_pooled = summed / counts  # (batch, hidden_dim)

        return [vec.cpu().numpy() for vec in mean_pooled]

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, directory: Union[str, Path]) -> None:
        """
        Persist the model to ``directory``. The directory is created if it does
        not exist.

        The transformer is *not* saved (it can be re‑loaded from its name);
        only the River model and configuration are stored.

        Parameters
        ----------
        directory: str | Path
            Target directory.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save River model
        river_path = path / self._RIVER_MODEL_FILE
        joblib.dump(self.river_model, river_path)
        logger.info("River model saved to %s", river_path)

        # Save configuration
        config = {
            "transformer_name": self.transformer_name,
            "device": str(self.device),
        }
        config_path = path / self._CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved to %s", config_path)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "AdaptiveSentimentModel":
        """
        Load a previously saved model from ``directory``.

        Parameters
        ----------
        directory: str | Path
            Directory containing the saved model.

        Returns
        -------
        AdaptiveSentimentModel
            Re‑instantiated model ready for inference or further learning.
        """
        path = Path(directory)
        config_path = path / cls._CONFIG_FILE
        river_path = path / cls._RIVER_MODEL_FILE

        if not config_path.is_file() or not river_path.is_file():
            raise FileNotFoundError("Model files not found in the specified directory")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        instance = cls(
            transformer_name=config["transformer_name"],
            device=config.get("device", "cpu"),
        )
        instance.river_model = joblib.load(river_path)
        logger.info("Model loaded from %s", directory)
        return instance

    # --------------------------------------------------------------------- #
    # Convenience methods for FastAPI integration
    # --------------------------------------------------------------------- #
    def as_fastapi_router(self, prefix: str = "/sentiment") -> Any:
        """
        Create a minimal FastAPI router exposing ``/predict`` and ``/learn``
        endpoints. The router can be mounted on an existing FastAPI app.

        Example
        -------
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> model = AdaptiveSentimentModel()
        >>> app.include_router(model.as_fastapi_router())
        """
        from fastapi import APIRouter, Body, HTTPException

        router = APIRouter(prefix=prefix)

        @router.post("/predict")
        def predict_endpoint(text: str = Body(..., embed=True)):
            try:
                label = self.predict(text)
                prob = self.predict_proba(text)[1]
                return {"label": label, "positive_probability": prob}
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        @router.post("/learn")
        def learn_endpoint(payload: Dict[str, Any] = Body(...)):
            try:
                text = payload["text"]
                label = int(payload["label"])
                self.learn(text, label)
                return {"status": "updated"}
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=f"Missing field {exc}")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        return router
