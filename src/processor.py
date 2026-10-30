import logging
import re
import itertools
from typing import Generator, Iterable, List, Union, Optional

import pandas as pd
import requests
from requests.exceptions import RequestException
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer

# Configure module level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def fetch_streaming_text(
    url: str,
    timeout: int = 10,
    chunk_size: int = 1024,
    encoding: str = "utf-8",
) -> Generator[str, None, None]:
    """
    Stream text lines from a HTTP endpoint.

    Parameters
    ----------
    url: str
        The URL to request. The endpoint must return a newline‑delimited text
        stream (e.g., a log file, a tweet fire‑hose, etc.).
    timeout: int, default 10
        Seconds to wait for the server to send data before raising a timeout.
    chunk_size: int, default 1024
        Number of bytes to read per iteration. Small values keep memory usage low.
    encoding: str, default "utf-8"
        Text encoding used by the remote source.

    Yields
    ------
    str
        Individual lines of text stripped of trailing newline characters.

    Raises
    ------
    RequestException
        If the request cannot be established or the connection drops.
    """
    logger.info("Opening streaming connection to %s", url)
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_content(chunk_size=chunk_size, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield line.strip()
            # Emit any trailing data that does not end with a newline
            if buffer:
                yield buffer.strip()
    except RequestException as exc:
        logger.error("Failed to stream from %s: %s", url, exc)
        raise


_URL_PATTERN = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
)
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#\w+")
_EMOJI_PATTERN = re.compile(
    "["                     # start character class
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")


def clean_text(text: str, lower: bool = True, remove_urls: bool = True,
               remove_mentions: bool = True, remove_hashtags: bool = True,
               remove_emoji: bool = True, remove_punct: bool = True) -> str:
    """
    Perform a light‑weight cleaning of raw text suitable for downstream
    sentiment analysis.

    The function removes (optionally) URLs, user mentions, hashtags, emojis,
    and punctuation. Whitespace is normalized to a single space.

    Parameters
    ----------
    text: str
        Raw input string.
    lower: bool, default True
        Convert text to lower‑case.
    remove_urls: bool, default True
        Strip URLs.
    remove_mentions: bool, default True
        Strip Twitter‑style ``@username`` mentions.
    remove_hashtags: bool, default True
        Strip ``#hashtag`` tokens.
    remove_emoji: bool, default True
        Strip Unicode emoji characters.
    remove_punct: bool, default True
        Remove punctuation characters.

    Returns
    -------
    str
        Cleaned text.
    """
    original = text
    if remove_urls:
        text = _URL_PATTERN.sub("", text)
    if remove_mentions:
        text = _MENTION_PATTERN.sub("", text)
    if remove_hashtags:
        text = _HASHTAG_PATTERN.sub("", text)
    if remove_emoji:
        text = _EMOJI_PATTERN.sub("", text)
    if remove_punct:
        text = _PUNCTUATION_PATTERN.sub(" ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if lower:
        text = text.lower()

    logger.debug(
        "clean_text: original=%r cleaned=%r", original[:50], text[:50]
    )
    return text


def batch_iterator(
    data: Iterable[Union[str, pd.Series]],
    batch_size: int,
    clean: bool = True,
    cleaner: Optional[callable] = None,
) -> Generator[List[Union[str, pd.Series]], None, None]:
    """
    Yield successive batches from an iterable of text items.

    Parameters
    ----------
    data: Iterable[str] or Iterable[pd.Series]
        Source of raw text items. ``pd.Series`` is accepted to enable direct
        usage with ``DataFrame.itertuples`` or ``Series`` objects.
    batch_size: int
        Number of items per batch.
    clean: bool, default True
        Apply :func:`clean_text` to each element before yielding.
    cleaner: callable, optional
        Custom cleaning function. If omitted and ``clean`` is True,
        :func:`clean_text` is used.

    Yields
    ------
    List[str] or List[pd.Series]
        Batches of (cleaned) text items.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Resolve cleaning function
    if clean:
        clean_fn = cleaner if cleaner is not None else clean_text
    else:
        clean_fn = lambda x: x  # identity

    iterator = iter(data)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        # Apply cleaning to each element
        cleaned_batch = [clean_fn(item) if isinstance(item, str) else clean_fn(str(item)) for item in batch]
        logger.debug("Yielding batch of size %d", len(cleaned_batch))
        yield cleaned_batch


class TextProcessor(BaseEstimator, TransformerMixin):
    """
    Scikit‑learn compatible transformer that fetches, cleans, and batches
    streaming text data. It can be inserted into a ``river`` or ``sklearn``
    pipeline.

    Example
    -------
    >>> processor = TextProcessor(source_url="http://example.com/stream")
    >>> for batch in processor.transform():
    ...     print(batch[:2])
    """

    def __init__(
        self,
        source_url: str,
        batch_size: int = 128,
        tokenizer_name: str = "distilbert-base-multilingual-cased",
        clean: bool = True,
        timeout: int = 10,
    ):
        """
        Parameters
        ----------
        source_url: str
            HTTP endpoint returning a newline‑delimited text stream.
        batch_size: int, default 128
            Number of sentences to collect before yielding.
        tokenizer_name: str, default "distilbert-base-multilingual-cased"
            Name of a HuggingFace tokenizer used for downstream tokenisation.
        clean: bool, default True
            Whether to apply :func:`clean_text` to each incoming line.
        timeout: int, default 10
            Request timeout in seconds.
        """
        self.source_url = source_url
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.clean = clean
        self.timeout = timeout
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def tokenizer(self):
        """Expose the underlying HuggingFace tokenizer."""
        return self._tokenizer

    def fit(self, X=None, y=None):
        """No‑op to satisfy the sklearn API."""
        return self

    def transform(self, X=None):
        """
        Stream, clean, batch, and tokenise the text data.

        Returns
        -------
        Generator[List[dict]]
            Each yielded element is a list of tokenised dictionaries ready for
            model consumption.
        """
        stream = fetch_streaming_text(
            self.source_url, timeout=self.timeout
        )
        for raw_batch in batch_iterator(
            stream,
            batch_size=self.batch_size,
            clean=self.clean,
        ):
            # Tokenise each sentence in the batch
            tokenised = self._tokenizer(
                raw_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # Convert to a list of per‑sample dicts for downstream usage
            batch_dicts = [
                {k: v[i] for k, v in tokenised.items()}
                for i in range(len(raw_batch))
            ]
            logger.debug(
                "Processed batch %d with %d samples", 
                getattr(self, "_batch_counter", 0), len(batch_dicts)
            )
            # Increment counter for debugging
            self._batch_counter = getattr(self, "_batch_counter", 0) + 1
            yield batch_dicts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source_url={self.source_url!r}, "
            f"batch_size={self.batch_size}, tokenizer_name={self.tokenizer_name!r}, "
            f"clean={self.clean})"
        )