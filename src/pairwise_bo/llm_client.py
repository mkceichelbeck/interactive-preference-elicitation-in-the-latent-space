from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, Protocol, Type, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from ollama import Client as OllamaClient
from ollama import ResponseError as OllamaResponseError
from pydantic import BaseModel, ValidationError

from pairwise_bo.logging_utils import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class LLMUsageMetadata:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(slots=True)
class LLMResponse:
    text: str
    parsed: Any | None = None
    usage: LLMUsageMetadata | None = None


class LLMClientError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMClientAdapter(Protocol):
    model_name: str
    backend: ClassVar[str]
    verbose: bool

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        timeout: Optional[int] = None,
    ) -> LLMResponse: ...

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Type[T],
        temperature: float = 0.8,
        timeout: Optional[int] = None,
    ) -> LLMResponse: ...


class GoogleGenAIAdapter:
    backend: ClassVar[str] = "google"

    def __init__(self, *, api_key: str, model_name: str, verbose: bool = False) -> None:
        if not api_key:
            raise ValueError("api_key must be provided for GoogleGenAIAdapter.")
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.verbose = verbose

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        if self.verbose:
            logger.info(
                f"[GoogleGenAI] Request to {self.model_name}: prompt={prompt[:100]}..., temperature={temperature}"
            )
        config = genai_types.GenerateContentConfig(temperature=temperature)  # type: ignore[call-arg]
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except genai_errors.APIError as exc:
            retryable = getattr(exc, "code", None) == 429
            raise LLMClientError(str(exc), retryable=retryable) from exc
        text = response.text or ""
        usage = _google_usage_from_metadata(getattr(response, "usage_metadata", None))
        if self.verbose:
            logger.info(f"[GoogleGenAI] Response: text={text[:100]}..., usage={usage}")
        return LLMResponse(text=text, usage=usage)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Type[T],
        temperature: float = 0.8,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        if self.verbose:
            logger.info(
                f"[GoogleGenAI] JSON Request to {self.model_name}: prompt={prompt[:100]}..., temperature={temperature}, schema={schema.__name__}"
            )
        config = genai_types.GenerateContentConfig(  # type: ignore[call-arg]
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except genai_errors.APIError as exc:
            retryable = getattr(exc, "code", None) == 429
            raise LLMClientError(str(exc), retryable=retryable) from exc
        text = response.text or ""
        parsed = schema.model_validate_json(text)
        usage = _google_usage_from_metadata(getattr(response, "usage_metadata", None))
        if self.verbose:
            logger.info(
                f"[GoogleGenAI] JSON Response: text={text[:100]}..., usage={usage}"
            )
        return LLMResponse(text=text, parsed=parsed, usage=usage)


class OllamaAdapter:
    backend: ClassVar[str] = "ollama"

    def __init__(self, *, host: str, model_name: str, verbose: bool = False) -> None:
        self._client = OllamaClient(host=host)
        self.model_name = model_name
        self.verbose = verbose

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.8,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        if self.verbose:
            logger.info(
                f"[Ollama] Request to {self.model_name}: prompt={prompt[:100]}..., temperature={temperature}"
            )
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
        except OllamaResponseError as exc:
            raise LLMClientError(str(exc), retryable=False) from exc
        text = response.get("message", {}).get("content", "")
        usage = _ollama_usage_from_metadata(response)
        if self.verbose:
            logger.info(f"[Ollama] Response: text={text[:100]}..., usage={usage}")
        return LLMResponse(text=text, usage=usage)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Type[T],
        temperature: float = 0.8,
        timeout: Optional[int] = None,
    ) -> LLMResponse:
        schema_instruction = _get_schema_instruction(schema)
        enhanced_prompt = prompt + schema_instruction
        if self.verbose:
            logger.info(
                f"[Ollama] JSON Request to {self.model_name}: prompt={prompt[:100]}..., temperature={temperature}, schema={schema.__name__}"
            )
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": enhanced_prompt}],
                options={"temperature": temperature},
                format="json",
            )
        except OllamaResponseError as exc:
            raise LLMClientError(str(exc), retryable=False) from exc
        raw = response.get("message", {}).get("content", "")
        parsed = _parse_json_response(raw, schema)
        usage = _ollama_usage_from_metadata(response)
        if self.verbose:
            logger.info(f"[Ollama] JSON Response: text={raw[:100]}..., usage={usage}")
        return LLMResponse(text=raw, parsed=parsed, usage=usage)


def _google_usage_from_metadata(metadata: Any) -> LLMUsageMetadata | None:
    if metadata is None:
        return None
    prompt = getattr(metadata, "prompt_token_count", None)
    completion = getattr(metadata, "candidates_token_count", None)
    total = getattr(metadata, "total_token_count", None)
    return LLMUsageMetadata(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _ollama_usage_from_metadata(metadata: Mapping[str, Any] | Any) -> LLMUsageMetadata:
    prompt = _safe_get(metadata, "prompt_eval_count")
    completion = _safe_get(metadata, "eval_count")
    total = None
    if prompt is not None or completion is not None:
        total = (prompt or 0) + (completion or 0)
    return LLMUsageMetadata(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _safe_get(metadata: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _parse_json_response(raw: str, schema: Type[T]) -> T:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fence(cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from Ollama response: %s", raw)
        raise LLMClientError("Ollama response was not valid JSON") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.error("Failed to validate JSON against schema: %s", data)
        logger.error(exc)
        raise LLMClientError("Parsed JSON did not match expected schema") from exc


def _strip_code_fence(raw: str) -> str:
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return raw
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _get_schema_instruction(schema: Type[BaseModel]) -> str:
    try:
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)
        return f"\n\nYou must respond with valid JSON matching this schema:\n{schema_str}\n\nRespond only with the JSON object, no additional text."
    except Exception as exc:
        logger.warning(f"Failed to generate schema instruction: {exc}")
        return "\n\nRespond with valid JSON only."


def parse_model_name(model_name: str) -> tuple[str, str]:
    if ":" not in model_name:
        raise ValueError(
            "LLM model name must include a provider prefix (gemini: or ollama:)."
        )
    prefix, actual = model_name.split(":", 1)
    provider = prefix.strip().lower()
    clean_name = actual.strip()
    if not provider or not clean_name:
        raise ValueError(
            "LLM model name must include a non-empty provider prefix and model identifier."
        )
    return provider, clean_name


def build_llm_client(
    model_name: str,
    api_key: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    verbose: bool = False,
) -> LLMClientAdapter:
    provider, clean_name = parse_model_name(model_name)
    if provider == "gemini":
        if not api_key:
            raise ValueError("api_key must be provided for Gemini models.")
        return GoogleGenAIAdapter(
            api_key=api_key,
            model_name=clean_name,
            verbose=verbose,
        )
    if provider == "ollama":
        return OllamaAdapter(
            host=ollama_host, model_name=clean_name, verbose=verbose
        )
    raise ValueError(
        "Unsupported LLM provider prefix. Use gemini: or ollama:."
    )
