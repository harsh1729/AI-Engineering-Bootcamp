from abc import ABC, abstractmethod
import random
import time
from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeVar

from llm_sdk.config import LLM_MAX_RETRIES, LLM_RETRY_BASE_DELAY_SECONDS
from llm_sdk.logs import get_logger
from llm_sdk.models import LLMRequest, LLMResponse, LLMResponseChunk
from llm_sdk.models.tools import LLMToolCall, LLMToolExecutionResult
from llm_sdk.tool_functions import ToolRegistry
from llm_sdk.tool_functions.web_search import WEB_SEARCH_UNAVAILABLE_MESSAGE

from concurrent.futures import ThreadPoolExecutor

logger = get_logger(__name__)
TOOL_TIMEOUT_SECONDS = 20

T = TypeVar("T")


class LLMProvider(ABC):

    @abstractmethod
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        request: LLMRequest,
    ) -> Generator[LLMResponseChunk, None, None]:
        """Generate a response stream for the given prompt."""
        pass

    def _get_retry_targets(self, request: LLMRequest) -> list[str | None]:
        """Return targets to try in order. None retries the same request without swapping models."""

        return [request.model]

    def _is_retryable_error(self, exc: Exception) -> bool:
        return False

    def _retry_error_label(self, exc: Exception) -> str | int:
        return getattr(exc, "code", type(exc).__name__)

    def _backoff_delay(self, attempt: int) -> float:
        base = LLM_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
        return base + random.uniform(0, base * 0.25)

    def _call_with_retry(
        self,
        call_fn: Callable[[str | None], T],
        *,
        request: LLMRequest,
        operation: str,
    ) -> T:

        last_exc = None

        for target in self._get_retry_targets(request):
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    return call_fn(target)
                except Exception as exc:
                    if not self._is_retryable_error(exc):
                        logger.exception("%s failed.", operation)
                        raise

                    last_exc = exc
                    logger.warning(
                        "%s target=%s attempt %s/%s failed with %s. Retrying...",
                        operation,
                        target or "default",
                        attempt + 1,
                        LLM_MAX_RETRIES,
                        self._retry_error_label(exc),
                    )

                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(self._backoff_delay(attempt))

            logger.warning(
                "%s target=%s exhausted retries. Trying next target...",
                operation,
                target or "default",
            )

        logger.exception("%s failed after all retries and fallback targets.", operation)
        raise last_exc

    def _stream_with_retry(
        self,
        stream_fn: Callable[[str | None], Iterable[T]],
        *,
        request: LLMRequest,
        operation: str,
    ) -> Generator[T, None, None]:

        last_exc = None

        for target in self._get_retry_targets(request):
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    for item in stream_fn(target):
                        yield item
                    return
                except Exception as exc:
                    if not self._is_retryable_error(exc):
                        logger.exception("%s failed.", operation)
                        raise

                    last_exc = exc
                    logger.warning(
                        "%s target=%s attempt %s/%s failed with %s. Retrying...",
                        operation,
                        target or "default",
                        attempt + 1,
                        LLM_MAX_RETRIES,
                        self._retry_error_label(exc),
                    )

                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(self._backoff_delay(attempt))

            logger.warning(
                "%s target=%s exhausted retries. Trying next target...",
                operation,
                target or "default",
            )

        logger.exception("%s failed after all retries and fallback targets.", operation)
        raise last_exc

    def _execute_tools(
    self,
    tool_calls: list[LLMToolCall],
    ) -> list[LLMToolExecutionResult]:
        
      
        tool_results = []

        for tool_call in tool_calls:

            tool = ToolRegistry.get(tool_call.name)

            start = time.perf_counter()

            logger.info(
                "Executing tool '%s' with arguments %s",
                tool_call.name,
                tool_call.arguments,
            )
        

            try:
                result = self._execute_tool_with_timeout(tool, tool_call.arguments)

                elapsed = time.perf_counter() - start

                logger.info(
                    "Tool '%s' completed successfully in %.3f seconds.",
                    tool_call.name,
                    elapsed,
                )

                tool_results.append(
                    LLMToolExecutionResult(
                        tool_call=tool_call,
                        result=result,
                    )
                )

            except Exception as ex:
                elapsed = time.perf_counter() - start

                logger.exception(
                    "Tool '%s' failed after %.3f seconds. %s: %s",
                    tool_call.name,
                    elapsed,
                    type(ex).__name__,
                    ex,
                )

                if tool_call.name == "web_search":
                    tool_results.append(
                        LLMToolExecutionResult(
                            tool_call=tool_call,
                            result=WEB_SEARCH_UNAVAILABLE_MESSAGE,
                        )
                    )
                else:
                    tool_results.append(
                        LLMToolExecutionResult(
                            tool_call=tool_call,
                            error=str(ex),
                        )
                    )


        return tool_results
    

        
    #HARSH: we wait for TOOL_TIMEOUT_SECONDS for tool execution else raise Timeout error
    def _execute_tool_with_timeout(self, tool : Callable, arguments : dict[str, Any],) -> Any:
         
        """
        HARSH Note:
        If orchestration-level timeouts are implemented in the future,
        avoid creating a new ThreadPoolExecutor for every execution using
        a 'with' block. The context manager waits for worker threads to
        finish during shutdown, which defeats the timeout. Instead, use a
        shared executor with proper lifecycle management.
        """
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool, **arguments)
            return future.result(timeout=TOOL_TIMEOUT_SECONDS)
