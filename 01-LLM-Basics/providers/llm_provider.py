from abc import ABC, abstractmethod
from typing import Generator,Any
from models import LLMRequest,LLMResponse,LLMResponseChunk
import time
from tool_functions import ToolRegistry
from collections.abc import Callable
from models.tools import LLMToolCall,LLMToolExecutionResult

from concurrent.futures import ThreadPoolExecutor, TimeoutError

from logs import get_logger


logger = get_logger(__name__)
TOOL_TIMEOUT_SECONDS = 10

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

                result = tool(**tool_call.arguments)
                
            
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