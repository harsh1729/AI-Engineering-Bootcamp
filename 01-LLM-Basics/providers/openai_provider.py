from openai import OpenAI
from typing import Generator,Any
from collections.abc import Callable
import time
from openai.types.responses import ResponseCreatedEvent
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from config import OPENAI_API_KEY
from providers import LLMProvider
from models import LLMResponse,LLMRequest,LLMResponseChunk
from serializers import OpenAIRequestSerializer,OpenAIResponseSerializer,OpenAIResponseChunkSerializer
from tool_functions import ToolRegistry
from models.tools import LLMToolCall,LLMToolExecutionResult
from logs import get_logger


logger = get_logger(__name__)
TOOL_TIMEOUT_SECONDS = 10

class OpenAIProvider(LLMProvider):

    

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.request_serializer = OpenAIRequestSerializer()
        self.response_serializer = OpenAIResponseSerializer()
        self.response_chunk_serializer = OpenAIResponseChunkSerializer()

    # Generate a complete response for the given prompt.
    def generate(self, request: LLMRequest) -> LLMResponse:

        response = self._call_llm(request)
        return self._finalize_response(request,response)


    def _call_llm(self, request: LLMRequest)-> LLMResponse:

        payload = self.request_serializer.serialize(request)

        response = self.client.responses.create(**payload)

        return self.response_serializer.serialize(response)
    
    
    def _finalize_response(self,request: LLMRequest, response: LLMResponse)-> LLMResponse:

        while response.tool_calls:


            tool_results = self._execute_tools(response.tool_calls)

            payload = self.request_serializer.serialize_function_outputs(
                request=request,
                previous_response_id=response.id,
                tool_results=tool_results,
            )

            response = self.response_serializer.serialize(
                self.client.responses.create(**payload)
            )

        return response
    


     # Generate a stream of response for the given prompt.
    def generate_stream(
    self,
    request: LLMRequest,
    )-> Generator[LLMResponseChunk, None, None]:
        

        stream = self._call_llm_stream(request)

        yield from self._finalize_stream(request,stream)

        # for event in stream:
        #     chunk = self.response_chunk_serializer.serialize(event)
        #     if chunk:
        #         yield chunk


    def _call_llm_stream(self, request: LLMRequest)-> LLMResponse:

        payload = self.request_serializer.serialize(request)
         
        payload["stream"] = True
         
        return self.client.responses.create(**payload)
    

    def _finalize_stream(
    self,
    request: LLMRequest,
    stream,
    ) -> Generator[LLMResponseChunk, None, None]:

        MAX_TOOL_CALL_DEPTH = 10

        for _ in range(MAX_TOOL_CALL_DEPTH):

            stream_response_id = None
            tool_calls: list[LLMToolCall] = []

            for event in stream:

                if isinstance(event, ResponseCreatedEvent):
                    stream_response_id = event.response.id
                    continue

                chunk = self.response_chunk_serializer.serialize(event)

                if not chunk:
                    continue

                if chunk.text:
                    yield chunk

                if chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                    continue

            if len(tool_calls) > 0:
                tool_results = self._execute_tools(tool_calls)

                payload = self.request_serializer.serialize_function_outputs(
                    request=request,
                    previous_response_id=stream_response_id,
                    tool_results=tool_results,
                )

                stream = self.client.responses.create(
                    **payload,
                    stream=True,
                )

                continue

            # No tool calls OR chunk.text means we're done 
            return

        raise RuntimeError(
            f"Maximum tool call depth ({MAX_TOOL_CALL_DEPTH}) exceeded."
        )



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
    
    
    