from llm_sdk.models import LLMRequest,LLMMessage
from llm_sdk.models.tools import LLMTool,LLMToolParam,LLMToolExecutionResult
from llm_sdk.serializers.tool_output import format_tool_output_for_llm
from llm_sdk.enums import MessageRole

from llm_sdk.serializers import BaseRequestSerializer
from llm_sdk.config import ANTHROPIC_MODEL
import json


class ClaudeRequestSerializer(BaseRequestSerializer):

    def serialize(
        self,
        request: LLMRequest,
    ) -> dict:
    
        payload = {
            "model": request.model or ANTHROPIC_MODEL,
            "max_tokens": request.max_tokens,
            "messages": self._serialize_messages(request.messages),
        }

        system = self._serialize_system_messages(request.messages)
        if system is not None:
            payload["system"] = system

        if request.tools:
            payload["tools"] = self._serialize_tools(request.tools)

        return payload
    
    def _serialize_messages(
    self,
    messages: list[LLMMessage],
    ) -> list[dict]:
        
        
        claude_messages = []

        for message in messages:
            if message.role != MessageRole.SYSTEM:
                claude_messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        return claude_messages
    
    def _serialize_system_messages(
    self,
    messages: list[LLMMessage],
    ) -> str | None:
        
        system_messages = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                system_messages.append(message.content)

        return "\n\n".join(system_messages) if system_messages else None
    

    def _serialize_tools(
    self,
    tools: list[LLMTool],
    ) -> list[dict]:

        return [
            self._serialize_tool(tool)
            for tool in tools
        ]
    

    def _serialize_tool(
            self, 
            tool:LLMTool) -> dict:

        properties = {}

        required = []

        for param in tool.parameters:
            properties[param.name] = self._serialize_parameter(param)
            if param.required:
                required.append(param.name)
        
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type":"object",
                "properties":properties,
                "required":required,
            },
        }


    def _serialize_parameter(
    self,
    parameter: LLMToolParam,
    ) -> dict:

        return {
            "type": parameter.type,
            "description": parameter.description,
        }
    

    def serialize_tool_results(
    self,
    request: LLMRequest,
    assistant_content: list[dict],
    tool_results: list[LLMToolExecutionResult],
    ) -> dict:

        messages = self._serialize_messages(request.messages)

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        messages.append(
            self._serialize_tool_results_message(tool_results)
        )

        payload = {
            "model": request.model or ANTHROPIC_MODEL,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }

        system = self._serialize_system_messages(request.messages)
        if system is not None:
            payload["system"] = system

        if request.tools:
            payload["tools"] = self._serialize_tools(request.tools)

        return payload
    
    def _serialize_tool_results_message(
        self,
        tool_results: list[LLMToolExecutionResult],
    ) -> dict:

        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_result.tool_call.call_id,
                    "content": format_tool_output_for_llm(tool_result),
                }
                for tool_result in tool_results
            ],
        }
    

 
   