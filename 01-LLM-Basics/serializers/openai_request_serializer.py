from models import LLMMessage,LLMRequest
from models.tools import LLMTool, LLMToolParam
from serializers import BaseRequestSerializer
from config import OPENAI_MODEL
from enums import MessageRole


class OpenAIRequestSerializer(BaseRequestSerializer):
    
    def serialize(
    self,
    request: LLMRequest,
    ) -> dict:
        return {
            "model": OPENAI_MODEL,
            "input": self._serialize_messages(request.messages),
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "tools": self._serialize_tools(request.tools),
        }
    

    def _serialize_messages(
    self,
    messages: list[LLMMessage],
    ) -> list[dict]:

        return [
            self._serialize_message(message)
            for message in messages
        ]
    

    def _serialize_message(
    self,
    message: LLMMessage,
    ) -> dict:

        
        if message.role == MessageRole.TOOL:
            return self._serialize_tool_message(message)

        if message.tool_calls:
            return self._serialize_assistant_tool_call_message(message)

        return self._serialize_chat_message(message)
    

    def _serialize_chat_message(
    self,
    message: LLMMessage,
    ) -> dict:

        return {
            "role": message.role.value,
            "content": message.content,
        }

    
    def serialize_function_outputs(
    self,
    request: LLMRequest,
    previous_response_id: str,
    tool_results: list[dict],
    ) -> dict:
        return {
            "model": OPENAI_MODEL,
            "previous_response_id": previous_response_id,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": tool_result["tool_call"].call_id,
                    "output": str(tool_result["result"]),
                }
                for tool_result in tool_results
            ],
        }
    

    def _serialize_tools(
    self,
    tools: list[LLMTool],
    ) -> list[dict]:

        return [
            self._serialize_tool(tool)
            for tool in tools
        ]
    

    def _serialize_tool(self, tool:LLMTool) -> dict:
        properties = {}

        required = []

        for param in tool.parameters:
            properties[param.name] = self._serialize_parameter(param)
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {
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
    
