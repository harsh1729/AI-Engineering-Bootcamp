from google.genai import types

from config import GEMINI_MODEL, GEMINI_THINKING_LEVEL
from enums import MessageRole
from models import LLMMessage, LLMRequest
from models.tools import LLMTool, LLMToolExecutionResult, LLMToolParam
from serializers import BaseRequestSerializer


class GeminiRequestSerializer(BaseRequestSerializer):

    def serialize(
        self,
        request: LLMRequest,
        model: str | None = None,
    ) -> dict:

        return {
            "model": model or GEMINI_MODEL,
            "contents": self._serialize_contents(request.messages),
            "config": self._build_config(request),
        }

    def serialize_tool_results(
        self,
        request: LLMRequest,
        assistant_content: types.Content,
        tool_results: list[LLMToolExecutionResult],
        model: str | None = None,
    ) -> dict:

        contents = self._serialize_contents(request.messages)
        contents.append(assistant_content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_result.tool_call.name,
                            response={
                                "success": tool_result.error is None,
                                "result": tool_result.result,
                                "error": tool_result.error,
                            },
                            id=tool_result.tool_call.call_id,
                        )
                    )
                    for tool_result in tool_results
                ],
            )
        )

        return {
            "model": model or GEMINI_MODEL,
            "contents": contents,
            "config": self._build_config(request),
        }

    def _build_config(
        self,
        request: LLMRequest,
    ) -> types.GenerateContentConfig:

        config_kwargs: dict = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "system_instruction": self._serialize_system_messages(request.messages),
            "thinking_config": types.ThinkingConfig(
                thinking_level=GEMINI_THINKING_LEVEL.value,
            ),
        }

        if request.tools:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=self._serialize_tools(request.tools))
            ]
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        return types.GenerateContentConfig(**config_kwargs)

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
        tool: LLMTool,
    ) -> dict:

        properties = {}
        required = []

        for param in tool.parameters:
            properties[param.name] = self._serialize_parameter(param)
            if param.required:
                required.append(param.name)

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
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

    def _serialize_contents(
        self,
        messages: list[LLMMessage],
    ) -> list[types.Content]:

        contents: list[types.Content] = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                continue

            contents.append(
                types.Content(
                    role=self._serialize_role(message.role),
                    parts=[types.Part(text=message.content)],
                )
            )

        return contents

    def _serialize_role(
        self,
        role: MessageRole,
    ) -> str:

        if role == MessageRole.ASSISTANT:
            return "model"

        return role.value

    def _serialize_system_messages(
        self,
        messages: list[LLMMessage],
    ) -> str | None:

        system_messages = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                system_messages.append(message.content)

        return "\n\n".join(system_messages) if system_messages else None
