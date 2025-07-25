import os
from time import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from core.registry import register_engine
from core.engine import Engine, EngineConfig
from core.evaluator import is_json_schema_valid
from core.types import (
    Token,
    CompileStatus,
    DecodingStatus,
    GenerationOutput,
    CompileStatusCode,
    DecodingStatusCode,
)

MAX_RETRIES=3
TIMEOUT=30

@dataclass
class OpenAICompatibleConfig(EngineConfig):
    model: str
    tokenizer: str
    provider: str
    base_url: str
    api_key_variable_name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_context_length: Optional[int] = 4096


class OpenAICompatibleEngine(Engine[OpenAICompatibleConfig]):
    name = "openai_compatible"

    def __init__(
        self,
        config: OpenAICompatibleConfig,
    ):
        super().__init__(config)

        from openai import OpenAI
        from transformers import AutoTokenizer

        self.client = OpenAI(
            api_key=os.getenv(self.config.api_key_variable_name),
            base_url=self.config.base_url,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES
        )
         
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)


    def _generate(self, output: GenerationOutput) -> None:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=output.messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": output.schema, "name": "json_schema"},
                },
                stream=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return

        tokens_str: List[str] = []
        try:
            for i, chunk in enumerate(response):
                if i == 0:
                    first_token_arrival_time = time()

                if len(chunk.choices) == 0 or (chunk.choices[0].finish_reason is not None and chunk.choices[0].finish_reason != "stop"):
                    continue

                chunk_content = chunk.choices[0].delta.content
                if chunk_content == "" or chunk_content is None:
                    continue

                tokens_str.append(chunk_content)
        except Exception as e:
            output.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.API_BAD_RESPONSE, message=str(e)
            )
            return
        
        output.token_usage.output_tokens = chunk.usage.completion_tokens
        output.metadata.first_token_arrival_time = first_token_arrival_time
        output.metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
        output.metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        output.generation = "".join(tokens_str)
        output.generated_tokens = [
            Token(id=self.convert_token_to_id(token), text=token)
            for token in tokens_str
        ]
        return

    def adapt_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        recursively_set_additional_properties_false(schema)
        add_root_type_if_missing(schema)
        schema = set_all_properties_required(schema)
        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def max_context_length(self):
        return self.config.max_context_length


def add_root_type_if_missing(schema: dict):
    if "type" not in schema:
        schema["type"] = "object"


def recursively_set_additional_properties_false(schema: dict):
    if not isinstance(schema, dict):
        return
    if (
        "additionalProperties" not in schema or schema["additionalProperties"]
    ) and schema.get("properties"):
        schema["additionalProperties"] = False
    if "properties" in schema:
        for prop in schema["properties"]:
            recursively_set_additional_properties_false(schema["properties"][prop])
    if "items" in schema:
        recursively_set_additional_properties_false(schema["items"])


def set_all_properties_required(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            set_all_properties_required(value)
        elif isinstance(value, list):
            for item in value:
                set_all_properties_required(item)
    return schema


register_engine(OpenAICompatibleEngine, OpenAICompatibleConfig)