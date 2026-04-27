import os
import time
import json
from typing import Type, TypeVar, Tuple
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Cấu hình Groq API
api_key = os.environ.get("GROQ_API_KEY")
client = None
if api_key:
    client = Groq(api_key=api_key)

MODEL_NAME = "llama-3.1-8b-instant"

T = TypeVar("T", bound=BaseModel)

def call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
    """
    Gọi Groq API, trả về (nội dung, số token ước tính, latency_ms)
    """
    start_time = time.time()
    
    if not client:
        return "", 0, 0
        
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content or ""
        token_count = response.usage.total_tokens if response.usage else 0
            
    except Exception as e:
        print(f"LLM Error: {e}")
        content = ""
        token_count = 0
        
    latency = int((time.time() - start_time) * 1000)
    
    if token_count == 0:
        token_count = len(system_prompt.split()) + len(user_prompt.split()) + len(content.split())
        
    return content.strip(), token_count, latency

def call_llm_json(system_prompt: str, user_prompt: str, schema: Type[T]) -> Tuple[T, int, int]:
    """
    Gọi Groq API và ép kiểu về Pydantic schema
    """
    start_time = time.time()
    
    if not client:
        return _fallback_schema(schema, "No API Key"), 0, 0
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content or "{}"
        result = schema.model_validate_json(text)
        token_count = response.usage.total_tokens if response.usage else 0
            
    except Exception as e:
        print(f"LLM JSON Error: {e}")
        result = _fallback_schema(schema, f"LLM Error: {e}")
        token_count = 0
        
    latency = int((time.time() - start_time) * 1000)
    
    if token_count == 0:
        token_count = len(system_prompt.split()) + len(user_prompt.split()) + 50
        
    return result, token_count, latency

def _fallback_schema(schema: Type[T], error_msg: str) -> T:
    if schema.__name__ == "JudgeResult":
        return schema(score=0, reason=error_msg, missing_evidence=[], spurious_claims=[])
    elif schema.__name__ == "ReflectionEntry":
        return schema(attempt_id=0, failure_reason=error_msg, lesson="API failed", next_strategy="Retry")
    return schema.construct()
