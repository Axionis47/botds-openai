"""LLM router: OpenAI for critical decisions, Ollama for non-critical drafting."""

import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from .context import DecisionLog


class LLMRouter:
    """Routes requests to appropriate LLM based on criticality."""
    
    def __init__(self, config: Dict[str, Any], decision_log: DecisionLog):
        self.config = config
        self.decision_log = decision_log
        
        # Initialize OpenAI client (required)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "OpenAI is the sole decision authority - no fallback available."
            )
        
        self.openai_client = OpenAI(api_key=api_key)
        self.openai_model = config.get("openai_model", "gpt-4o-mini")
        
        # Ollama settings (optional)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "llama3.2")
        self.ollama_available = self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def openai_decide(
        self,
        stage: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a critical decision using OpenAI with function calling."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior data scientist making critical decisions. "
                    "Be precise, evidence-based, and explain your reasoning clearly. "
                    "Use the provided tools to gather information before deciding."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        if context:
            messages.insert(1, {
                "role": "system", 
                "content": f"Context: {json.dumps(context, indent=2)}"
            })
        
        try:
            if tools:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1  # Low temperature for consistent decisions
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.1
                )
            
            # Extract decision and rationale
            message = response.choices[0].message
            decision_text = message.content or ""
            
            # Handle tool calls if present
            tool_calls = []
            if message.tool_calls:
                tool_calls = [
                    {
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments),
                        "id": call.id
                    }
                    for call in message.tool_calls
                ]
            
            result = {
                "decision": decision_text,
                "tool_calls": tool_calls,
                "model": self.openai_model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Log the decision
            self.decision_log.record_decision(
                stage=stage,
                decision=decision_text,
                rationale=f"OpenAI decision with {len(tool_calls)} tool calls",
                inputs_refs=[f"prompt:{len(prompt)} chars"],
                auth_model="openai"
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"OpenAI decision failed: {str(e)}")
    
    def ollama_draft(
        self,
        prompt: str,
        max_tokens: int = 500
    ) -> Optional[str]:
        """Generate non-critical draft using Ollama (optional)."""
        if not self.ollama_available:
            return None
        
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return None
                
        except Exception:
            return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "openai_model": self.openai_model,
            "ollama_model": self.ollama_model,
            "ollama_available": self.ollama_available,
            "decisions_logged": len(self.decision_log.get_decisions())
        }
