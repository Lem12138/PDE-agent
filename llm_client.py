# llm_client.py — DeepSeek API 统一调用封装，支持按 agent 角色覆盖模型配置
import os
import time
import requests
from dotenv import load_dotenv

# Load .env: workspace first, then project root
_file_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_file_dir, ".."))
for env_path in [os.path.join(_file_dir, ".env"), os.path.join(_root, ".env")]:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions API."""

    def __init__(self):
        self.api_key = os.environ.get("API_KEY", "")
        self.base_url = os.environ.get("API_BASE_URL", "")
        self.default_model = os.environ.get("API_MODEL", "deepseek-v4-pro")
        self.timeout = int(os.environ.get("LLM_TIMEOUT", "600"))

    # ------------------------------------------------------------------
    # Per-agent overrides
    # ------------------------------------------------------------------
    def _agent_config(self, agent_name: str):
        """Return (api_key, base_url, model) for a named agent, falling back to defaults."""
        prefix = agent_name.upper().replace("-", "_")
        key = os.environ.get(f"AGENT_{prefix}_API_KEY") or self.api_key
        url = os.environ.get(f"AGENT_{prefix}_BASE_URL") or self.base_url
        model = os.environ.get(f"AGENT_{prefix}_MODEL") or self.default_model
        return key, url, model

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------
    def call(self, agent_name: str, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request. Returns reply text or "ERROR: ...".
        """
        key, url, model = self._agent_config(agent_name)

        if not key or not url:
            return f"ERROR: API_KEY or API_BASE_URL not set for agent '{agent_name}'"

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
        except requests.Timeout:
            reply = f"ERROR: LLM timeout after {self.timeout}s"
        except Exception as e:
            reply = f"ERROR: {e}"

        elapsed = time.time() - t0
        print(f"[LLM] {agent_name} ({model}) → {elapsed:.1f}s, {len(reply)} chars")
        return reply


# Global singleton
_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def llm_call(agent_name: str, system_prompt: str, user_prompt: str) -> str:
    """Convenience function: one-line LLM call."""
    return get_client().call(agent_name, system_prompt, user_prompt)
