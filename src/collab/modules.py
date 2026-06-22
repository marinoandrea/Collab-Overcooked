from __future__ import annotations

import difflib
import os
import sys
import time
from typing import Any, Union

import tiktoken
from openai import OpenAI
from rich import print as rprint

from .utils import retry_with_exponential_backoff

cwd = os.getcwd()
gpt4_key_file = os.path.join(cwd, "openai_key.txt")


def _load_api_key() -> str:
    """API key from ``$OPENAI_API_KEY``, falling back to openai_key.txt if present."""
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        with open(gpt4_key_file, "r") as f:
            return f.read().split("\n")[0]
    except FileNotFoundError:
        return ""


def _client() -> OpenAI:
    """One OpenAI-compatible client, configured purely from the environment.

    ``$OPENAI_BASE_URL`` selects the endpoint (OpenAI, llama.cpp, vLLM, Ollama,
    LiteLLM, …); when unset the official OpenAI endpoint is used.
    """
    return OpenAI(
        api_key=_load_api_key() or "sk-no-key",
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


openai_key = _load_api_key()


def _count_tokens(text: str, model: str) -> int:
    """Token count via tiktoken; cl100k_base fallback for non-OpenAI model ids."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# global statistics
statistics_dict = {
    "total_timestamp": [],
    "total_order_finished": [],
    "total_score": 0,
    "total_action_list": [[], []],
    "content": [],
}

# turn statistics
turn_statistics_dict = {
    "timestamp": 0,
    "order_list": [],
    "actions": [],
    "map": "",
    "statistical_data": {
        "score": 0,
        "communication": [
            {"call": 0, "turn": [], "token": []},
            {"call": 0, "turn": [], "token": []},
        ],
        "error": [
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
        ],
        "error_correction": [
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
        ],
    },
    "content": {
        "observation": [[], []],
        "reflection": [[], []],
        "content": [[], []],
        "action_list": [[], []],
        "original_log": "",
    },
}

# LLM models (kept as module globals: imported by main.py)
tokenizer, model = None, None

# Context-window budget for restrict_dialogue(); override via env per deployment.
MODEL_CONTEXT_LIMIT = int(os.environ.get("MODEL_CONTEXT_LIMIT", "8192"))
sys.path.append(os.getcwd())


class Module(object):
    """
    This module is responsible for communicating with LLMs.
    """

    def __init__(
        self,
        role_messages,
        model="gpt-3.5-turbo-0301",
        model_dirname="~/",
        local_server_api="http://localhost:8000/v1",
        retrival_method="recent_k",
        K=3,
    ):

        self.model = model
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api
        self.retrival_method = retrival_method
        self.K = K

        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        # a dynamic changed dialog_history used for generating  different input for each failure
        self.dialog_history_list = []
        # save the dialog_history of meetting first failture
        self.dialog_history_list_storage = []
        self.current_user_message = None
        self.cache_list = None
        self.experience = []
        self.embedding = None
        self.current_timestep = None

    def load_embedding(self) -> None:
        """Example-retrieval embeddings are disabled in the OpenAI-only fork."""
        self.embedding = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)

    def get_cache(self) -> list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K :]
            else:
                return []
        else:
            return None

    def query_messages(self, rethink) -> list:
        sytem_message = [
            {
                "role": "system",
                "content": "You are an intelligent agent planner, you need to generate output and plan in the specified format according to the game rules and environmental status.",
            }
        ]
        query = sytem_message + [
            {
                "role": "user",
                "content": self.instruction_head_list[0]["content"]
                + "<input>\n"
                + self.current_user_message["content"],
            }
        ]
        return query

    @retry_with_exponential_backoff
    def query(
        self,
        key,
        proxy,
        stop=None,
        temperature=0.7,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
        # One OpenAI-compatible chat completion; endpoint/key/model come from env.
        messages = self.query_messages(rethink)
        self.cache_list = self.get_cache()

        if trace is False and not rethink:
            messages[-1]["content"] += (
                " Based on the failure explanation and scene description, "
                "analyze and plan again."
            )

        max_tokens_env = os.environ.get("MODEL_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else None

        response: Any = None
        for _ in range(3):
            try:
                response = _client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:  # noqa: BLE001 - log API/network errors and retry
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(1)
        else:
            rprint("[red][ERROR][/red]: query failed repeatedly!")
            self.current_user_message["content"] = self.current_user_message["content"][:-40]
            return "", 0

        rs = self.parse_response(response)
        return rs, _count_tokens(rs, self.model)

    def parse_response(self, response: Any) -> str:
        """Extract assistant text from a chat completion (single client path)."""
        return response.choices[0].message.content or ""

    def restrict_dialogue(self) -> None:
        """Trim oldest dialogue turns when the prompt exceeds the context budget."""
        limit = MODEL_CONTEXT_LIMIT
        print(f"Current token: {self.prompt_token_length}")
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f"Update token: {self.prompt_token_length}")

    def reset(self):
        self.dialog_history_list = []

    def get_top_k_similar_example(self, key: str, k: int = 4) -> str:
        """Example retrieval is disabled in the OpenAI-only fork (no embeddings)."""
        return ""


def if_two_sentence_similar_meaning(
    key: str, proxy: Any, sentence1: str, sentence2: str
) -> bool:
    """Local, embedding-free near-duplicate check (works against any endpoint)."""
    a = sentence1 or " "
    b = sentence2 or " "
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.9
