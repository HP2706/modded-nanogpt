# tools for reading, writing, applying code edits..
# This file contains MCP implementations that copy code from anthropic_tools with proper attribution
# Using only MCP types and standard Python types, no external tool framework dependencies
import asyncio
import logging
from modal import Volume
import modal

agent_volume = Volume.from_name("mle-sandbox", create_if_missing=True)
fineweb10B_volume = Volume.from_name("fineweb10B", create_if_missing=True)

logger = logging.getLogger(__name__)

# Utility functions copied from anthropic_tools/run.py with attribution
TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


class LazySandBox:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        
        self._sandbox = modal.Sandbox.create(
            **self._kwargs,
        )
    
    def exec(self, *args, **kwargs):
        try:
            return self._sandbox.exec(*args, **kwargs)
        except Exception as e:
            print("error, timeout, restarting sandbox")
            self._sandbox = modal.Sandbox.create(
                **self._kwargs,
            )
            return self._sandbox.exec(*args, **kwargs)

    def reload_volumes(self):
        try:
            self._sandbox.reload_volumes()
        except Exception as e:
            self._sandbox = modal.Sandbox.create(
                **self._kwargs,
            )
            