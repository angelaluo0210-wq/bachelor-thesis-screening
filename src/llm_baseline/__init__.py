"""Offline-first infrastructure for the future Qwen zero-shot baseline.

Nothing in this package sends a network request unless a caller explicitly
injects a transport and invokes it.  The Phase 1 command-line runner exposes
only Train/Validation inputs; it has no Test option.
"""

from .prompt import PROMPT_VERSION, build_prompt, prompt_fingerprint

__all__ = ("PROMPT_VERSION", "build_prompt", "prompt_fingerprint")
