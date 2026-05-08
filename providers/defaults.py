"""Re-exports default upstream base URLs from the config provider catalog."""

from config.provider_catalog import (
    NVIDIA_NIM_DEFAULT_BASE,
    OPENAI_COMPATIBLE_DEFAULT_BASE,
)

__all__ = (
    "NVIDIA_NIM_DEFAULT_BASE",
    "OPENAI_COMPATIBLE_DEFAULT_BASE",
)
