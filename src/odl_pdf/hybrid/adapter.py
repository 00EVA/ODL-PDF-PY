# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Provider abstraction: the rewrite's equivalent of Java's HybridClient +
# HybridClientFactory seam (docs/architecture/08-hybrid-ai-mode.md §14). A
# backend takes a rendered page and returns semantic IObjects; the registry
# maps a name to a constructed adapter.
"""Backend adapter protocol and registry."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from odl_pdf.entities import IObject
from odl_pdf.hybrid.config import HybridConfig
from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)


class BackendError(Exception):
    """Raised when a backend is unavailable or a conversion fails."""


@runtime_checkable
class BackendAdapter(Protocol):
    """An AI backend that converts a rendered page into semantic content.

    Implementations: a real vision backend (Bedrock/Claude) or a deterministic
    mock for tests. The orchestrator only ever depends on this protocol.
    """

    @property
    def name(self) -> str:
        """Stable backend identifier (matches the registry key)."""
        ...

    def check_availability(self) -> None:
        """Fail-fast probe. Raises :class:`BackendError` if unusable."""
        ...

    def convert_page(
        self, page_number: int, image_png: bytes, page_width: float, page_height: float
    ) -> list[IObject]:
        """Convert one rendered page image into semantic ``IObject``s.

        Coordinates in the returned objects are PDF points, BOTTOMLEFT origin,
        on ``page_number``.
        """
        ...


# Registry: name -> factory(config) -> adapter. Mirrors HybridClientFactory.
_FACTORIES: dict[str, Callable[[HybridConfig], BackendAdapter]] = {}


class BackendRegistry:
    """Maps backend names to adapter factories (the plug-in seam)."""

    @staticmethod
    def register(name: str, factory: Callable[[HybridConfig], BackendAdapter]) -> None:
        logger.debug("registering backend %r", name)
        _FACTORIES[name] = factory

    @staticmethod
    def create(name: str, config: HybridConfig) -> BackendAdapter:
        if name not in _FACTORIES:
            raise BackendError(
                f"unknown backend {name!r}; registered: {sorted(_FACTORIES)}"
            )
        logger.info("creating backend %r", name)
        return _FACTORIES[name](config)

    @staticmethod
    def supported() -> list[str]:
        return sorted(_FACTORIES)
