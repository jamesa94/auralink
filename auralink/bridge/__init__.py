"""Connectors that bridge an audio encoder sequence into language-model space."""

from auralink.bridge.base import Connector
from auralink.bridge.layers import RMSNorm, SwiGLU
from auralink.bridge.linear import LinearConnector
from auralink.bridge.pooling import PoolingConnector
from auralink.bridge.qformer import QFormerConnector
from auralink.bridge.registry import build_connector, list_connectors, register_connector

__all__ = [
    "Connector",
    "LinearConnector",
    "PoolingConnector",
    "QFormerConnector",
    "RMSNorm",
    "SwiGLU",
    "build_connector",
    "list_connectors",
    "register_connector",
]
