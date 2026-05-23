#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.rag_utils (preferred) or src.rag_utils."""

try:
    from drugagent.kinase.rag_utils import *  # noqa: F401,F403
except Exception:
    from src.rag_utils import *  # noqa: F401,F403
