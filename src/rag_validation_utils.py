#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.rag_validation_utils (preferred) or src.rag_validation_utils."""

try:
    from drugagent.kinase.rag_validation_utils import *  # noqa: F401,F403
except Exception:
    from src.rag_validation_utils import *  # noqa: F401,F403
