#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.kg_utils (preferred) or src.kg_utils."""

try:
    from drugagent.kinase.kg_utils import *  # noqa: F401,F403
except Exception:
    from src.kg_utils import *  # noqa: F401,F403
