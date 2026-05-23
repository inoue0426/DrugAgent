#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.ml_utils (preferred) or src.ml_utils."""

try:
    from drugagent.kinase.ml_utils import *  # noqa: F401,F403
except Exception:
    from src.ml_utils import *  # noqa: F401,F403
