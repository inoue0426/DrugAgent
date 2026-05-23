#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.token_usage (preferred) or src.token_usage."""

try:
    from drugagent.kinase.token_usage import *  # noqa: F401,F403
except Exception:
    from src.token_usage import *  # noqa: F401,F403
