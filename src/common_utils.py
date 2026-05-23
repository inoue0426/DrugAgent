#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.common_utils (preferred) or src.common_utils."""

try:
    from drugagent.kinase.common_utils import *  # noqa: F401,F403
except Exception:
    from src.common_utils import *  # noqa: F401,F403
