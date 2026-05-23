#!/usr/bin/env python
# coding: utf-8

"""Shim module to expose drugagent.kinase.config_utils (preferred) or src.config_utils."""

try:
    from drugagent.kinase.config_utils import *  # noqa: F401,F403
except Exception:
    from src.config_utils import *  # noqa: F401,F403
