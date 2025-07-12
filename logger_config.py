#!/usr/bin/env python3
"""
logger_config.py
-------------------
loggerの共通フォーマットを提供
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
