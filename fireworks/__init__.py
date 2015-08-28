#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon Chemistry """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.1"

import logging
from numpy import RankWarning
from warnings import simplefilter

logger = logging.getLogger("fireworks")
logger.setLevel(logging.INFO)

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)
