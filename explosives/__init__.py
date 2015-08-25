#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon Chemistry """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.1"

import logging
from numpy import RankWarning
from warnings import simplefilter

logger = logging.getLogger("explosives")
logger.setLevel(logging.INFO)

#logging.basicConfig(level=logging.INFO, 
#    format="%(asctime)s [%(levelname)s] %(message)s")

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)
