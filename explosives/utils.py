#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utilities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import sys

logger = logging.getLogger("explosives")

def progressbar(iterable, message=None, size=100):
    """
    A progressbar.

    :param iterable:
        Some iterable to show progress for.

    :type iterable:
        iterable

    :param message: [optional]
        A string message to show as the progressbar header.

    :type message:
        str

    :param size: [optional]
        The size of the progressbar. If the size given is zero or negative, then
        no progressbar will be shown.

    :type size:
        int
    """

    count = len(iterable)
    def _update(i):
        if 0 >= size: return
        increment = max(1, int(count / 100))
        if i == 0 or i % increment == 0:
            sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%".format(
                done="=" * int((i + 1) / increment),
                not_done=" " * int((count - i - 1)/increment),
                percent=100. * (i + 1)/count))
            sys.stdout.flush()

    # Initialise
    if size > 0:
        logger.info((message or "").rstrip())
        sys.stdout.flush()

    for i, item in enumerate(iterable):
        yield item
        _update(i + 1)

    if size > 0:
        sys.stdout.write("\r\n")
        sys.stdout.flush()
