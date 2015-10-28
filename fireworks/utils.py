#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utilities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import sys
import logging
from time import time
from collections import Counter
from itertools import combinations_with_replacement

logger = logging.getLogger("cannon")


def element(species, full_output=False):
    """
    Convert a floating point representation of a species to a string
    representation of the element, and optionally, its ionisation level.
    
    :param species:
        The floating-point representation of the transition.

    :type species:
        float or int

    :param full_output: [optional]
        Return the ionisation stage as well as the element.

    :type full_output:
        bool

    :returns:
        The element name, and optionally the ionisation level.
    """
    
    periodic_table = '''H                                                  He
                        Li Be                               B  C  N  O  F  Ne
                        Na Mg                               Al Si P  S  Cl Ar
                        K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                        Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                        Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                        Fr Ra Lr Rf'''
    
    lanthanoids    = '''La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb'''
    actinoids      = '''Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No'''
    
    periodic_table = periodic_table.replace(' Ba ', ' Ba ' + lanthanoids + ' ')\
        .replace(' Ra ', ' Ra ' + actinoids + ' ').split()
    del actinoids, lanthanoids

    try:
        species = float(species)
    except (TypeError, ValueError):
        raise TypeError("species must be represented by a floating point-type")

    if int(species) > len(periodic_table):
        raise ValueError("""species '{0}' not recognised 
            (molecular representations not supported)""".format(species))
        
    element = periodic_table[int(species) - 1]
    ionisation = int(round(10 * (species % int(species))) + 1)
    
    return (element, ionisation) if full_output else element


def species(element):
    """
    Converts a string representation of an element and its ionisation state
    to a floating species value.

    :param element:
        The element to convert. Some examples are: "Si", "S 2", "Fe I".

    :type element:
        str

    :returns:
        A floating-point representation of the atomic number and the ionisation
        stage.
    """
    
    periodic_table = '''H                                                  He
                        Li Be                               B  C  N  O  F  Ne
                        Na Mg                               Al Si P  S  Cl Ar
                        K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                        Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                        Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                        Fr Ra Lr Rf''' 
    
    lanthanoids    = '''La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb'''
    actinoids      = '''Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No'''
    
    periodic_table = periodic_table.replace(' Ba ', ' Ba ' + lanthanoids + ' ')\
        .replace(' Ra ', ' Ra ' + actinoids + ' ').split()
    del actinoids, lanthanoids
    
    try:
        element = str(element)
    except (TypeError, ValueError):
        raise TypeError("element must be represented by a string-type")
        
    _ = element.split()
    element = _.pop(0).title()
    if len(_):
        # Can only be an integer, or multiples of 'I'
        _ = _[0].upper()
        try:
            ionisation = (_.count("I") if "I" in _ else int(_)) - 1
        except (TypeError, ValueError):
            raise TypeError("unrecognised ionisation level '{0}'".format(_))
    else:
        ionisation = 0

    return 1 + periodic_table.index(element) + ionisation/0.1
    


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

    t_init = time()
    count = len(iterable)
    def _update(i, t=None):
        if 0 >= size: return
        increment = max(1, int(count / 100))
        if i % increment == 0 or i in (0, count):
            sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%{t}".format(
                done="=" * int(i/increment),
                not_done=" " * int((count - i)/increment),
                percent=100. * i/count,
                t="" if t is None else " ({0:.0f}s)".format(t-t_init)))
            sys.stdout.flush()

    # Initialise
    if size > 0:
        logger.info((message or "").rstrip())
        sys.stdout.flush()

    for i, item in enumerate(iterable):
        yield item
        _update(i)

    if size > 0:
        _update(count, time())
        sys.stdout.write("\r\n")
        sys.stdout.flush()


def label_vector(labels, order, cross_term_order=0, mul="*", pow="^"):
    """
    Build a label vector description.

    :param labels:
        The labels to use in describing the label vector.

    :type labels:
        iterable

    :param order:
        The maximum order of the terms (e.g., order 3 implies A^3 is a term).

    :type order:
        int

    :param cross_term_order: [optional]
        The maximum order of the cross-terms (e.g., cross_term_order 2 implies
        A^2*B is a term).
    
    :type cross_term_order:
        int

    :param mul: [optional]
        The operator to use to represent multiplication in the description of 
        the label vector.

    :type mul:
        str

    :param pow: [optional]
        The operator to use to represent exponents in the description of the
        label vector.

    :type pow:
        str

    :returns:
        A human-readable form of the label vector.
    """

    # I make no apologies; it's fun to code this way for short complex functions
    
    items = []
    for o in range(1, 1 + max(order, 1 + cross_term_order)):
        for t in map(Counter, combinations_with_replacement(labels, o)):
            if len(t) == 1 and order >= max(t.values()) \
            or len(t) > 1 and cross_term_order >= max(t.values()):
                c = [pow.join([[l], [l, str(p)]][p > 1]) for l, p in t.items()]
                if c: items.append(mul.join(map(str, c)))
    return " ".join(items)
