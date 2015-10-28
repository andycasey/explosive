#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test the general utilities. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import unittest

from fireworks import utils

class TestElementRepresentations(unittest.TestCase):
    """
    Test transformations between element and ionisation stage.
    """

    def setUp(self):

        table =       '''H                                                  He
                         Li Be                               B  C  N  O  F  Ne
                         Na Mg                               Al Si P  S  Cl Ar
                         K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                         Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                         Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                         Fr Ra Lr Rf'''

        lanthanoids = '''La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb'''
        actinoids   = '''Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No'''

        self.periodic_table = table.replace(' Ba ', ' Ba ' + lanthanoids + ' ')\
            .replace(' Ra ', ' Ra ' + actinoids + ' ').split()
        

    def test_get_species(self):
        for i, element in enumerate(self.periodic_table, start=1):
            self.assertEqual(i, utils.species(element))


    def test_get_species_lowercase(self):
        for i, element in enumerate(self.periodic_table, start=1):
            self.assertEqual(i, utils.species(element.lower()))


    def test_get_species_uppercase(self):
        for i, element in enumerate(self.periodic_table, start=1):
            self.assertEqual(i, utils.species(element.upper()))
            

    def test_get_species_roman(self):
        for i, element in enumerate(self.periodic_table, start=1):
            for j in range(3):
                self.assertEqual(i + 0.1 * j,
                    utils.species("{0} {1}".format(element, "I" * (j + 1))))


    def test_get_species_roman_lowercase(self):
        for i, element in enumerate(self.periodic_table, start=1):
            for j in range(3):
                self.assertEqual(i + 0.1 * j,
                    utils.species("{0} {1}".format(element.lower(), "i" * (j + 1))))


    def test_get_species_roman_uppercase(self):
        for i, element in enumerate(self.periodic_table, start=1):
            for j in range(3):
                self.assertEqual(i + 0.1 * j,
                    utils.species("{0} {1}".format(element.upper(), "I" * (j + 1))))


    def test_get_element(self):
        for i, element in enumerate(self.periodic_table, start=1):
            self.assertEqual(element, utils.element(i))  


    def test_get_ionisation(self):
        for i, element in enumerate(self.periodic_table, start=1):
            for j in range(1, 3):
                self.assertEqual((element, j), 
                    utils.element(i + (j - 1) * 0.1, True))

