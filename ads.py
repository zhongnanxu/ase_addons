"""Module that stores adsorbates"""
from __future__ import division

from ase import Atom, Atoms
from ase.constraints import FixAtoms, FixScaled
from myvasp import condense
from numpy import pi
import numpy as np

def get_O():
    height = 1.7
    atoms = Atoms([Atom('O', (0, 0, 0), magmom=0)])
    return atoms, height

def get_OH(rotate=0):
    height = 1.8
    atoms = Atoms([Atom('O', (-0.15, 0, 0), magmom=0),
                  Atom('H', (0.655, 0, 0.564), magmom=0)])
    rad = rotate * pi/180
    atoms.rotate('z', rad)
    return atoms, height

def get_OOH(rotate=0):
    height = 1.8
    atoms = Atoms([Atom('O', (-0.433, 0 , 0), magmom=0),
                  Atom('O', (0.648, 0, 0.964), magmom=0),
                  Atom('H', (0.112, 0, 1.79), magmom=0)])
    rad = rotate * pi/180
    atoms.rotate('z', rad)
    return atoms, height

        
        
