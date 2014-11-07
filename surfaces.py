"""Module that stores surfaces"""
from __future__ import division

from ase import Atom, Atoms
from ase.constraints import FixAtoms, FixScaled
import numpy as np

def rocksalt100(symbol, a, area=(2,1), layers=4, vacuum=0, afm=True, mag=2.,
                fixlayers=0, symmetry=False, fixrelax=False):
    """Returns an atom object with the rocksalt(100) surface

    Parameters
    ----------
    symbol: list
        The atoms used in the cell
    a: flt
        The lattice constant of the rocksalt structure
    area: tuple
        The amount supercells in the (x,y,z) directions
    layers: int
        The amount of layers in the supercell. Each layer consists
        of 2 A and 2 B atoms
    vacuum: flt
        The amount of vacuum you want between slabs
    afm: bool
        Decide whether the slab is ferro-magnetic
        or anti-ferromagnetic
    mag: flt
        The initial magnetic moment of the cation
    fixlayers: int
        Amount of layers to fix. Note if you're using a symmetric
        slab, then you MUST specify an odd number
    symmetry: bool
        Specify whether the slab should relaxed symmetrically or not
        In order for the symmetric slab to be correct, you MUST specify
        an odd number of slabs for both fixlayers and layers.
    """
    fixlayers -= 1
    if afm == False:
        # One layer is a 2x2x2 surface
        slab = Atoms([Atom(symbol[0],(0.,  0.,  0.)),  # Layer one
                      Atom(symbol[1],(a/2, 0.,  0.)),
                      Atom(symbol[1],(0.,  a/2, 0.)),
                      Atom(symbol[0],(a/2, a/2, 0.)),
                      Atom(symbol[1],(0.,  0.,  a/2)), # Layer two
                      Atom(symbol[0],(a/2, 0.,  a/2)),
                      Atom(symbol[0],(0.,  a/2, a/2)),
                      Atom(symbol[1],(a/2, a/2, a/2))],
                     cell=[(a,  0., 0.),
                           (0., a,  0.),
                           (0., 0., a)])
    elif afm == True:
        # Create the base layer that is a 2x2x1 surface with a rotated x-y plane
        a1 = np.array([a/2, -a/2, 0])
        a2 = np.array([a, a, 0])
        a3 = np.array([0, 0, layers*a/2])
        up = mag
        down = -mag
        slab = Atoms(cell=(a1,a2,a3))
        # Differentiate between symmetric and asymmetric relaxation
        if symmetry == True and fixlayers > 0:
            mid_layer = layers // 2 - 0.5
            fixlayers = [mid_layer - fixlayers//2 - 1,
                         mid_layer + fixlayers//2 + 1]
        elif symmetry == False and fixlayers > 0:
            fixlayers = [-1, -1 + fixlayers]
        else:
            fixlayers = [-1, -1]
        # Now add layers
        for i in range(layers):
            if i % 4 == 0:
                if i >= fixlayers[0] and i <= fixlayers[1]:
                    tag = 1
                else:
                    tag = 0
                layer = Atoms([Atom(symbol[0], np.array([a/2, 0, i * a/2]),
                                    magmom=down, tag=tag),
                               Atom(symbol[0], np.array([a, a/2, i * a/2]),
                                    magmom=up, tag=tag),
                               Atom(symbol[1], np.array([0, 0, i * a/2]),
                                    magmom=0., tag=tag),
                               Atom(symbol[1], np.array([a/2, a/2, i * a/2]),
                                    magmom=0., tag=tag)])
            elif i % 4 == 1:
                if i >= fixlayers[0] and i <= fixlayers[1]:
                    tag = 1
                else:
                    tag = 0
                layer = Atoms([Atom(symbol[0], np.array([0, 0, i * a/2]),
                                    magmom=down, tag=tag),
                               Atom(symbol[0], np.array([a/2, a/2, i * a/2]),
                                    magmom=up, tag=tag),
                               Atom(symbol[1], np.array([a/2, 0, i * a/2]),
                                    magmom=0., tag=tag),
                               Atom(symbol[1], np.array([a, a/2, i * a/2]),
                                    magmom=0., tag=tag)])
            elif i % 4 == 2:
                if i >= fixlayers[0] and i <= fixlayers[1]:
                    tag = 1
                else:
                    tag = 0
                layer = Atoms([Atom(symbol[0], np.array([a/2, 0, i * a/2]),
                                    magmom=up, tag=tag),
                               Atom(symbol[0], np.array([a, a/2, i * a/2]),
                                    magmom=down, tag=tag),
                               Atom(symbol[1], np.array([0, 0, i * a/2]),
                                    magmom=0., tag=tag),
                               Atom(symbol[1], np.array([a/2, a/2, i * a/2]),
                                    magmom=0., tag=tag)])
            elif i % 4 == 3:
                if i >= fixlayers[0] and i <= fixlayers[1]:
                    tag = 1
                else:
                    tag = 0
                layer = Atoms([Atom(symbol[0], np.array([0, 0, i * a/2]),
                                    magmom=up, tag=tag),
                               Atom(symbol[0], np.array([a/2, a/2, i * a/2]),
                                    magmom=down, tag=tag),
                               Atom(symbol[1], np.array([a/2, 0, i * a/2]),
                                    magmom=0., tag=tag),
                               Atom(symbol[1], np.array([a, a/2, i * a/2]),
                                    magmom=0., tag=tag)])
            slab.extend(layer)

    # Multiply it in the x and y directions by the dimensions and add vacuum
    slab = slab*(area[0], area[1], 1)
    if vacuum != 0:
        slab.center(vacuum=vacuum, axis=2)
    slab = condense(slab)
    if fixlayers > 0:
        if fixrelax == True:
            c = []
            for atom in slab:
                if atom.tag == 0:
                    c.append(FixScaled(cell=slab.cell, a=atom.index,
                                       mask=(1,1,0)))
                else:
                    c.append(FixScaled(cell=slab.cell, a=atom.index,
                                       mask=(1,1,1)))
            slab.set_constraint(c)
        else:
            c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
            slab.set_constraint(c)
    return slab

def rocksalt111(symbol, a=4.20, b=0, area=(1, 1), layers=5, vacuum=0, mag=2,
                fixlayers=0, base=0, cell=None, afm=True):
    """Returns an atoms object with rocksalt(111) surface
    
    
    Parameters
    ----------
    symbol: list
        The atoms used in the cell
     
    """
    if cell == None:
        a1 = np.array([2**(0.5)/2*(b-a), 6**(0.5)/6*(b-a), 3**(0.5)/3*(2*a+b)])
        a2 = np.array([2**(0.5)/2*(a-b), 6**(0.5)/6*(b-a), 3**(0.5)/3*(2*a+b)])
        a3 = np.array([0               , 6**(0.5)/3*(a-b), 3**(0.5)/3*(2*a+b)])
        h1 = a2 - a1
        h2 = a3 - a1
        h3 = (a1 + a2 + a3)
    else:
        h1 = cell[0]
        h2 = cell[1]
        h3 = cell[2]
    up = mag
    if afm == True:
        down = -mag
    else:
        down = mag
    slab = Atoms(cell=(h1, h2, h3 / 6 * layers))
    if fixlayers > 0:
        mid_layer = layers // 2 + base
        fixlayers = [mid_layer - fixlayers // 2, mid_layer + fixlayers // 2]
    else:
        fixlayers = [-1, -1]
    # Now add the layers
    for i in range(base, base + layers):
        j = i // 12
        if i >= fixlayers[0] and i <= fixlayers[1]:
            tag = 1
        else:
            tag = 0            
        if i % 12 == 0:
            layer = Atoms([Atom(symbol[0], h3*2*j, magmom=up, tag=tag)])
        elif i % 12 == 1:
            layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(1/6 + 2*j), magmom=0, tag=tag)])
        elif i % 12 == 2:
            layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(1/3 + 2*j), magmom=down, tag=tag)])
        elif i % 12 == 3:
            layer = Atoms([Atom(symbol[1], h3*(1/2 + 2*j), magmom=0, tag=tag)])
        elif i % 12 == 4:
            layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(2/3 + 2*j), magmom=up, tag=tag)])
        elif i % 12 == 5:
            layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(5/6 + 2*j), magmom=0, tag=tag)])
        elif i % 12 == 6:
            layer = Atoms([Atom(symbol[0], h3*(2*j + 1), magmom=down, tag=tag)])
        elif i % 12 == 7:
            layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(7/6 + 2*j), magmom=0, tag=tag)])
        elif i % 12 == 8:
            layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(4/3 + 2*j), magmom=up, tag=tag)])
        elif i % 12 == 9:
            layer = Atoms([Atom(symbol[1], h3*(3/2 + 2*j), magmom=0, tag=tag)])
        elif i % 12 == 10:
            layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(5/3 + 2*j), magmom=down, tag=tag)])
        elif i % 12 == 11:
            layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(11/6 + 2*j), magmom=0, tag=tag)])
        slab.extend(layer)

    slab = slab * (area[0], area[1], 1)
    if vacuum != 0:
        slab.center(vacuum=vacuum, axis=2)
    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
        slab.set_constraint(c)
    return slab

def spinel111(symbol, a, x, area=(1,1), layers=18, vacuum=0,
              mag=(5,5,0), fixlayers=0, base=3, symmetry=False):
   """Returns an atoms object with the spinel(111) surface

   Parameters
   ----------
   symbol: tuple of strings
       The atoms used in the cell. (A,B,C) corresponds to
       A in the tetrahedrally bonded site
       B in the octahedrally bonded site
       C in the cation site (FCC lattice)
   a: flt
       The lattice constant of the spinel structure
   x: flt
       oxygen parameter of the spinel structure
   area: tuple
       The amount of supercells in the (x,y) directions
   layers: int
       This is the number of layers in the supercell. For the
       spinel(111) slab, a stoichiometric slab consists of
       multiples of 6 layers. Therefore, slabs with 6, 12, 18, etc
       layers have complete formula units. Furthermore, a complete,
       repeating bulk structure requires 18 layers.
   vacuum: flt
       The amount of vacuum you want between slabs
   mag: tuple
       The initial magnetic moments of the A, B, and O ions
   fixlayers: int
       This determines the number of layers to fix for a relaxation.
       For a symmetric cell, this number must be odd to keep the symmetry
   base: int
       This determines the base layer. The layers go in the order of...
       oct1, O1, tet1, oct2, tet2, oct2
   symmetry: bool
       This specifies whether we want a symmetric or asymmetric relaxation
       Note: for a symmetric, cell, only certain combinations of 'base' and
       'layers' works. For a oct2 terminated cell, can only use layers=19 or
       layers=31, base=3. To get different terminations, use this general rule.
       To get a termination less than oct2 (tet1, O1, etc), subtract/add base to
       get to what layer you want and add/subtract double this to the layers

   """
   x -= 0.125
   Amag, Bmag, Omag = -mag[0], mag[1], mag[2]
   a1 = np.array([-0.25*2**0.5*a, -1./12*6**0.5*a, 1./3*3**0.5*a])
   a2 = np.array([0.25*2**0.5*a, -1./12*6**0.5*a, 1./3*3**0.5*a])
   a3 = np.array([0, 1./6*6**0.5*a, 1./3*3**0.5*a])
   h1 = a2 - a1
   h2 = a3 - a1
   h3 = a1 + a2 + a3
   l1 = a1
   l2 = a1 + a2
   slab = Atoms(cell=(h1, h2, layers*h3/18))
   # Differentiate between symmetric and asymmetric slabs
   if symmetry == False and fixlayers > 0:
       fixlayers = [base, base + fixlayers - 1]
   elif symmetry == True and fixlayers > 0:
       mid_layer = layers // 2 + base
       fixlayers = [mid_layer - fixlayers // 2, mid_layer + fixlayers // 2]
   else:
       fixlayers = [-1, -1]
   # Now add the layers
   for i in range(base, base + layers):
       j =  i // 18
       if i >= fixlayers[0] and i<= fixlayers[1]:
           tag = 1
       else:
           tag = 0
       if i % 18 == 0:
           layer = Atoms([Atom(symbol[1], h1/2 + h2/2 + h3*j,
                               magmom=Bmag, tag=tag),
                          Atom(symbol[1], h2/2 + h3*j, magmom=Bmag, tag=tag),
                          Atom(symbol[1], h1/2 + h3*j, magmom=Bmag, tag=tag)])
       elif i % 18 == 1:
           layer = Atoms([Atom(symbol[2], h1*(4*x/3 - 1/6) + h2*(4*x/3 - 1/6) + h3*(-x/3 + 1/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(-8*x/3 + 4/3) + h2*(4*x/3 - 1/6) + h3*(-x/3 + 1/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(4*x/3 - 1/6) + h2*(-8*x/3 + 4/3) + h3*(-x/3 + 1/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], 2*h1/3 + 2*h2/3 + h3*(-x + 1/3 + j),
                               magmom=Omag, tag=tag)])
       elif i % 18 == 2:
           layer = Atoms([Atom(symbol[0], h3*(1/8 + j),
                               magmom=Amag, tag=tag)])
       elif i % 18 == 3:
           layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(1/6 + j),
                               magmom=Bmag, tag=tag)])
       elif i % 18 == 4:
           layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(5/24 + j),
                               magmom=Amag, tag=tag)])
       elif i % 18 == 5:
           layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 5/6) + h2*(-4*x/3 + 5/6) + h3*(x/3 + 1/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(8*x/3 - 2/3) + h2*(-4*x/3 + 5/6) + h3*(x/3 + 1/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(-4*x/3 + 5/6) + h2*(8*x/3 - 2/3) + h3*(x/3 + 1/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h3*(x + j),
                               magmom=Omag, tag=tag)])
       elif i % 18 == 6:
           layer = Atoms([Atom(symbol[1], h1/6 + h2/6 + h3*(1/3 + j),
                               magmom=Bmag, tag=tag),
                          Atom(symbol[1], 2*h1/3 + h2/6 + h3*(1/3 + j),
                               magmom=Bmag, tag=tag),
                          Atom(symbol[1], h1/6 + 2*h2/3 + h3*(1/3 + j),
                               magmom=Bmag, tag=tag)])
       elif i % 18 == 7:
           layer = Atoms([Atom(symbol[2], h1*(4*x/3 + 1/2) + h2*(4*x/3 + 1/2) + h3*(-x/3 + 1/2 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(-8*x/3 + 1) + h2*(4*x/3 + 1/2) + h3*(-x/3 + 1/2 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(4*x/3 + 1/2) + h2*(-8*x/3 + 1) + h3*(-x/3 + 1/2 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1/3 + h2/3 + h3*(-x + 2/3 + j),
                               magmom=Omag, tag=tag)])
       elif i % 18 == 8:
           layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(11/24 + j),
                               magmom=Amag, tag=tag)])
       elif i % 18 == 9:
           layer = Atoms([Atom(symbol[1], h3*(1/2 + j),
                              magmom=Bmag, tag=tag)])
       elif i % 18 == 10:
           layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(13/24 + j),
                               magmom=Amag, tag=tag)])
       elif i % 18 == 11:
           layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 1/2) + h2*(-4*x/3 + 1/2) + h3*(x/3 + 1/2 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], 8*h1*x/3 + h2*(-4*x/3 + 1/2) + h3*(x/3 + 1/2 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(-4*x/3 + 1/2) + 8*h2*x/3 + h3*(x/3 + 1/2 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], 2*h1/3 + 2*h2/3 + h3*(x + 1/3 + j),
                               magmom=Omag, tag=tag)])
       elif i % 18 == 12:
           layer = Atoms([Atom(symbol[1], 5*h1/6 + 5*h2/6 + h3*(2/3 + j),
                               magmom=Bmag, tag=tag),
                          Atom(symbol[1], h1/3 + 5*h2/6 + h3*(2/3 + j),
                               magmom=Bmag, tag=tag),
                          Atom(symbol[1], 5*h1/6 + h2/3 + h3*(2/3 + j),
                               magmom=Bmag, tag=tag)])
       elif i % 18 == 13:
           layer = Atoms([Atom(symbol[2], h1*(4*x/3 + 1/6) + h2*(4*x/3 + 1/6) + h3*(-x/3 + 5/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(-8*x/3 + 2/3) + h2*(4*x/3 + 1/6) + h3*(-x/3 + 5/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(4*x/3 + 1/6) + h2*(-8*x/3 + 2/3) + h3*(-x/3 + 5/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h3*(-x + 1 + j),
                               magmom=Omag, tag=tag)])
       elif i % 18 == 14:
           layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(19/24 + j),
                               magmom=Amag, tag=tag)])
       elif i % 18 == 15:
           layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(5/6 + j),
                               magmom=Bmag, tag=tag)])
       elif i % 18 == 16:
           layer = Atoms([Atom(symbol[0], h3*(7/8 + j),
                               magmom=Amag, tag=tag)])
       elif i % 18 == 17:
           layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 7/6) + h2*(-4*x/3 + 7/6) + h3*(x/3 + 5/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(8*x/3 - 1/3) + h2*(-4*x/3 + 7/6) + h3*(x/3 + 5/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1*(-4*x/3 + 7/6) + h2*(8*x/3 - 1/3) + h3*(x/3 + 5/6 + j), 
                               magmom=Omag, tag=tag),
                          Atom(symbol[2], h1/3 + h2/3 + h3*(x + 2/3 + j),
                               magmom=Omag, tag=tag)])
       slab.extend(layer)

   slab = slab*(area[0], area[1], 1)
   if vacuum != 0:
       slab.center(vacuum=vacuum, axis=2)
   slab = condense(slab)
   if fixlayers > 0:
       c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
       slab.set_constraint(c)
   return slab

def inverse_spinel111(symbol, a, x, area=(1,1), layers=18, vacuum=0,
                      mag=(2,5,0), fixlayers=0, base=3, setting=0,
                      symmetry=False):
   """Returns an atoms object with the spinel(111) surface

   Parameters
   ----------
   symbol: tuple of strings
       The atoms used in the cell. (A,B,C) corresponds to
       A in half of the octahedrally bonded sites
       B in all the tetrahedally and half the octahedrally bonded sites
       C in the cation site (FCC lattice)
   a: flt
       The lattice constant of the spinel structure
   x: flt
       oxygen parameter of the spinel structure
   area: tuple
       The amount of supercells in the (x,y) directions
   layers: int
       This is the number of layers in the supercell. For the
       spinel(111) slab, a stoichiometric slab consists of
       multiples of 6 layers. Therefore, slabs with 6, 12, 18, etc
       layers have complete formula units. Furthermore, a complete,
       repeating bulk structure requires 18 layers.
   vacuum: flt
       The amount of vacuum you want between slabs
   mag: tuple
       The initial magnetic moments of the A, B, and O ions
   fixlayers: int
       This determines the number of layers to fix for a relaxation.
       For a symmetric cell, this number must be odd to keep the symmetry
   base: int
       This determines the base layer. The layers go in the order of...
       oct1, O1, tet1, oct2, tet2, oct2
   symmetry: bool
       This specifies whether we want a symmetric or asymmetric relaxation
       Note: for a symmetric, cell, only certain combinations of 'base' and
       'layers' works. For a oct2 terminated cell, can only use layers=19 or
       layers=31. To get different terminations, use this general rule.
       To get a termination less than oct2 (tet1, O1, etc), subtract/add base to
       get to what layer you want and add/subtract this to the layers

   """
   fixlayers -= 1
   x -= 0.125
   Amag, Bmag, Omag = mag[0], mag[1], mag[2]

   a1 = np.array([-0.25*2**0.5*a, -1./12*6**0.5*a, 1./3*3**0.5*a])
   a2 = np.array([0.25*2**0.5*a, -1./12*6**0.5*a, 1./3*3**0.5*a])
   a3 = np.array([0, 1./6*6**0.5*a, 1./3*3**0.5*a])
   h1 = a2 - a1
   h2 = a3 - a1
   h3 = a1 + a2 + a3
   l1 = a1
   l2 = a1 + a2
   slab = Atoms(cell=(h1, h2, layers*h3/18))
   if symmetry == False and fixlayers > 0:
       fixlayers = [base, base + fixlayers - 1]
   elif symmetry == True and fixlayers > 0:
       mid_layer = layers// 2 + base
       fixlayers = [mid_layer - fixlayers // 2, mid_layer + fixlayers // 2]
   else:
       fixlayers = [-1, -1]
   # Now add the layers
   if setting == 0:
       for i in range(base, base + layers):
           j =  i // 18
           if i % 18 == 0:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/2 + h2/2 + h3*j,
                                   magmom=Bmag, tag=tag),
                              Atom(symbol[0], h2/2 + h3*j, magmom=Amag, tag=tag),
                              Atom(symbol[0], h1/2 + h3*j, magmom=Amag, tag=tag)])
           elif i % 18 == 1:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(4*x/3 - 1/6) + h2*(4*x/3 - 1/6) + h3*(-x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-8*x/3 + 4/3) + h2*(4*x/3 - 1/6) + h3*(-x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(4*x/3 - 1/6) + h2*(-8*x/3 + 4/3) + h3*(-x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], 2*h1/3 + 2*h2/3 + h3*(-x + 1/3 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 2:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h3*(1/8 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 3:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(1/6 + j),
                                   magmom=Bmag, tag=tag)])
           elif i % 18 == 4:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(5/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 5:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 5/6) + h2*(-4*x/3 + 5/6) + h3*(x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(8*x/3 - 2/3) + h2*(-4*x/3 + 5/6) + h3*(x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-4*x/3 + 5/6) + h2*(8*x/3 - 2/3) + h3*(x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h3*(x + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 6:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/6 + h2/6 + h3*(1/3 + j),
                                   magmom=Bmag, tag=tag),
                              Atom(symbol[0], 2*h1/3 + h2/6 + h3*(1/3 + j),
                                   magmom=Amag, tag=tag),
                              Atom(symbol[0], h1/6 + 2*h2/3 + h3*(1/3 + j),
                                   magmom=Amag, tag=tag)])
           elif i % 18 == 7:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(4*x/3 + 1/2) + h2*(4*x/3 + 1/2) + h3*(-x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-8*x/3 + 1) + h2*(4*x/3 + 1/2) + h3*(-x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(4*x/3 + 1/2) + h2*(-8*x/3 + 1) + h3*(-x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1/3 + h2/3 + h3*(-x + 2/3 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 8:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(11/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 9:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h3*(1/2 + j),
                                  magmom=Bmag, tag=tag)])
           elif i % 18 == 10:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(13/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 11:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 1/2) + h2*(-4*x/3 + 1/2) + h3*(x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], 8*h1*x/3 + h2*(-4*x/3 + 1/2) + h3*(x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-4*x/3 + 1/2) + 8*h2*x/3 + h3*(x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], 2*h1/3 + 2*h2/3 + h3*(x + 1/3 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 12:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], 5*h1/6 + 5*h2/6 + h3*(2/3 + j),
                                   magmom=Bmag, tag=tag),
                              Atom(symbol[0], h1/3 + 5*h2/6 + h3*(2/3 + j),
                                   magmom=Amag, tag=tag),
                              Atom(symbol[0], 5*h1/6 + h2/3 + h3*(2/3 + j),
                                   magmom=Amag, tag=tag)])
           elif i % 18 == 13:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(4*x/3 + 1/6) + h2*(4*x/3 + 1/6) + h3*(-x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-8*x/3 + 2/3) + h2*(4*x/3 + 1/6) + h3*(-x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(4*x/3 + 1/6) + h2*(-8*x/3 + 2/3) + h3*(-x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h3*(-x + 1 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 14:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(19/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 15:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(5/6 + j),
                                   magmom=Bmag, tag=tag)])
           elif i % 18 == 16:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h3*(7/8 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 17:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 7/6) + h2*(-4*x/3 + 7/6) + h3*(x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(8*x/3 - 1/3) + h2*(-4*x/3 + 7/6) + h3*(x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-4*x/3 + 7/6) + h2*(8*x/3 - 1/3) + h3*(x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1/3 + h2/3 + h3*(x + 2/3 + j),
                                   magmom=Omag, tag=tag)])
           slab.extend(layer)
   elif setting == 1:
       for i in range(base, base + layers):
           j =  i // 18
           if i % 18 == 0:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[0], h1/2 + h2/2 + h3*j,
                                   magmom=Amag, tag=tag),
                              Atom(symbol[1], h2/2 + h3*j, magmom=Bmag, tag=tag),
                              Atom(symbol[1], h1/2 + h3*j, magmom=Bmag, tag=tag)])
           elif i % 18 == 1:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(4*x/3 - 1/6) + h2*(4*x/3 - 1/6) + h3*(-x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-8*x/3 + 4/3) + h2*(4*x/3 - 1/6) + h3*(-x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(4*x/3 - 1/6) + h2*(-8*x/3 + 4/3) + h3*(-x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], 2*h1/3 + 2*h2/3 + h3*(-x + 1/3 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 2:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h3*(1/8 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 3:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(1/6 + j),
                                   magmom=Amag, tag=tag)])
           elif i % 18 == 4:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(5/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 5:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 5/6) + h2*(-4*x/3 + 5/6) + h3*(x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(8*x/3 - 2/3) + h2*(-4*x/3 + 5/6) + h3*(x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-4*x/3 + 5/6) + h2*(8*x/3 - 2/3) + h3*(x/3 + 1/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h3*(x + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 6:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[0], h1/6 + h2/6 + h3*(1/3 + j),
                                   magmom=Amag, tag=tag),
                              Atom(symbol[1], 2*h1/3 + h2/6 + h3*(1/3 + j),
                                   magmom=Bmag, tag=tag),
                              Atom(symbol[1], h1/6 + 2*h2/3 + h3*(1/3 + j),
                                   magmom=Bmag, tag=tag)])
           elif i % 18 == 7:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(4*x/3 + 1/2) + h2*(4*x/3 + 1/2) + h3*(-x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-8*x/3 + 1) + h2*(4*x/3 + 1/2) + h3*(-x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(4*x/3 + 1/2) + h2*(-8*x/3 + 1) + h3*(-x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1/3 + h2/3 + h3*(-x + 2/3 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 8:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], 2*h1/3 + 2*h2/3 + h3*(11/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 9:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[0], h3*(1/2 + j),
                                  magmom=Amag, tag=tag)])
           elif i % 18 == 10:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(13/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 11:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 1/2) + h2*(-4*x/3 + 1/2) + h3*(x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], 8*h1*x/3 + h2*(-4*x/3 + 1/2) + h3*(x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-4*x/3 + 1/2) + 8*h2*x/3 + h3*(x/3 + 1/2 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], 2*h1/3 + 2*h2/3 + h3*(x + 1/3 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 12:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[0], 5*h1/6 + 5*h2/6 + h3*(2/3 + j),
                                   magmom=Amag, tag=tag),
                              Atom(symbol[1], h1/3 + 5*h2/6 + h3*(2/3 + j),
                                   magmom=Bmag, tag=tag),
                              Atom(symbol[1], 5*h1/6 + h2/3 + h3*(2/3 + j),
                                   magmom=Bmag, tag=tag)])
           elif i % 18 == 13:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(4*x/3 + 1/6) + h2*(4*x/3 + 1/6) + h3*(-x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-8*x/3 + 2/3) + h2*(4*x/3 + 1/6) + h3*(-x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(4*x/3 + 1/6) + h2*(-8*x/3 + 2/3) + h3*(-x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h3*(-x + 1 + j),
                                   magmom=Omag, tag=tag)])
           elif i % 18 == 14:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h1/3 + h2/3 + h3*(19/24 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 15:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(5/6 + j),
                                   magmom=Amag, tag=tag)])
           elif i % 18 == 16:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[1], h3*(7/8 + j),
                                   magmom=-Bmag, tag=tag)])
           elif i % 18 == 17:
               if i >= fixlayers[0] and i<= fixlayers[1]:
                   tag = 1
               else:
                   tag = 0
               layer = Atoms([Atom(symbol[2], h1*(-4*x/3 + 7/6) + h2*(-4*x/3 + 7/6) + h3*(x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(8*x/3 - 1/3) + h2*(-4*x/3 + 7/6) + h3*(x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1*(-4*x/3 + 7/6) + h2*(8*x/3 - 1/3) + h3*(x/3 + 5/6 + j), 
                                   magmom=Omag, tag=tag),
                              Atom(symbol[2], h1/3 + h2/3 + h3*(x + 2/3 + j),
                                   magmom=Omag, tag=tag)])
           slab.extend(layer)

   slab = slab*(area[0], area[1], 1)
   if vacuum != 0:
       slab.center(vacuum=vacuum, axis=2)
   slab = condense(slab)
   if fixlayers > 0:
       c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
       slab.set_constraint(c)
   return slab

def corundum0001(symbol=('Fe', 'O'), chex=13, ahex=5, z=0.144, x=0.31, area=(1,1), layers=18, vacuum=0,
                 mag=5, fixlayers=0, symmetry=False):
    """Returns an atoms object with the corundum(0001) surface

    Parameters
    ----------
    symbol: tuple of strings
        The A and B atoms used with stoichiometry A2B3
    chex: flt
        The height of the hexagonal unit cell. ~13
    ahex: flt
        The length of the base of the hexagonal unit cell. ~5
    z: flt
        The cation location parameter. Around ~ 0.144
    x: flt
        The oxygen parameter of the spinel structure. Around ~ 0.31
    area: tuple
        The amount of supercells in the (x,y) direction
    layers: int
        This is the number of layers in the supercell. For the
        corundum(0001) slab, a stoichiometric slab consists of muliples
        of 3 layers. Therefore, slabs with 3, 6, 9, etc layers have complete
        formula units. Furthermore, a complete repeating bulk structure
        requires 18 layers.
    vacuum: flt
        The amount of vacuum you want between slabs
    mag: tuple
        The initial magnetic moment of the cation
    fixlayers: int
        This determines the number of layers to fix for a relaxation.
        For a symmetric cell, this number must be even to keep the symmetry.
    symmetry: bool
        This specifies whether we want a symmetric or asymmetric relaxation
        Note: for a symmetric cell, only layers with multiples of six produce
        symmetric slabs.

    """
    fixlayers -= 1
    z = 0.5 - z
    x = 0.25 - x
    # Calculating primitive cell
    a = chex/(3.0*(3.0) ** (0.5)) + ahex/(3.0 * (2.0) ** (0.5))
    b = ahex*(1.0/(3.0*(2.0) ** 0.5) - 1.0/(2.0) ** 0.5) + chex/(3.0*(3.0) ** 0.5)
    # Primitive lattice vectors
    a1 = np.array([2**(0.5)/2*(b-a), 6**(0.5)/6*(b-a), 3**(0.5)/3*(2*a+b)])
    a2 = np.array([2**(0.5)/2*(a-b), 6**(0.5)/6*(b-a), 3**(0.5)/3*(2*a+b)])
    a3 = np.array([0               , 6**(0.5)/3*(a-b), 3**(0.5)/3*(2*a+b)])
    h1 = a2 - a1
    h2 = a3 - a1
    h3 = a1 + a2 + a3
    l1 = a1
    l2 = a1 + a2
    # Create the cell based on how large we want it to be
    slab = Atoms(cell=(h1, h2, layers*h3/18))
    # Differentiate between asymmetric and symmetric relaxations
    if symmetry == True and fixlayers > 0:
        mid_layer = layers / 2 - 0.5
        fixlayers = [mid_layer - fixlayers//2 - 1, mid_layer + fixlayers//2 + 1]
    elif symmetry == False and fixlayers > 0:
        fixlayers = [-1, -1 + fixlayers]
    else:
        fixlayers = [-1, -1]
    # Now add the layers
    for i in range(layers):
        j = i // 18
        if i % 18 == 0:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(z - 1/3 + j),
                                magmom=-mag, tag=tag)])
        elif i % 18 == 1:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[1], h1*(x + 5/12) + 2*h2/3 + h3*(1/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], 2*h1/3 + h2*(-x + 11/12) + h3*(1/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1*(-x + 11/12) + h2*(x + 5/12) + h3*(1/12 + j), 
                                magmom=0.00, tag=tag)])
        elif i % 18 == 2:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h3*(-z + 1/2 + j),
                                magmom=mag, tag=tag)])
        elif i % 18 == 3:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(z - 1/6 + j),
                                magmom=mag, tag=tag)])
        elif i % 18 == 4:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[1], h1*(-x + 1/4) + h3*(1/4 + j),
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h2*(x + 3/4) + h3*(1/4 + j),
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1*(x + 3/4) + h2*(-x + 1/4) + h3*(1/4 + j), 
                                magmom=0.00, tag=tag)])
        elif i % 18 == 5:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(-z + 2/3 + j),
                                magmom=-mag, tag=tag)])
        elif i % 18 == 6:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h3*(z + j),
                                magmom=-mag, tag=tag)])
        elif i % 18 == 7:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[1], h1*(x + 1/12) + h2/3 + h3*(5/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1/3 + h2*(-x + 7/12) + h3*(5/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1*(-x + 7/12) + h2*(x + 1/12) + h3*(5/12 + j), 
                                magmom=0.00, tag=tag)])
        elif i % 18 == 8:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(-z + 5/6 + j),
                                magmom=mag, tag=tag)])
        elif i % 18 == 9:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(z + 1/6 + j),
                                magmom=mag, tag=tag)])
        elif i % 18 == 10:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[1], h1*(-x + 11/12) + 2*h2/3 + h3*(7/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], 2*h1/3 + h2*(x + 5/12) + h3*(7/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1*(x + 5/12) + h2*(-x + 11/12) + h3*(7/12 + j), 
                                magmom=0.00, tag=tag)])
        elif i % 18 == 11:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h3*(-z + 1 + j),
                                magmom=-mag, tag=tag)])
        elif i % 18 == 12:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(z + 1/3 + j),
                                magmom=-mag, tag=tag)])
        elif i % 18 == 13:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[1], h1*(x + 3/4) + h3*(3/4 + j),
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h2*(-x + 1/4) + h3*(3/4 + j),
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1*(-x + 1/4) + h2*(x + 3/4) + h3*(3/4 + j), 
                                magmom=0.00, tag=tag)])
        elif i % 18 == 14:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h1/3 + h2/3 + h3*(-z + 7/6 + j),
                                magmom=mag, tag=tag)])
        elif i % 18 == 15:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], h3*(z + 1/2 + j),
                                magmom=mag, tag=tag)])
        elif i % 18 == 16:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[1], h1*(-x + 7/12) + h2/3 + h3*(11/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1/3 + h2*(x + 1/12) + h3*(11/12 + j), 
                                magmom=0.00, tag=tag),
                           Atom(symbol[1], h1*(x + 1/12) + h2*(-x + 7/12) + h3*(11/12+ j), 
                                magmom=0.00, tag=tag)])
        elif i % 18 == 17:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(symbol[0], 2*h1/3 + 2*h2/3 + h3*(-z + 4/3 + j),
                                magmom=-mag, tag=tag)])
        slab.extend(layer)

    slab = slab*(area[0], area[1], 1)
    if vacuum != 0:
        slab.center(vacuum=vacuum, axis=2)
    slab = condense(slab)
    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
        slab.set_constraint(c)
    return slab

def rutile110(symbol, a, c, u, area=(2,1), base=0, layers=6, vacuum=0, 
              mag=0.5, fixlayers=0, symmetry=False):
    """
    """

    B = symbol[0]
    X = symbol[1]

    # Cell lattice parameters from primitive lattice parameters
    a1 = np.array((0, 2. ** 0.5 * a / 2., 2. ** 0.5 * a / 2.))
    a2 = np.array((0, 2. ** 0.5 * a / 2., -2. ** 0.5 * a / 2.))
    a3 = np.array((c, 0, 0))
    b2 = a1 + a2
    b3 = a1 - a2
    b1 = a3

    # Create the cell based on how large we want it
    slab = Atoms(cell=(b1, b2, b3))
    
    # Differentiate between asymmetric and symmetric relaxations
    if symmetry == True and fixlayers > 0:
        mid_layer = layers / 2 - 0.5
        fixlayers = [mid_layer - fixlayers//2 - 1, mid_layer + fixlayers//2 + 1]
    elif symmetry == False and fixlayers > 0:
        fixlayers = [base, fixlayers + base - 1]
    else:
        fixlayers = [-1, -1]

    # Now add the layerse
    for i in range(base, base + layers):
        j = i // 6
        if i % 6 == 0:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(X, np.array((0, 0, 0)) + b3 * j, magmom=0, tag=tag)])
        elif i % 6 == 1:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(B, b1/2 + b3*(0.5 - u + j), magmom=mag, tag=tag),
                           Atom(B, b2/2 + b3*(0.5 - u + j), magmom=mag, tag=tag),
                           Atom(X, b1/2 + b2*(1 - u) + b3*(0.5 - u + j), magmom=0, tag=tag),
                           Atom(X, b1/2 + b2*u + b3*(0.5 - u + j), magmom=0, tag=tag)])
        elif i % 6 == 2:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(X, b3*(1 - 2*u + j), magmom=0, tag=tag)])
        elif i % 6 == 3:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(X, b2/2 + b3*(0.5 + j), magmom=0, tag=tag)])
        elif i % 6 == 4:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(B, b1/2 + b2/2 + b3*(1 - u + j), magmom=mag, tag=tag),
                           Atom(B, b3*(1 - u + j), magmom=mag, tag=tag),
                           Atom(X, b1/2 + b2*(u + 0.5) + b3*(1 - u + j), magmom=0, tag=tag),
                           Atom(X, b1/2 + b2*(0.5 - u) + b3*(1 - u + j), magmom=0, tag=tag)])
        elif i % 6 == 5:
            if i >= fixlayers[0] and i <= fixlayers[1]:
                tag = 1
            else:
                tag = 0
            layer = Atoms([Atom(X, b2/2 + b3*(1.5 - 2*u + j), magmom=0, tag=tag)])
        slab.extend(layer)

    slab = slab*(area[0], area[1], 1)

    if vacuum != 0:
        slab.center(vacuum=vacuum, axis=2)

    # slab = condense(slab)
    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
        slab.set_constraint(c)

    # We now want to add the hydrogens to the bottom of the slab. Note this only
    # works when base=2 and layers=7

    if not (base == 2 and layers == 7 and symmetry == False):
        return slab
    
    O_cus_pos = (slab.get_positions()[0], slab.get_positions()[13])
    O_bridge_pos = (slab.get_positions()[1], slab.get_positions()[14])
    layer = Atoms([Atom('H', O_bridge_pos[0] + np.array([0, 0.8, -0.6]), magmom=0),
                   Atom('H', O_bridge_pos[1] + np.array([0, 0.8, -0.6]), magmom=0),
                   Atom('H', O_cus_pos[0] + np.array([0.8, 0, -0.6]), magmom=0),
                   Atom('H', O_cus_pos[1] + np.array([0.8, 0, -0.6]), magmom=0)])
    slab.extend(layer)

    return slab


def add_adsorbate(slab_original, adsorbate, height, position=None, mol_index=None,
                  top=True, fix_relax=False):
    """Add an adsorbate to a surface

    This function adds an adsorbate to a slab. If the adsorbate is a molecule,
    the atom indexed by the mol_index optional argument is positioned on top
    the adsorption position on the surface if top is true. If not, then it is
    positioned on the bottom (useful for symmetric slabs).

    Parameters:
    -----------

    slab: Atoms object
        The slab onto which the adsorbate should be added
    adsorbate: Atoms object
        The adsorbate you wish to adsorb on
    height: flt
        Height above (or below) the surface
    position: tuple
        The x-y position of the adsorbate
    mol_index (default 0): int
        If the adsorbate is a molecule, index of the atom to be positioned
        above the location specified by the position argument. If the argument
        is none, the origin is taken as the place where the adsorbate should
        be attached.

    """
    slab = slab_original.copy()

    # Get the atom on the surface we want to attach the atom to
    if top == True:
        a = slab.positions[:, 2].argmax()
    else:
        a = slab.positions[:, 2].argmin()

    # Get the x,y position to attach the adsorbate to
    if position is None:
        pos = np.array((slab.positions[a,0], slab.positions[a,1]))
    else:
        pos = np.array(position) # (x, y) par

    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, Atoms):
        ads = adsorbate.copy()
    elif isinstance(adsorbate, Atom):
        ads = Atoms([adsorbate])
    else:
        raise TypeError('Adsorbate needs to be an Atom or Atoms object')
        
    # Flip the adsorbate if its supposed to be on the bottom
    if top == False:
        ads.positions = -ads.positions

    # Get the z-coordinate and attach it
    if top == True:
        z = slab.positions[a, 2] + height
    else:
        z = slab.positions[a, 2] - height
    if mol_index is not None:
        ads.translate([pos[0], pos[1], z] - ads.positions[mol_index])
    else:
        ads.translate([pos[0], pos[1], z])
    slab.extend(ads)
    
    return slab

def rotate_111(atoms):
    '''The point of this function is to rotate rocksalt/spinel surfaces
    in the 111 direction so that the eg and t2g orbitals accurately
    represent the correct projections'''

    scaled_pos = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    a1 = cell[0]
    a2 = cell[1]
    a3 = cell[2]

    r1 = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/2, 0.],
                   [np.sqrt(2)/2,  -np.sqrt(2)/2, 0.],
                   [0        , 0         , 1]])
    r2 = np.array([[1, 0             , 0               ],
                   [0, -1/np.sqrt(3)     , -np.sqrt(2)/np.sqrt(3)],
                   [0, np.sqrt(2)/np.sqrt(3), -1/np.sqrt(3)      ]])
    
    a1 = np.dot(r1, np.dot(r2, a1))
    a2 = np.dot(r1, np.dot(r2, a2))
    a3 = np.dot(r1, np.dot(r2, a3))    
    atoms.set_cell((a1, a2, a3))
    atoms.set_scaled_positions(scaled_pos)
    
    return atoms

def condense(atoms):
    # First, create a set of the magmoms
    magmoms = list(set(atoms.get_initial_magnetic_moments()))

    # Now, make lists to store each atom of type
    temp = []
    for i in magmoms:
        temp.append([])
    for atom in atoms:
        i = 0
        for magmom in magmoms:
            if atom.magmom == magmom:
                atom.cut_reference_to_atoms()
                temp[i].append(atom)
            i += 1
    new_atoms = Atoms(cell=atoms.get_cell())
    for group in temp:
        for atom in group:
            new_atoms.extend(atom)
    return new_atoms
