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
            fixlayers -= 1
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
    # slab = condense(slab)
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
   # slab = condense(slab)
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
   # slab = condense(slab)
   if fixlayers > 0:
       c = FixAtoms(indices=[atom.index for atom in slab if atom.tag == 1])
       slab.set_constraint(c)
   return slab

def corundum10_12(symbols, chex=None, c_over_a=2.766, z=0.144, x=0.309, mags=(2, 2),
                  vol=None, vacuum=0, fixlayers=2):
    '''Returns a corundum atoms object
    
    Parameters
    ----------
    symbol: tuple
        The atoms in the unit cell.
    chex: flt
        This is the hexagonal height. 
    c_over_a: flt
        This is the hexagonal height to hexagonal base length ratio
    z: flt
        This is the cation position parameter
    x: flt
        This is the anion positoin parameter
    mags: tuple
        This is the magnetic moments of the cation and anion, respectively
    vol: flt
        The volume of the primitive cell
    NOTE: One must provide either the volume or the lattice 
    constant.
    '''
    up = mags[0]
    down = -mags[0]
    if chex == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif chex != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        chex = ((6 * vol * (c_over_a) ** 2) / 3 ** 0.5) ** (1/3)
    ahex = chex / c_over_a
    x = 0.25 - x
    a = chex/(3.0*(3.0) ** (0.5)) + ahex/(3.0 * (2.0) ** (0.5))
    b = ahex*(1.0/(3.0*(2.0) ** 0.5) - 1.0/(2.0) ** 0.5) + chex/(3.0*(3.0) ** 0.5)
    a1 = np.array([b,a,a])
    a2 = np.array([a,b,a])
    a3 = np.array([a,a,b])
    b1 = -a3 + a2
    b2 = -a1 + a2 + a3
    b3 = a1
    Rx = np.array([[1, 0, 0],
                   [0, 1.0/2.0**0.5, -1.0/2.0**0.5],
                   [0, 1.0/2.0**0.5, 1.0/2.0**0.5]])
    b1 = np.dot(Rx, b1)
    b2 = np.dot(Rx, b2)
    b3 = np.dot(Rx, b3)    

    hyp = (2*b**2 + (2*a - b)**2)**0.5

    Ry = np.array([[(2*a - b)/hyp, 0, b*2**0.5/hyp],
                   [0            , 1, 0            ],
                   [-b*2**0.5/hyp , 0, (2*a - b)/hyp]])

    b1 = np.dot(Ry, b1)
    b2 = np.dot(Ry, b2)
    b3 = np.dot(Ry, b3)    

    Rz = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]])

    b1 = np.dot(Rz, b1)
    b2 = np.dot(Rz, b2)
    b3 = np.dot(Rz, b3)
        
    bulk = Atoms([Atom(symbols[1], b1*(-x + 3.0/4.0) + b2/4.0, tag=1),                                                # 0
                  Atom(symbols[1], b1*(x/2.0 + 3.0/8.0) + b2*(x/2.0 + 1.0/8.0) + b3*(-x/2.0 + 1.0/8.0), tag=1),       # 1
                  Atom(symbols[1], b1*(-x/2.0 + 5.0/8.0) + b2*(x/2.0 + 5.0/8.0) + b3*(-x/2.0 + 1.0/8.0), tag=1),      # 2
                  Atom(symbols[0], b1/2.0 + b2*(-z + 1.0/2.0) + b3*(-2*z + 1.0/2.0), magmom=up, tag=1),                          # 3 
                  Atom(symbols[0], 1.0/2.0*b1 + b2*(-z + 1.0) + b3*(-2*z + 1.0/2.0), magmom=down, tag=1),                          # 4 
                  Atom(symbols[0], b2*z + 2*b3*z, magmom=up, tag=1),                                                             # 5
                  Atom(symbols[0], b2*(z + 1.0/2.0) + 2*b3*z, magmom=down, tag=1),                                                 # 6
                  Atom(symbols[1], b1*(-x/2.0 + 1.0/8.0) + b2*(-x/2.0 + 3.0/8.0) + b3*(x/2.0 + 3.0/8.0), tag=1),      # 7
                  Atom(symbols[1], b1*(x/2 + 7.0/8.0) + b2*(-x/2.0 + 7.0/8.0) + b3*(x/2.0 + 3.0/8.0), tag=1),         # 8
                  Atom(symbols[1], b1*(-x + 1.0/4.0) + 3.0*b2/4.0 + b3/2.0, tag=1),                                   # 9
                  
                  Atom(symbols[1], b1*(x + 3.0/4.0) + b2/4.0 + b3/2.0, tag=2),                                        # 10
                  Atom(symbols[0], b2*(-z + 1.0/2.0) + b3*(-2*z + 1.0), magmom=down, tag=2),                                       # 11
                  Atom(symbols[1], b1*(x/2.0 + 7.0/8.0) + b2*(x/2.0 + 5.0/8.0) + b3*(-x/2.0 + 5.0/8.0), tag=2),       # 12
                  Atom(symbols[1], b1*(-x/2.0 + 1.0/8.0) + b2*(x/2.0 + 1.0/8.0) + b3*(-x/2.0 + 5.0/8.0), tag=2),      # 13
                  Atom(symbols[0], b2*(-z + 1) + b3*(-2*z + 1), magmom=up, tag=2),                                               # 14
                  Atom(symbols[0], b1/2.0 + b2*(z + 1.0/2.0) + b3*(2*z + 1.0/2.0), magmom=up, tag=2),                            # 15
                  Atom(symbols[0], 1.0/2.0*b1 + b2*z + b3*(2*z + 1.0/2.0), magmom=down, tag=2),                                    # 16
                  Atom(symbols[1], b1*(-x/2.0 + 5.0/8.0) + b2*(-x/2.0 + 7.0/8.0) + b3*(x/2.0 + 7.0/8.0), tag=2),      # 17
                  Atom(symbols[1], b1*(x/2.0 + 3.0/8.0) + b2*(-x/2.0 + 3.0/8.0) + b3*(x/2.0 + 7.0/8.0), tag=2),       # 18
                  Atom(symbols[1], b1*(x + 1.0/4.0) + 3.0*b2/4.0 + b3, tag=2),                                        # 19

                  Atom(symbols[1], b1*(-x + 3.0/4.0) + b2/4.0 + b3, tag=3),                                           # 20
                  Atom(symbols[1], b1*(x/2.0 + 3.0/8.0) + b2*(x/2.0 + 1.0/8.0) + b3*(-x/2.0 + 1.0/8.0) + b3, tag=3),  # 21
                  Atom(symbols[1], b1*(-x/2.0 + 5.0/8.0) + b2*(x/2.0 + 5.0/8.0) + b3*(-x/2.0 + 1.0/8.0) + b3, tag=3), # 22
                  Atom(symbols[0], b1/2.0 + b2*(-z + 1.0/2.0) + b3*(-2*z + 1.0/2.0) + b3, magmom=up, tag=3),                     # 23
                  Atom(symbols[0], 1.0/2.0*b1 + b2*(-z + 1.0) + b3*(-2*z + 1.0/2.0) + b3, magmom=down, tag=3),                     # 24
                  Atom(symbols[0], b2*z + 2*b3*z + b3, magmom=up, tag=3),                                                        # 25
                  Atom(symbols[0], b2*(z + 1.0/2.0) + 2*b3*z + b3, magmom=down, tag=3),                                            # 26
                  Atom(symbols[1], b1*(-x/2.0 + 1.0/8.0) + b2*(-x/2.0 + 3.0/8.0) + b3*(x/2.0 + 3.0/8.0) + b3, tag=3), # 27
                  Atom(symbols[1], b1*(x/2 + 7.0/8.0) + b2*(-x/2.0 + 7.0/8.0) + b3*(x/2.0 + 3.0/8.0) + b3, tag=3),    # 28
                  Atom(symbols[1], b1*(-x + 1.0/4.0) + 3.0*b2/4.0 + b3/2.0 + b3, tag=3),                              # 29
                  
                  Atom(symbols[1], b1*(x + 3.0/4.0) + b2/4.0 + b3/2.0 + b3, tag=4),                                   # 30
                  Atom(symbols[0], b2*(-z + 1.0/2.0) + b3*(-2*z + 1.0) + b3, magmom=down, tag=4),                                  # 31
                  Atom(symbols[1], b1*(x/2.0 + 7.0/8.0) + b2*(x/2.0 + 5.0/8.0) + b3*(-x/2.0 + 5.0/8.0) + b3, tag=4),  # 32
                  Atom(symbols[1], b1*(-x/2.0 + 1.0/8.0) + b2*(x/2.0 + 1.0/8.0) + b3*(-x/2.0 + 5.0/8.0) + b3, tag=4), # 33
                  Atom(symbols[0], b2*(-z + 1) + b3*(-2*z + 1) + b3, magmom=up, tag=4),                                          # 34
                  Atom(symbols[0], b1/2.0 + b2*(z + 1.0/2.0) + b3*(2*z + 1.0/2.0) + b3, magmom=up, tag=4),                       # 35
                  Atom(symbols[0], 1.0/2.0*b1 + b2*z + b3*(2*z + 1.0/2.0) + b3, magmom=down, tag=4),                               # 36
                  Atom(symbols[1], b1*(-x/2.0 + 5.0/8.0) + b2*(-x/2.0 + 7.0/8.0) + b3*(x/2.0 + 7.0/8.0) + b3, tag=4), # 37
                  Atom(symbols[1], b1*(x/2.0 + 3.0/8.0) + b2*(-x/2.0 + 3.0/8.0) + b3*(x/2.0 + 7.0/8.0) + b3, tag=4),  # 38
                  Atom(symbols[1], b1*(x + 1.0/4.0) + 3.0*b2/4.0 + b3 + b3, tag=4),                                   # 39
              ],
                 cell=(b1, b2, 2*b3))

    c = FixAtoms(indices=[atom.index for atom in bulk if atom.tag <= fixlayers])
    bulk.set_constraint(c)

    if vacuum != 0:
        slab.center(vacuum=vacuum, axis=2)
    
    return bulk

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
    # slab = condense(slab)
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

def anatase001(B='Ti', X='O', a=3.7842, c=2*4.7573, z=0.0831, mags=[0.5, 0], vacuum=10, area=(2, 2), fixlayers=0):
    '''http://cst-www.nrl.navy.mil/lattice/struk/c5.html
    spacegroup 141 (I4_1/amd)

    note: this is different than the structure in ref Phys Rev B 65, 224112

    Basis Vectors:
Atom    Lattice Coordinates                Cartesian Coordinates

Ti -0.12500000  0.62500000  0.25000000     0.00000000  2.83815000  1.18932500
Ti  0.12500000  0.37500000  0.75000000     1.89210000  2.83815000  3.56797500
O  -0.08310000  0.16690000  0.16620000     0.00000000  0.94605000  0.79066326
O  -0.33310000  0.41690000  0.66620000     0.00000000  2.83815000  3.16931326
O   0.08310000 -0.16690000  0.83380000     1.89210000  0.94605000  3.96663674
O   0.33310000  0.58310000  0.33380000     1.89210000  2.83815000  1.58798674
    '''
    
    m1 = mags[0]
    m2 = mags[1]

    a1 = a*np.array([1.0, 0.0, 0.0])
    a2 = a*np.array([0.0, 1.0, 0.0])
    a3 = np.array([0.5*a, 0.5*a, 0.5*c])

    b3 = 2*a3 - a1 - a2

    atoms = Atoms([Atom(B, -0.125*a1 + 0.625*a2 + 0.25*a3, tag=1, magmom=m1),
                   Atom(B,  0.125*a1 + 0.375*a2 + 0.75*a3, tag=2, magmom=m1),
                   Atom(X, -z*a1 + (0.25-z)*a2 + 2*z*a3, tag=1, magmom=m2),
                   Atom(X, -(0.25+z)*a1 + (0.5-z)*a2 + (0.5+2*z)*a3, tag=2, magmom=m2),
                   Atom(X, z*a1 - (0.25 - z)*a2 + (1-2*z)*a3, tag=2, magmom=m2),
                   Atom(X, (0.25 + z)*a1 + (0.5 + z)*a2 + (0.5-2*z)*a3, tag=1, magmom=m2),
                   Atom(B, -0.125*a1 + 0.625*a2 + 0.25*a3 + a3 - a2, tag=3, magmom=m1),
                   Atom(B,  0.125*a1 + 0.375*a2 + 0.75*a3 + a3 - a2 - a1, tag=4, magmom=m1),
                   Atom(X, -z*a1 + (0.25-z)*a2 + 2.*z*a3 + a3, tag=3, magmom=m2),
                   Atom(X, -(0.25+z)*a1 + (0.5-z)*a2 + (0.5+2*z)*a3 + a3 - a2, tag=4, magmom=m2),
                   Atom(X, z*a1 - (0.25 - z)*a2 + (1-2*z)*a3 + a3 - a1, tag=4, magmom=m2),
                   Atom(X, (0.25 + z)*a1 + (0.5 + z)*a2 + (0.5-2*z)*a3 + a3 - a2 - a1, tag=3, magmom=m2)],
                   cell=[a1,a2,b3])

    if vacuum != 0:
        atoms.center(vacuum=vacuum, axis=2)
        
    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.tag <= fixlayers])
        atoms.set_constraint(c)

    atoms *= (2, 2, 1)

    return atoms

def columbite101(B='Pb', X='O', a=4.947, b=5.951, c=5.497, mags=[0.5, 0],
                 u=0.178, x=0.276, y=0.41, z=0.425, vacuum=10, fixlayers=0):
    '''
    the parameters are derived from the output of the sg code above for PbO2.

    Phys Rev B, 65, 224112
    spacegroup 60 (Pbcn)

    note: there seems to be a typo in the description for columbite in
    Phys Rev B, 65, 224112. for two B atoms I think 0.5 should be 0.25
    '''

    m1 = mags[0]
    m2 = mags[1]

    b1 = np.array([(a**2 + c**2)**0.5, 0, 0])
    b2 = b*np.array([0.0, 1.0, 0.0])
    b3 = np.array([(a**2 - c**2) / (a**2 + c**2)**0.5, 0, 2*a*c/(a**2 + c**2)**0.5])

    atoms = Atoms([Atom(B, 7*b1/8 + b2*u + b3/8, magmom=m1, tag=1),
                   Atom(B, 5*b1/8 + b2*(-u + 1) + 3*b3/8, magmom=m1, tag=2),
                   Atom(B, b1/8 + b2*(u + 0.5) + 3*b3/8, magmom=m1, tag=2),
                   Atom(B, 7*b1/8 + b2*(-u + 0.5) + 5*b3/8, magmom=m1, tag=3),
                   Atom(X, b1*(x/2 - z/2 + 1) + b2*y + b3*(x/2 + z/2), magmom=m2, tag=2),
                   Atom(X, b1*(-x/2 + z/2) + b2*(-y + 1) + b3*(-x/2 - z/2 + 1), magmom=m2, tag=3),
                   Atom(X, b1*(-x/2 - z/2 + 1) + b2*(-y + 0.5) + b3*(-x/2 + z/2 + 0.5), magmom=m2, tag=3),
                   Atom(X, b1*(x/2 + z/2) + b2*(y + 0.5) + b3*(x/2 - z/2 + 0.5), magmom=m2, tag=2),
                   Atom(X, b1*(x/2 + z/2 - 0.25) + b2*(-y + 0.5) + b3*(x/2 - z/2 + 0.75), magmom=m2, tag=3),
                   Atom(X, b1*(-x/2 - z/2 + 1.25) + b2*(y + 0.5) + b3*(-x/2 + z/2 + 0.25), magmom=m2, tag=2),
                   Atom(X, b1*(-x/2 + z/2 + 0.25) + b2*y + b3*(-x/2 - z/2 + 0.75), magmom=m2, tag=2),
                   Atom(X, b1*(x/2 - z/2 + 0.75) + b2*(-y + 1) + b3*(x/2 + z/2 + 0.25), magmom=m2, tag=3),
                   Atom(B, 3*b1/8 + b2*u + 5*b3/8, magmom=m1, tag=3),
                   Atom(B, b1/8 + b2*(-u + 1) + 7*b3/8, magmom=m1, tag=4),
                   Atom(B, 5*b1/8 + b2*(u + 0.5) + 7*b3/8, magmom=m1, tag=4),
                   Atom(B, 3*b1/8 + b2*(-u + 0.5) + b3/8, magmom=m1, tag=1),
                   Atom(X, b1*(x/2 - z/2 + 0.5) + b2*y + b3*(x/2 + z/2 + 0.5), magmom=m2, tag=4),
                   Atom(X, b1*(-x/2 + z/2 + 0.5) + b2*(-y + 1) + b3*(-x/2 - z/2 + 0.5), magmom=m2, tag=1),
                   Atom(X, b1*(-x/2 - z/2 + 0.5) + b2*(-y + 0.5) + b3*(-x/2 + z/2), magmom=m2, tag=1),
                   Atom(X, b1*(x/2 + z/2 + 0.5) + b2*(y + 0.5) + b3*(x/2 - z/2 + 1), magmom=m2, tag=4),
                   Atom(X, b1*(x/2 + z/2 + 0.25) + b2*(-y + 0.5) + b3*(x/2 - z/2 + 0.25), magmom=m2, tag=1),
                   Atom(X, b1*(-x/2 - z/2 + 0.75) + b2*(y + 0.5) + b3*(-x/2 + z/2 + 0.75), magmom=m2, tag=4),
                   Atom(X, b1*(-x/2 + z/2 + 0.75) + b2*y + b3*(-x/2 - z/2 + 1.25), magmom=m2, tag=4),
                   Atom(X, b1*(x/2 - z/2 + 0.25) + b2*(-y + 1) + b3*(x/2 + z/2 - 0.25), magmom=m2, tag=1),

                   Atom(B, 7*b1/8 + b2*u + b3/8 + b3, magmom=m1, tag=5),
                   Atom(B, 5*b1/8 + b2*(-u + 1) + 3*b3/8 + b3, magmom=m1, tag=6),
                   Atom(B, b1/8 + b2*(u + 0.5) + 3*b3/8 + b3, magmom=m1, tag=6),
                   Atom(B, 7*b1/8 + b2*(-u + 0.5) + 5*b3/8 + b3, magmom=m1, tag=7),
                   Atom(X, b1*(x/2 - z/2 + 1) + b2*y + b3*(x/2 + z/2) + b3, magmom=m2, tag=6),
                   Atom(X, b1*(-x/2 + z/2) + b2*(-y + 1) + b3*(-x/2 - z/2 + 1) + b3, magmom=m2, tag=7),
                   Atom(X, b1*(-x/2 - z/2 + 1) + b2*(-y + 0.5) + b3*(-x/2 + z/2 + 0.5) + b3, magmom=m2, tag=7),
                   Atom(X, b1*(x/2 + z/2) + b2*(y + 0.5) + b3*(x/2 - z/2 + 0.5) + b3, magmom=m2, tag=6),
                   Atom(X, b1*(x/2 + z/2 - 0.25) + b2*(-y + 0.5) + b3*(x/2 - z/2 + 0.75) + b3, magmom=m2, tag=7),
                   Atom(X, b1*(-x/2 - z/2 + 1.25) + b2*(y + 0.5) + b3*(-x/2 + z/2 + 0.25) + b3, magmom=m2, tag=6),
                   Atom(X, b1*(-x/2 + z/2 + 0.25) + b2*y + b3*(-x/2 - z/2 + 0.75) + b3, magmom=m2, tag=6),
                   Atom(X, b1*(x/2 - z/2 + 0.75) + b2*(-y + 1) + b3*(x/2 + z/2 + 0.25) + b3, magmom=m2, tag=7),
                   Atom(B, 3*b1/8 + b2*u + 5*b3/8 + b3, magmom=m1, tag=7),
                   Atom(B, b1/8 + b2*(-u + 1) + 7*b3/8 + b3, magmom=m1, tag=8),
                   Atom(B, 5*b1/8 + b2*(u + 0.5) + 7*b3/8 + b3, magmom=m1, tag=8),
                   Atom(B, 3*b1/8 + b2*(-u + 0.5) + b3/8 + b3, magmom=m1, tag=5),
                   Atom(X, b1*(x/2 - z/2 + 0.5) + b2*y + b3*(x/2 + z/2 + 0.5) + b3, magmom=m2, tag=8),
                   Atom(X, b1*(-x/2 + z/2 + 0.5) + b2*(-y + 1) + b3*(-x/2 - z/2 + 0.5) + b3, magmom=m2, tag=5),
                   Atom(X, b1*(-x/2 - z/2 + 0.5) + b2*(-y + 0.5) + b3*(-x/2 + z/2) + b3, magmom=m2, tag=5),
                   Atom(X, b1*(x/2 + z/2 + 0.5) + b2*(y + 0.5) + b3*(x/2 - z/2 + 1) + b3, magmom=m2, tag=8),
                   Atom(X, b1*(x/2 + z/2 + 0.25) + b2*(-y + 0.5) + b3*(x/2 - z/2 + 0.25) + b3, magmom=m2, tag=5),
                   Atom(X, b1*(-x/2 - z/2 + 0.75) + b2*(y + 0.5) + b3*(-x/2 + z/2 + 0.75) + b3, magmom=m2, tag=8),
                   Atom(X, b1*(-x/2 + z/2 + 0.75) + b2*y + b3*(-x/2 - z/2 + 1.25) + b3, magmom=m2, tag=8),
                   Atom(X, b1*(x/2 - z/2 + 0.25) + b2*(-y + 1) + b3*(x/2 + z/2 - 0.25) + b3, magmom=m2, tag=5)],
                  cell = [b1, b2, 2*b3])

    if vacuum != 0:
        atoms.center(vacuum=vacuum, axis=2)
        
    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.tag <= fixlayers])
        atoms.set_constraint(c)


    return atoms

def pyrite001(B='Ir', X='O', a=5.407, u=2.0871/5.407, mags=[0.5, 0], vacuum=10, fixlayers=0):
    '''http://cst-www.nrl.navy.mil/lattice/struk/c2.html

    spacegroup 205 Pa\overline{3}
    http://cst-www.nrl.navy.mil/lattice/struk.xmol/c2.pos'''

    m1, m2 = mags

    a1 = a*np.array([1.0, 0.0, 0.0])
    a2 = a*np.array([0.0, 1.0, 0.0])
    a3 = a*np.array([0.0, 0.0, 1.0])

    atoms = Atoms([Atom(B, 0.25*a3, magmom=m1, tag=1),
                   Atom(B, 0.5*a2 + 0.75*a3, magmom=m1, tag=2),
                   Atom(B, 0.5*a1 + 0.75*a3, magmom=m1, tag=2),
                   Atom(B, 0.5*a1 + 0.5*a2 + 0.25*a3, magmom=m1, tag=1),
                   Atom(X, u*a1 + u*a2 + (u + 0.25)*a3, magmom=m2, tag=2),
                   Atom(X, (1 - u)*a1 + (1 - u)*a2 + (1.25 - u)*a3, magmom=m2, tag=2),
                   Atom(X, (0.5 + u)*a1 + (0.5 - u)*a2 + (1.25 - u)*a3, magmom=m2, tag=2),
                   Atom(X, (0.5 - u)*a1 + (u - 0.5)*a2 + (u + 0.25)*a3, magmom=m2, tag=2),
                   Atom(X, (1 - u)*a1 + (0.5 + u)*a2 + (0.75 - u)*a3, magmom=m2, tag=1),
                   Atom(X, u*a1 + (0.5 - u)*a2 + (u - 0.25)*a3, magmom=m2, tag=1),
                   Atom(X, (0.5 - u)*a1 + (1 - u)*a2 + (u - 0.25)*a3, magmom=m2, tag=1),
                   Atom(X, (u + 0.5)*a1 + u*a2 + (0.75 - u)*a3, magmom=m2, tag=1),
                   Atom(B, 0.25*a3 + a3, magmom=m1, tag=3),
                   Atom(B, 0.5*a2 + 0.75*a3 + a3, magmom=m1, tag=4),
                   Atom(B, 0.5*a1 + 0.75*a3 + a3, magmom=m1, tag=4),
                   Atom(B, 0.5*a1 + 0.5*a2 + 0.25*a3 + a3, magmom=m1, tag=3),
                   Atom(X, u*a1 + u*a2 + (u + 0.25)*a3 + a3, magmom=m2, tag=4),
                   Atom(X, (1 - u)*a1 + (1 - u)*a2 + (1.25 - u)*a3 + a3, magmom=m2, tag=4),
                   Atom(X, (0.5 + u)*a1 + (0.5 - u)*a2 + (1.25 - u)*a3 + a3, magmom=m2, tag=4),
                   Atom(X, (0.5 - u)*a1 + (u - 0.5)*a2 + (u + 0.25)*a3 + a3, magmom=m2, tag=4),
                   Atom(X, (1 - u)*a1 + (0.5 + u)*a2 + (0.75 - u)*a3 + a3, magmom=m2, tag=3),
                   Atom(X, u*a1 + (0.5 - u)*a2 + (u - 0.25)*a3 + a3, magmom=m2, tag=3),
                   Atom(X, (0.5 - u)*a1 + (1 - u)*a2 + (u - 0.25)*a3 + a3, magmom=m2, tag=3),
                   Atom(X, (u + 0.5)*a1 + u*a2 + (0.75 - u)*a3 + a3, magmom=m2, tag=3)],
                   cell=[a1,a2,2*a3])

    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.tag <= fixlayers])
        atoms.set_constraint(c)

    if vacuum != 0:
        atoms.center(vacuum=vacuum, axis=2)

    return atoms

def brookite110(B='Ti', X='O', a=9.16, b=5.43, c=5.13,
                x1=0.12,  x2=0.01, x3=0.23,
                y1=0.11,  y2=0.15, y3=0.10,
                z1=-0.12, z2=0.18, z3=-0.46, mags=[0.5, 0], vacuum=10, fixlayers=0):

    m1, m2 = mags

    h1 = np.array([0.0, (a**2 + b**2)**0.5, 0.0])
    h2 = np.array([0, (a**2 - b**2) / (a**2 + b**2)**0.5, 2*a*b/(a**2 + b**2)**0.5])
    h3 = c * np.array([1.0, 0.0, 0.0])

    atoms = Atoms([Atom(B, h1*(x1/2 - y1/2) + h2*(x1/2 + y1/2) + h3*(z1 + 1), magmom=m1, tag=1),                    # 0
                   Atom(B, h1*(x1/2 + y1/2) + h2*(x1/2 - y1/2 + 0.5) - h3*z1, magmom=m1, tag=2),                    # 1
                   Atom(B, h1*(-x1/2 - y1/2 + 0.75) + h2*(-x1/2 + y1/2 + 0.25) + h3*(-z1 + 0.5), magmom=m1, tag=2), # 2
                   Atom(B, h1*(-x1/2 + y1/2 + 0.25) + h2*(-x1/2 - y1/2 + 0.25) + h3*(z1 + 0.5), magmom=m1, tag=1),  # 3
                   Atom(B, h1*(-x1/2 + y1/2 + 1) + h2*(-x1/2 - y1/2 + 1) - h3*z1, magmom=m1, tag=4),                # 4
                   Atom(B, h1*(-x1/2 - y1/2 + 1) + h2*(-x1/2 + y1/2 + 0.5) + h3*(z1 + 1), magmom=m1, tag=3),        # 5
                   Atom(B, h1*(x1/2 + y1/2 + 0.75) + h2*(x1/2 - y1/2 + 0.25) + h3*(z1 + 0.5), magmom=m1, tag=2),    # 6
                   Atom(B, h1*(x1/2 - y1/2 + 0.25) + h2*(x1/2 + y1/2 + 0.25) + h3*(-z1 + 0.5), magmom=m1, tag=2),   # 7
                   Atom(X, h1*(x2/2 - y2/2 + 1) + h2*(x2/2 + y2/2) + h3*z2, magmom=m2, tag=1),                      # 8
                   Atom(X, h1*(x2/2 + y2/2) + h2*(x2/2 - y2/2 + 0.5) + h3*(-z2 + 1), magmom=m2, tag=2),             # 9 
                   Atom(X, h1*(-x2/2 - y2/2 + 0.75) + h2*(-x2/2 + y2/2 + 0.25) + h3*(-z2 + 0.5), magmom=m2, tag=2), # 10
                   Atom(X, h1*(-x2/2 + y2/2 + 0.25) + h2*(-x2/2 - y2/2 + 0.25) + h3*(z2 + 0.5), magmom=m2, tag=1),  # 11
                   Atom(X, h1*(-x2/2 + y2/2) + h2*(-x2/2 - y2/2 + 1) + h3*(-z2 + 1), magmom=m2, tag=4),             # 12
                   Atom(X, h1*(-x2/2 - y2/2 + 1) + h2*(-x2/2 + y2/2 + 0.5) + h3*z2, magmom=m2, tag=3),              # 13
                   Atom(X, h1*(x2/2 + y2/2 + 0.75) + h2*(x2/2 - y2/2 + 0.25) + h3*(z2 + 0.5), magmom=m2, tag=2),    # 14
                   Atom(X, h1*(x2/2 - y2/2 + 0.25) + h2*(x2/2 + y2/2 + 0.25) + h3*(-z2 + 0.5), magmom=m2, tag=2),   # 15
                   Atom(X, h1*(x3/2 - y3/2) + h2*(x3/2 + y3/2) + h3*(z3 + 1), magmom=m2, tag=1),                    # 16
                   Atom(X, h1*(x3/2 + y3/2) + h2*(x3/2 - y3/2 + 0.5) - h3*z3, magmom=m2, tag=3),                    # 17
                   Atom(X, h1*(-x3/2 - y3/2 + 0.75) + h2*(-x3/2 + y3/2 + 0.25) + h3*(-z3 + 0.5), magmom=m2, tag=2), # 18
                   Atom(X, h1*(-x3/2 + y3/2 + 0.25) + h2*(-x3/2 - y3/2 + 0.25) + h3*(z3 + 0.5), magmom=m2, tag=1),  # 19
                   Atom(X, h1*(-x3/2 + y3/2 + 1) + h2*(-x3/2 - y3/2 + 1) - h3*z3, magmom=m2, tag=4),                # 20
                   Atom(X, h1*(-x3/2 - y3/2 + 1) + h2*(-x3/2 + y3/2 + 0.5) + h3*(z3 + 1), magmom=m2, tag=2),        # 21
                   Atom(X, h1*(x3/2 + y3/2 + 0.75) + h2*(x3/2 - y3/2 + 0.25) + h3*(z3 + 0.5), magmom=m2, tag=2),    # 22
                   Atom(X, h1*(x3/2 - y3/2 + 0.25) + h2*(x3/2 + y3/2 + 0.25) + h3*(-z3 + 0.5), magmom=m2, tag=2),   # 23
                   Atom(B, h1*(x1/2 - y1/2 + 0.5) + h2*(x1/2 + y1/2 + 0.5) + h3*(z1 + 1), magmom=m1, tag=3),        # 24
                   Atom(B, h1*(x1/2 + y1/2 + 0.5) + h2*(x1/2 - y1/2) - h3*z1, magmom=m1, tag=1),                    # 25
                   Atom(B, h1*(-x1/2 - y1/2 + 0.25) + h2*(-x1/2 + y1/2 + 0.75) + h3*(-z1 + 0.5), magmom=m1, tag=3), # 26
                   Atom(B, h1*(-x1/2 + y1/2 + 0.75) + h2*(-x1/2 - y1/2 + 0.75) + h3*(z1 + 0.5), magmom=m1, tag=3),  # 27
                   Atom(B, h1*(-x1/2 + y1/2 + 0.5) + h2*(-x1/2 - y1/2 + 0.5) - h3*z1, magmom=m1, tag=2),            # 28
                   Atom(B, h1*(-x1/2 - y1/2 + 0.5) + h2*(-x1/2 + y1/2 + 1) + h3*(z1 + 1), magmom=m1, tag=4),        # 29
                   Atom(B, h1*(x1/2 + y1/2 + 0.25) + h2*(x1/2 - y1/2 + 0.75) + h3*(z1 + 0.5), magmom=m1, tag=3),    # 30
                   Atom(B, h1*(x1/2 - y1/2 + 0.75) + h2*(x1/2 + y1/2 + 0.75) + h3*(-z1 + 0.5), magmom=m1, tag=4),   # 31
                   Atom(X, h1*(x2/2 - y2/2 + 0.5) + h2*(x2/2 + y2/2 + 0.5) + h3*z2, magmom=m2, tag=3),              # 32
                   Atom(X, h1*(x2/2 + y2/2 + 0.5) + h2*(x2/2 - y2/2 + 1) + h3*(-z2 + 1), magmom=m2, tag=4),         # 33
                   Atom(X, h1*(-x2/2 - y2/2 + 0.25) + h2*(-x2/2 + y2/2 + 0.75) + h3*(-z2 + 0.5), magmom=m2, tag=3), # 34
                   Atom(X, h1*(-x2/2 + y2/2 + 0.75) + h2*(-x2/2 - y2/2 + 0.75) + h3*(z2 + 0.5), magmom=m2, tag=3),  # 35
                   Atom(X, h1*(-x2/2 + y2/2 + 0.5) + h2*(-x2/2 - y2/2 + 0.5) + h3*(-z2 + 1), magmom=m2, tag=2),     # 36
                   Atom(X, h1*(-x2/2 - y2/2 + 0.5) + h2*(-x2/2 + y2/2) + h3*z2, magmom=m2, tag=1),                  # 37
                   Atom(X, h1*(x2/2 + y2/2 + 0.25) + h2*(x2/2 - y2/2 + 0.75) + h3*(z2 + 0.5), magmom=m2, tag=3),    # 38
                   Atom(X, h1*(x2/2 - y2/2 + 0.75) + h2*(x2/2 + y2/2 + 0.75) + h3*(-z2 + 0.5), magmom=m2, tag=4),   # 39
                   Atom(X, h1*(x3/2 - y3/2 + 0.5) + h2*(x3/2 + y3/2 + 0.5) + h3*(z3 + 1), magmom=m2, tag=3),        # 40
                   Atom(X, h1*(x3/2 + y3/2 + 0.5) + h2*(x3/2 - y3/2) - h3*z3, magmom=m2, tag=1),                    # 41
                   Atom(X, h1*(-x3/2 - y3/2 + 0.25) + h2*(-x3/2 + y3/2 + 0.75) + h3*(-z3 + 0.5), magmom=m2, tag=3), # 42
                   Atom(X, h1*(-x3/2 + y3/2 + 0.75) + h2*(-x3/2 - y3/2 + 0.75) + h3*(z3 + 0.5), magmom=m2, tag=3),  # 43
                   Atom(X, h1*(-x3/2 + y3/2 + 0.5) + h2*(-x3/2 - y3/2 + 0.5) - h3*z3, magmom=m2, tag=2),            # 44
                   Atom(X, h1*(-x3/2 - y3/2 + 0.5) + h2*(-x3/2 + y3/2 + 1) + h3*(z3 + 1), magmom=m2, tag=4),        # 45
                   Atom(X, h1*(x3/2 + y3/2 + 0.25) + h2*(x3/2 - y3/2 + 0.75) + h3*(z3 + 0.5), magmom=m2, tag=3),    # 46
                   Atom(X, h1*(x3/2 - y3/2 + 0.75) + h2*(x3/2 + y3/2 + 0.75) + h3*(-z3 + 0.5), magmom=m2, tag=4)],  # 47
                  cell=[h3,h1,h2])

    if vacuum != 0:
        atoms.center(vacuum=vacuum, axis=2)

    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.tag <= fixlayers])
        atoms.set_constraint(c)

    return atoms


def CuO111(symbol=['Cu', 'O'], cell=[[4.20, 0, 0], [0, 4.10, 0], [-0.220, 0, 5.15]],
           y=0.986481, mags=[0.5, 0], vacuum=10, fixlayers=2):
    """The CuO (111) surface. The cell variable is an array of the relaxed bulk cell"""

    up = mags[0]
    down = -mags[0]
    
    a1, a2, a3 = cell

    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    a = a1 - a2
    b = a1 - a3
    c = a1 + a2 + a3

    # Given two rotation angles theta (t) and beta (b), we need to save some quantities
    a1 = abs(a[0])
    a2 = abs(a[1])
    a3 = abs(a[2])
    theta_hyp = (a1**2 + a2**2)**0.5
    cos_theta = a1 / theta_hyp
    sin_theta = a2 / theta_hyp

    b1 = abs(b[0])
    b2 = abs(b[1])
    b3 = abs(b[2])
    beta_hyp = ((b1*a2 / theta_hyp)**2 + b3**2)**0.5
    sin_beta = b3 / beta_hyp
    cos_beta = (b1*a2 / theta_hyp) / beta_hyp

    R_z = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta,  cos_theta, 0],
                    [0        ,  0        , 1]])

    R_x = np.array([[1, 0       ,  0       ],
                    [0, cos_beta, -sin_beta],
                    [0, sin_beta,  cos_beta]])

    b1 = np.dot(R_x, np.dot(R_z, a))
    b2 = np.dot(R_x, np.dot(R_z, b))
    b3 = np.dot(R_x, np.dot(R_z, c))

    y = 0.986481

    atoms = Atoms([Atom('Cu', b1/4 + b2/2 + b3/6, tag=1, magmom=down), # 1
                   Atom('O', b1*(2*y/3 - 1.0/4) + b2*(-y/3 + 1.0/2) + b3*(-y/3 + 5.0/12), tag=1), # 1
                   Atom('Cu', 3*b1/4 + b3/6, tag=1, magmom=up), # 1
                   Atom('Cu', b1/4 + b3/6, tag=1, magmom=up), # 1
                   Atom('Cu', 3*b1/4 + b2/2 + b3/6, tag=1, magmom=down), # 1
                   Atom('O', b1*(-2*y/3 + 3.0/4) + b2*(y/3 + 1.0/2) + b3*(y/3 - 1.0/12), tag=1), # 1
                   Atom('O', b1*(2*y/3 + 1.0/4) + b2*(-y/3 + 1.0/2) + b3*(-y/3 + 5.0/12), tag=1), # 1
                   Atom('O', b1*(-2*y/3 + 5.0/4) + b2*(y/3 + 1.0/2) + b3*(y/3 - 1.0/12), tag=1), # 1
                   Atom('Cu', b1/12 + b2/3 + b3/2, tag=2, magmom=up), # 2
                   Atom('Cu', 7*b1/12 + b2/3 + b3/2, tag=2, magmom=up), # 2
                   Atom('Cu', b1/12 + 5*b2/6 + b3/2, tag=2, magmom=down), # 2
                   Atom('O', b1*(-2*y/3 + 13.0/12) + b2*(y/3 - 1.0/6) + b3*(y/3 + 3.0/12), tag=2), # 2
                   Atom('O', b1*(2*y/3 - 5.0/12) + b2*(-y/3 + 5.0/6) + b3*(-y/3 + 9.0/12), tag=2), # 2
                   Atom('O', b1*(-2*y/3 + 19.0/12) + b2*(y/3 - 1.0/6) + b3*(y/3 + 3.0/12), tag=2), # 2
                   Atom('Cu', 7*b1/12 + 5*b2/6 + b3/2, tag=2, magmom=down), # 2
                   Atom('O', b1*(2*y/3 + 1.0/12) + b2*(-y/3 + 5.0/6) + b3*(-y/3 + 9.0/12), tag=2), # 2
                   Atom('O', b1*(2*y/3 - 7.0/12) + b2*(-y/3 + 7.0/6) + b3*(-y/3 + 13.0/12), tag=3), # 3
                   Atom('Cu', 5*b1/12 + 2*b2/3 + 5*b3/6, tag=3, magmom=down), # 3
                   Atom('Cu', 11*b1/12 + b2/6 + 5*b3/6, tag=3, magmom=up), # 3
                   Atom('Cu', 11*b1/12 + 2*b2/3 + 5*b3/6, tag=3, magmom=down), # 3
                   Atom('Cu', 5*b1/12 + b2/6 + 5*b3/6, tag=3, magmom=up), # 3
                   Atom('O', b1*(-2*y/3 + 17.0/12) + b2*(y/3 + 1.0/6) + b3*(y/3 + 7.0/12), tag=3), # 3
                   Atom('O', b1*(2*y/3 - 1.0/12) + b2*(-y/3 + 7.0/6) + b3*(-y/3 + 13.0/12), tag=3), # 3
                   Atom('O', b1*(-2*y/3 + 11.0/12) + b2*(y/3 + 1.0/6) + b3*(y/3 + 7.0/12), tag=3), # 3
                   Atom('Cu', b1/4 + b2/2 + b3/6 + b3, tag=4, magmom=down), # 4
                   Atom('O', b1*(2*y/3 - 1.0/4) + b2*(-y/3 + 1.0/2) + b3*(-y/3 + 5.0/12) + b3, tag=4), # 4
                   Atom('Cu', 3*b1/4 + b3/6 + b3, tag=4, magmom=up), # 4
                   Atom('Cu', b1/4 + b3/6 + b3, tag=4, magmom=up), # 4
                   Atom('Cu', 3*b1/4 + b2/2 + b3/6 + b3, tag=4, magmom=down), # 4
                   Atom('O', b1*(-2*y/3 + 3.0/4) + b2*(y/3 + 1.0/2) + b3*(y/3 - 1.0/12) + b3, tag=4), # 4
                   Atom('O', b1*(2*y/3 + 1.0/4) + b2*(-y/3 + 1.0/2) + b3*(-y/3 + 5.0/12) + b3, tag=4), # 4
                   Atom('O', b1*(-2*y/3 + 5.0/4) + b2*(y/3 + 1.0/2) + b3*(y/3 - 1.0/12) + b3, tag=4)], # 4
                  cell=(b1, b2, b3))

    if fixlayers > 0:
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.tag <= fixlayers])
        atoms.set_constraint(c)

    if vacuum != 0:
        atoms.center(vacuum=vacuum, axis=2)

    
    return atoms


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
