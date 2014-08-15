"""Helper function for bulk Atoms objects"""
from __future__ import division
from ase import Atom, Atoms
from ase.constraints import FixAtoms, FixScaled
import numpy as np

def fcc(symbol, a=None, mag=2, vol=None):
    '''Returns an FCC atoms object

    Parameters
    ----------
    symbol: str
        The atom in the cell
    a: flt
        The lattice constant.
    mag: flt
        The initial magnetic moment of the cell
    vol: flt
        This the volume of the unit cell 
    NOTE: One must provide either the volume or the lattice 
    constant.
    '''
    if a == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif a != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        a = (4 * vol) ** (1/3)
    b = a / 2
    bulk = Atoms([Atom(symbol, (0, 0, 0), magmom=mag)],
                 cell=((0, b, b),
                       (b, 0, b),
                       (b, b, 0)))
    return bulk

def bcc(symbol, a=None, mag=2, vol=None):
    '''Returns a BCC atoms object
    
    Parameters
    ----------
    symbol: str
        The atom in the cell
    a: flt
        The lattice constant.
    mag: flt
        The initial magnetic moment of the cell
    vol: flt
        This the volume of the unit cell 
    NOTE: One must provide either the volume or the lattice 
    constant.
    '''
    if a == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif a != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        a = (2 * vol) ** (1/3)
    b = a / 2
    bulk = Atoms([Atom(symbol, (0.0, 0.0, 0.0), magmom=mag)],
                 cell=[(-b, b, b),
                       (b, -b, b),
                       (b, b, -b)])
    return bulk

def rocksalt(symbols, a=None, mags=(2, 2), vol=None, afm=True):
    '''Returns a rocksalt atoms object
    
    Parameters
    ----------
    symbol: tuple
        The atoms in the unit cell. 
    a: flt
        The lattice constant.
    mags: flt
        The initial magnetic moment of the cell in the order of symbols
    vol: flt
        This the volume of the unit cell. NOTE: One must provide either
        the volume or the lattice constant.
    afm: bool
        If true, it returns the primitive cell of rocksalt with the
        anti-ferromagnetic ordering (four atoms). If false, it returns
        the two atom, primitive cell of rocksalt

    '''
    if a == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif a != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        a = (2 * vol) ** (1/3)
    b = a / 2
    if afm == True:
        bulk = Atoms([Atom(symbols[0], (0.0, 0.0, 0.0), magmom=mags[0]),
                      Atom(symbols[0], (a,a,a), magmom=-mags[0]),
                      Atom(symbols[1], (b,b,b), magmom=mags[1]),
                      Atom(symbols[1], (3*b,3*b,3*b), magmom=mags[1])],
                     cell=[(a,b,b),
                           (b,a,b),
                           (b,b,a)])
        return bulk
    else:
        bulk = Atoms([Atom(symbols[0], (0.0, 0.0, 0.0), magmom=mags[0]),
                      Atom(symbols[1], (b, b, b), magmom=mags[1])],
                     cell=((0, b, b),
                           (b, 0, b),
                           (b, b, 0)))
        return bulk

def rhombo_rocksalt(symbols, a, b, mags=(2, 2), primitive=True, afm=True):
    '''Returns a rhombohedral rocksalt atoms object

    Parameters
    ----------
    symbol: tuple
        The atoms in the unit cell
    A rhombohedral, AFM, rocksalt cell has the lattice parameters
    
    A1 = (b, a, a)
    A2 = (a, b, a)
    A3 = (a, a, b)

    Since it was difficult for me to back out expressions for a and b 
    out of the unit cell length and degrees, you will have to manually read
    these out of the CONTCAR, unfortunately.

    a: flt
        'a' parameter in above unit cell
    b: flt
        'b' parameter in above unit cell
    mags: tuple
        The initial magnetic moment of the 1st and 2nd atom
    '''

    up = mags[0]
    if afm == False:
        down = mags[0]
    else:
        down = -mags[0]

    if primitive == True:
        a1 = np.array((b, a, a))
        a2 = np.array((a, b, a))
        a3 = np.array((a, a, b))
        bulk = Atoms([Atom(symbols[0], (0.0, 0.0, 0.0), magmom=up),
                      Atom(symbols[0], 0.5*a1 + 0.5*a2 + 0.5*a3, magmom=down),
                      Atom(symbols[1], 0.25*a1 + 0.25*a2 + 0.25*a3, magmom=mags[1]),
                      Atom(symbols[1], 0.75*a1 + 0.75*a2 + 0.75*a3, magmom=mags[1])],
                     cell=[a1, a2, a3])
        return bulk

    else:
        b = b / 2.0
        c = a - b
        a1 = 2 * np.array((c, b, b))
        a2 = 2 * np.array((b, c, b))
        a3 = 2 * np.array((b, b, c))
        bulk = Atoms([Atom(symbols[0], (0, 0, 0), magmom=up),
                      Atom(symbols[0], 0.5 * a1 + 0.5 * a2, magmom=up),
                      Atom(symbols[0], 0.5 * a1 + 0.5 * a3, magmom=up),
                      Atom(symbols[0], 0.5 * a2 + 0.5 * a3, magmom=up),
                      Atom(symbols[0], 0.5 * a1, magmom=down),
                      Atom(symbols[0], 0.5 * a2, magmom=down),
                      Atom(symbols[0], 0.5 * a3, magmom=down),
                      Atom(symbols[0], 0.5 * a1 + 0.5 * a2 + 0.5 * a3, magmom=down),
                      Atom(symbols[1], 0.25 * a1 + 0.25 * a2 + 0.25 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.75 * a1 + 0.25 * a2 + 0.25 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.25 * a1 + 0.75 * a2 + 0.25 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.25 * a1 + 0.25 * a2 + 0.75 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.25 * a1 + 0.75 * a2 + 0.75 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.75 * a1 + 0.25 * a2 + 0.75 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.75 * a1 + 0.75 * a2 + 0.25 * a3, magmom=mags[1]),
                      Atom(symbols[1], 0.75 * a1 + 0.75 * a2 + 0.75 * a3, magmom=mags[1])],
                     cell=(a1, a2, a3))
        return bulk

def cubic_rocksalt(symbols, a=None, mags=(2, 2), vol=None):
    if a == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif a != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        a = (2 * vol) ** (1/3) * 4
        
    a1 = a * np.array([1, 0, 0])
    a2 = a * np.array([0, 1, 0])
    a3 = a * np.array([0, 0, 1])

    bulk = Atoms([Atom(symbols[0], (0, 0, 0), magmom=mags[0]),
                  Atom(symbols[0], 0.5*a1 + 0.5*a2, magmom=mags[0]),
                  Atom(symbols[0], 0.5*a2 + 0.5*a3, magmom=mags[0]),
                  Atom(symbols[0], 0.5*a1 + 0.5*a3, magmom=mags[0]),
                  Atom(symbols[1], 0.5*a1 + 0.5*a2 + 0.5*a3, magmom=mags[1]),
                  Atom(symbols[1], 0.5*a1, magmom=mags[1]),
                  Atom(symbols[1], 0.5*a2, magmom=mags[1]),
                  Atom(symbols[1], 0.5*a3, magmom=mags[1])],
                 cell = (a1, a2, a3))
    return bulk
                       

def read_rhombo_rocksalt(bulk):
    ''' Returns the unit cell length and degree between unit cell vectors for
    a rhombohedral, anti-ferromagnetic cell

    Parameters
    ----------
    bulk: bulk atoms object
        This will most likeley be read from the CONTCAR
    '''

    cell = bulk.get_cell()
    A1 = cell[0]
    A2 = cell[1]
    A3 = cell[2]
    B1 = A3 - A1 + A2
    B2 = A3 - A2 + A1
    B3 = A1 - A3 + A2
    lat = (B2 - B1 + B3)/2
    lat_1 = (B1 - B2 + B3)/2
    angle = 180. / np.pi * np.arccos(np.dot(lat, lat_1) /
                                     np.sqrt((lat*lat).sum()) /
                                     np.sqrt((lat_1*lat_1).sum()))
    return np.linalg.norm(lat), angle
    
def spinel(symbols, a=None, u=0.379, mags=(2, 2, 2), vol=None, inverse=True, ferri=True):
    '''Returns a spinel atoms object
    
    Parameters
    ----------
    symbol: tuple
        The atoms in the unit cell. This assums an inverse spinel, so
        the order goes like...
        symbols[0] is the cation that is on half of octahedrally bonded sites
        symbols[1] is the cation on all the tetrahedral and half the octahedral
        symbols[2] is the anion
    a: flt
        The lattice constant.
    u: flt
        The oxygen parameter 
    mags: flt
        The initial magnetic moment of the cell in the order of symbols
    vol: flt
        This the volume of the unit cell 
    NOTE: One must provide either the volume or the lattice 
    constant.
    '''
    if a == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif a != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        a = (4 * vol) ** (1/3)
    x = u - 0.125
    if ferri == True:
        magi = -1
    else:
        magi = 1
    if inverse == True:
        bulk = Atoms([Atom(symbols[1], (-a/8+a, -a/8+a, -a/8+a), magmom=magi*mags[1]),
                      Atom(symbols[1], (a/8, a/8, a/8), magmom=magi*mags[1]),
                      Atom(symbols[0], (a/2, a/2, a/2), magmom=mags[0]),
                      Atom(symbols[0], (a/2, a/4, a/4), magmom=mags[0]),
                      Atom(symbols[1], (a/4, a/2, a/4), magmom=mags[1]),
                      Atom(symbols[1], (a/4, a/4, a/2), magmom=mags[1]),
                      Atom(symbols[2], (x*a, x*a, x*a), magmom=0),
                      Atom(symbols[2], (x*a, (0.25-x)*a+a/2, (0.25-x)*a+a/2),
                           magmom=mags[2]),
                      Atom(symbols[2], ((0.25-x)*a+a/2, x*a, (0.25-x)*a+a/2),
                           magmom=mags[2]),
                      Atom(symbols[2], ((0.25-x)*a+a/2, (0.25-x)*a+a/2, x*a),
                           magmom=mags[2]),
                      Atom(symbols[2], (-x*a+a, -x*a+a, -x*a+a),magmom=mags[2]),
                      Atom(symbols[2], (-x*a+a, -(0.25-x)*a+a/2, -(0.25-x)*a+a/2),
                           magmom=mags[2]),
                      Atom(symbols[2], (-(0.25-x)*a+a/2, -x*a+a, -(0.25-x)*a+a/2)
                           ,magmom=mags[2]),
                      Atom(symbols[2], (-(0.25-x)*a+a/2, -(0.25-x)*a+a/2, -x*a+a),
                           magmom=mags[2])],
                     cell=[(0.0, a/2, a/2),
                           (a/2, 0.0 ,a/2),
                           (a/2, a/2, 0.0)])
    else:
        bulk = Atoms([Atom(symbols[0], (-a/8+a, -a/8+a, -a/8+a), magmom=magi*mags[0]),
                      Atom(symbols[0], (a/8, a/8, a/8), magmom=magi*mags[0]),
                      Atom(symbols[1], (a/2, a/2, a/2), magmom=mags[1]),
                      Atom(symbols[1], (a/2, a/4, a/4), magmom=mags[1]),
                      Atom(symbols[1], (a/4, a/2, a/4), magmom=mags[1]),
                      Atom(symbols[1], (a/4, a/4, a/2), magmom=mags[1]),
                      Atom(symbols[2], (x*a, x*a, x*a), magmom=0),
                      Atom(symbols[2], (x*a, (0.25-x)*a+a/2, (0.25-x)*a+a/2),
                           magmom=mags[2]),
                      Atom(symbols[2], ((0.25-x)*a+a/2, x*a, (0.25-x)*a+a/2),
                           magmom=mags[2]),
                      Atom(symbols[2], ((0.25-x)*a+a/2, (0.25-x)*a+a/2, x*a),
                           magmom=mags[2]),
                      Atom(symbols[2], (-x*a+a, -x*a+a, -x*a+a),magmom=mags[2]),
                      Atom(symbols[2], (-x*a+a, -(0.25-x)*a+a/2, -(0.25-x)*a+a/2),
                           magmom=mags[2]),
                      Atom(symbols[2], (-(0.25-x)*a+a/2, -x*a+a, -(0.25-x)*a+a/2)
                           ,magmom=mags[2]),
                      Atom(symbols[2], (-(0.25-x)*a+a/2, -(0.25-x)*a+a/2, -x*a+a),
                           magmom=mags[2])],
                     cell=[(0.0, a/2, a/2),
                           (a/2, 0.0 ,a/2),
                           (a/2, a/2, 0.0)])
        
    pos = bulk.get_scaled_positions()
    bulk.set_scaled_positions(pos %[1, 1, 1])
    c = FixAtoms(indices=[atom.index for atom in bulk if atom.symbol != symbols[2]])
    bulk.set_constraint(c)
    return bulk

def Co3O4(symbols, a=None, u=0.379, mags=(2, 2, 2), vol=None):
    '''Returns a spinel atoms object
    
    Parameters
    ----------
    symbol: tuple
        The atoms in the unit cell. This assums an inverse spinel, so
        the order goes like...
        symbols[0] is the cation that is on half of octahedrally bonded sites
        symbols[1] is the cation on all the tetrahedral and half the octahedral
        symbols[2] is the anion
    a: flt
        The lattice constant.
    u: flt
        The oxygen parameter 
    mags: flt
        The initial magnetic moment of the cell in the order of symbols
    vol: flt
        This the volume of the unit cell 
    NOTE: One must provide either the volume or the lattice 
    constant.
    '''
    if a == None and vol == None:
        raise TypeError('Please provide either a lattice or volume')
    elif a != None and vol != None:
        raise TypeError('Cannot provide both lattice constant and volume')
    elif vol != None:
        a = (4 * vol) ** (1/3)
    x = u - 0.125

    bulk = Atoms([Atom(symbols[1], (-a/8+a, -a/8+a, -a/8+a), magmom=-mags[0]),
                  Atom(symbols[1], (a/8, a/8, a/8), magmom=mags[0]),
                  Atom(symbols[0], (a/2, a/2, a/2), magmom=0),
                  Atom(symbols[0], (a/2, a/4, a/4), magmom=0),
                  Atom(symbols[1], (a/4, a/2, a/4), magmom=0),
                  Atom(symbols[1], (a/4, a/4, a/2), magmom=0),
                  Atom(symbols[2], (x*a, x*a, x*a), magmom=mags[2]),
                  Atom(symbols[2], (x*a, (0.25-x)*a+a/2, (0.25-x)*a+a/2),
                       magmom=mags[2]),
                  Atom(symbols[2], ((0.25-x)*a+a/2, x*a, (0.25-x)*a+a/2),
                       magmom=mags[2]),
                  Atom(symbols[2], ((0.25-x)*a+a/2, (0.25-x)*a+a/2, x*a),
                       magmom=mags[2]),
                  Atom(symbols[2], (-x*a+a, -x*a+a, -x*a+a),magmom=mags[2]),
                  Atom(symbols[2], (-x*a+a, -(0.25-x)*a+a/2, -(0.25-x)*a+a/2),
                       magmom=mags[2]),
                  Atom(symbols[2], (-(0.25-x)*a+a/2, -x*a+a, -(0.25-x)*a+a/2)
                       ,magmom=mags[2]),
                  Atom(symbols[2], (-(0.25-x)*a+a/2, -(0.25-x)*a+a/2, -x*a+a),
                       magmom=mags[2])],
                 cell=[(0.0, a/2, a/2),
                       (a/2, 0.0 ,a/2),
                       (a/2, a/2, 0.0)])
        
    pos = bulk.get_scaled_positions()
    bulk.set_scaled_positions(pos %[1, 1, 1])
    c = FixAtoms(indices=[atom.index for atom in bulk if atom.symbol != symbols[2]])
    bulk.set_constraint(c)

    return bulk
    
def corundum(symbols, chex=None, c_over_a=2.766, z=0.144, x=0.309, mags=(2, 2),
             vol=None):
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
    bulk = Atoms([Atom(symbols[0], z*(a1 + a2 + a3), magmom=-mags[0], tag=1),
                  Atom(symbols[0], -z*(a1 + a2 + a3), magmom=-mags[0], tag=1),
                  Atom(symbols[0], (0.5 + z)*(a1 + a2 + a3), magmom=mags[0], tag=1),
                  Atom(symbols[0], (-0.5 - z)*(a1 + a2 + a3), magmom=mags[0], tag=1),
                  Atom(symbols[1], x*a1 + (0.5 - x)*a2 + 0.25*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], -x*a1 - (0.5 - x)*a2 - 0.25*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], (0.5 - x)*a1 + 0.25*a2 + x*a3, magmom=mags[1], tag=3),
                  Atom(symbols[1], -(0.5 - x)*a1 - 0.25*a2 - x*a3, magmom=mags[1], tag=3),
                  Atom(symbols[1], 0.25*a1 + x*a2 + (0.5 - x)*a3, magmom=mags[1], tag=4),
                  Atom(symbols[1], -0.25*a1 - x*a2 - (0.5 - x)*a3, magmom=mags[1], tag=4)],
                 cell=(a1,a2,a3))
    pos = bulk.get_scaled_positions()
    bulk.set_scaled_positions(pos %[1,1,1])
    c = []
    for atom in bulk:
        if atom.tag == 1:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0,0,0)))
        elif atom.tag == 2:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0,0,1)))
        elif atom.tag == 3:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0,1,0)))
        elif atom.tag == 4:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(1,0,0)))
        else:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0,0,0)))
    bulk.set_constraint(c)

    return bulk

def A2B5(symbols, a=11.512, b=3.564, c=4.368, z1=0.001, x2=0.8564, z2=0.531,
         x3=0.6811, z3=0.003, x4=0.8512, z4=0.892, mags=(2, 0)):
    a1 = np.array((a, 0, 0))
    a2 = np.array((0, b, 0))
    a3 = np.array((0, 0, c))
    bulk = Atoms([Atom(symbols[1], z1*a3, magmom=mags[1], tag=1),
                  Atom(symbols[1], 0.5*a1 + 0.5*a2 - z1*a3, magmom=mags[1], tag=1),
                  Atom(symbols[1], x2*a1 +  z2*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], -x2*a1 + z2*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], (-x2 + 0.5)*a1 + 0.5*a2 - z2*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], (x2 + 0.5)*a1 + 0.5*a2 - z2*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], x3*a1 +  z3*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], -x3*a1 + z3*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], (-x3 + 0.5)*a1 + 0.5*a2 - z3*a3, magmom=mags[1], tag=2),
                  Atom(symbols[1], (x3 + 0.5)*a1 + 0.5*a2 - z3*a3, magmom=mags[1], tag=2),
                  Atom(symbols[0], x4*a1 +  z4*a3, magmom=mags[0], tag=2),
                  Atom(symbols[0], -x4*a1 + z4*a3, magmom=mags[0], tag=2),
                  Atom(symbols[0], (-x4 + 0.5)*a1 + 0.5*a2 - z4*a3, magmom=mags[0], tag=2),
                  Atom(symbols[0], (x4 + 0.5)*a1 + 0.5*a2 - z4*a3, magmom=mags[0], tag=2)],
                 cell=(a1, a2, a3))
    pos = bulk.get_scaled_positions()
    bulk.set_scaled_positions(pos %[1, 1, 1])
    c = []
    for atom in bulk:
        if atom.tag == 1:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(1,1,0)))
        elif atom.tag == 2:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0,1,0)))
        else:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0,0,0)))
    bulk.set_constraint(c)
    return bulk

def Mn2O3(pos=None, a=9.407, b=9.447, c=9.366, mags=(2, 0)):
    '''The Mn2O3 has an orthorhombic space group with 8 unique positions. If pos==None,
    we will start from the trial ones detailed in Norrestam (1967)'''
    up = mags[0]
    down = -mags[0]
    if pos == None:
        pos = (( 0.28479,  0.25253, -0.0059),
               ( 0.00462,  0.28507,  0.24564),
               ( 0.25301,  0.00130,  0.28533),
               ( 0.13299, -0.08466,  0.15030),
               ( 0.14435,  0.12989, -0.08507),
               (-0.08038,  0.14693,  0.12412),
               (-0.37447,  0.41757, -0.35569),
               (-0.35081, -0.37238,  0.41947),
               ( 0.41306, -0.35285, -0.36571))

    a1 = np.array((9.407, 0, 0))
    a2 = np.array((0, 9.447, 0))
    a3 = np.array((0, 0, 9.366))

    bulk = Atoms([Atom('Mn', (0, 0, 0), magmom=up),
                  Atom('Mn', 0.5 * a1 + 0*a2     + 0.5 * a3, magmom=down),
                  Atom('Mn', 0   * a1 + 0.5 * a2 + 0.5 * a3, magmom=up),
                  Atom('Mn', 0.5 * a1 + 0.5 * a2 + 0   * a3, magmom=down),
                  Atom('Mn', 0   * a1 + 0   * a2 + 0.5 * a3, magmom=down),
                  Atom('Mn', 0.5 * a1 + 0   * a2 + 0   * a3, magmom=up),
                  Atom('Mn', 0   * a1 + 0.5 * a2 + 0   * a3, magmom=down),
                  Atom('Mn', 0.5 * a1 + 0.5 * a2 + 0.5 * a3, magmom=up),
                  Atom('Mn', pos[0][0]*a1 + pos[0][1]*a2 + pos[0][2]*a3, magmom=down),
                  Atom('Mn', (-pos[0][0]+0.5)*a1 - pos[0][1]*a2 + (pos[0][2]+0.5)*a3, magmom=up),
                  Atom('Mn', -pos[0][0]*a1 + (pos[0][1]+0.5)*a2 + (-pos[0][2]+0.5)*a3, magmom=up),
                  Atom('Mn', (pos[0][0]+0.5)*a1 + (-pos[0][1]+0.5)*a2 - pos[0][2]*a3, magmom=up),
                  Atom('Mn', -pos[0][0]*a1 - pos[0][1]*a2 - pos[0][2]*a3, magmom=down),
                  Atom('Mn', (pos[0][0]+0.5)*a1 + pos[0][1]*a2 + (-pos[0][2]+0.5)*a3, magmom=down),
                  Atom('Mn', pos[0][0]*a1 + (-pos[0][1]+0.5)*a2 + (pos[0][2]+0.5)*a3, magmom=down),
                  Atom('Mn', (-pos[0][0]+0.5)*a1 + (pos[0][1]+0.5)*a2 + pos[0][2]*a3, magmom=up),
                  Atom('Mn', pos[1][0]*a1 + pos[1][1]*a2 + pos[1][2]*a3, magmom=down),
                  Atom('Mn', (-pos[1][0]+0.5)*a1 - pos[1][1]*a2 + (pos[1][2]+0.5)*a3, magmom=up),
                  Atom('Mn', -pos[1][0]*a1 + (pos[1][1]+0.5)*a2 + (-pos[1][2]+0.5)*a3, magmom=down),
                  Atom('Mn', (pos[1][0]+0.5)*a1 + (-pos[1][1]+0.5)*a2 - pos[1][2]*a3, magmom=up),
                  Atom('Mn', -pos[1][0]*a1 - pos[1][1]*a2 - pos[1][2]*a3, magmom=down),
                  Atom('Mn', (pos[1][0]+0.5)*a1 + pos[1][1]*a2 + (-pos[1][2]+0.5)*a3, magmom=up),
                  Atom('Mn', pos[1][0]*a1 + (-pos[1][1]+0.5)*a2 + (pos[1][2]+0.5)*a3, magmom=down),
                  Atom('Mn', (-pos[1][0]+0.5)*a1 + (pos[1][1]+0.5)*a2 + pos[1][2]*a3, magmom=up),
                  Atom('Mn', pos[2][0]*a1 + pos[2][1]*a2 + pos[2][2]*a3, magmom=up),
                  Atom('Mn', (-pos[2][0]+0.5)*a1 - pos[2][1]*a2 + (pos[2][2]+0.5)*a3, magmom=up),
                  Atom('Mn', -pos[2][0]*a1 + (pos[2][1]+0.5)*a2 + (-pos[2][2]+0.5)*a3, magmom=up),
                  Atom('Mn', (pos[2][0]+0.5)*a1 + (-pos[2][1]+0.5)*a2 - pos[2][2]*a3, magmom=up),
                  Atom('Mn', -pos[2][0]*a1 - pos[2][1]*a2 - pos[2][2]*a3, magmom=down),
                  Atom('Mn', (pos[2][0]+0.5)*a1 + pos[2][1]*a2 + (-pos[2][2]+0.5)*a3, magmom=down),
                  Atom('Mn', pos[2][0]*a1 + (-pos[2][1]+0.5)*a2 + (pos[2][2]+0.5)*a3, magmom=down),
                  Atom('Mn', (-pos[2][0]+0.5)*a1 + (pos[2][1]+0.5)*a2 + pos[2][2]*a3, magmom=down),
                  Atom('O', pos[3][0]*a1 + pos[3][1]*a2 + pos[3][2]*a3, magmom=mags[1]),
                  Atom('O', (-pos[3][0]+0.5)*a1 - pos[3][1]*a2 + (pos[3][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', -pos[3][0]*a1 + (pos[3][1]+0.5)*a2 + (-pos[3][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (pos[3][0]+0.5)*a1 + (-pos[3][1]+0.5)*a2 - pos[3][2]*a3, magmom=mags[1]),
                  Atom('O', -pos[3][0]*a1 - pos[3][1]*a2 - pos[3][2]*a3, magmom=mags[1]),
                  Atom('O', (pos[3][0]+0.5)*a1 + pos[3][1]*a2 + (-pos[3][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', pos[3][0]*a1 + (-pos[3][1]+0.5)*a2 + (pos[3][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (-pos[3][0]+0.5)*a1 + (pos[3][1]+0.5)*a2 + pos[3][2]*a3, magmom=mags[1]),
                  Atom('O', pos[4][0]*a1 + pos[4][1]*a2 + pos[4][2]*a3, magmom=mags[1]),
                  Atom('O', (-pos[4][0]+0.5)*a1 - pos[4][1]*a2 + (pos[4][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', -pos[4][0]*a1 + (pos[4][1]+0.5)*a2 + (-pos[4][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (pos[4][0]+0.5)*a1 + (-pos[4][1]+0.5)*a2 - pos[4][2]*a3, magmom=mags[1]),
                  Atom('O', -pos[4][0]*a1 - pos[4][1]*a2 - pos[4][2]*a3, magmom=mags[1]),
                  Atom('O', (pos[4][0]+0.5)*a1 + pos[4][1]*a2 + (-pos[4][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', pos[4][0]*a1 + (-pos[4][1]+0.5)*a2 + (pos[4][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (-pos[4][0]+0.5)*a1 + (pos[4][1]+0.5)*a2 + pos[4][2]*a3, magmom=mags[1]),
                  Atom('O', pos[5][0]*a1 + pos[5][1]*a2 + pos[5][2]*a3, magmom=mags[1]),
                  Atom('O', (-pos[5][0]+0.5)*a1 - pos[5][1]*a2 + (pos[5][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', -pos[5][0]*a1 + (pos[5][1]+0.5)*a2 + (-pos[5][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (pos[5][0]+0.5)*a1 + (-pos[5][1]+0.5)*a2 - pos[5][2]*a3, magmom=mags[1]),
                  Atom('O', -pos[5][0]*a1 - pos[5][1]*a2 - pos[5][2]*a3, magmom=mags[1]),
                  Atom('O', (pos[5][0]+0.5)*a1 + pos[5][1]*a2 + (-pos[5][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', pos[5][0]*a1 + (-pos[5][1]+0.5)*a2 + (pos[5][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (-pos[5][0]+0.5)*a1 + (pos[5][1]+0.5)*a2 + pos[5][2]*a3, magmom=mags[1]),
                  Atom('O', pos[6][0]*a1 + pos[6][1]*a2 + pos[6][2]*a3, magmom=mags[1]),
                  Atom('O', (-pos[6][0]+0.5)*a1 - pos[6][1]*a2 + (pos[6][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', -pos[6][0]*a1 + (pos[6][1]+0.5)*a2 + (-pos[6][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (pos[6][0]+0.5)*a1 + (-pos[6][1]+0.5)*a2 - pos[6][2]*a3, magmom=mags[1]),
                  Atom('O', -pos[6][0]*a1 - pos[6][1]*a2 - pos[6][2]*a3, magmom=mags[1]),
                  Atom('O', (pos[6][0]+0.5)*a1 + pos[6][1]*a2 + (-pos[6][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', pos[6][0]*a1 + (-pos[6][1]+0.5)*a2 + (pos[6][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (-pos[6][0]+0.5)*a1 + (pos[6][1]+0.5)*a2 + pos[6][2]*a3, magmom=mags[1]),
                  Atom('O', pos[7][0]*a1 + pos[7][1]*a2 + pos[7][2]*a3, magmom=mags[1]),
                  Atom('O', (-pos[7][0]+0.5)*a1 - pos[7][1]*a2 + (pos[7][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', -pos[7][0]*a1 + (pos[7][1]+0.5)*a2 + (-pos[7][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (pos[7][0]+0.5)*a1 + (-pos[7][1]+0.5)*a2 - pos[7][2]*a3, magmom=mags[1]),
                  Atom('O', -pos[7][0]*a1 - pos[7][1]*a2 - pos[7][2]*a3, magmom=mags[1]),
                  Atom('O', (pos[7][0]+0.5)*a1 + pos[7][1]*a2 + (-pos[7][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', pos[7][0]*a1 + (-pos[7][1]+0.5)*a2 + (pos[7][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (-pos[7][0]+0.5)*a1 + (pos[7][1]+0.5)*a2 + pos[7][2]*a3, magmom=mags[1]),
                  Atom('O', pos[8][0]*a1 + pos[8][1]*a2 + pos[8][2]*a3, magmom=mags[1]),
                  Atom('O', (-pos[8][0]+0.5)*a1 - pos[8][1]*a2 + (pos[8][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', -pos[8][0]*a1 + (pos[8][1]+0.5)*a2 + (-pos[8][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (pos[8][0]+0.5)*a1 + (-pos[8][1]+0.5)*a2 - pos[8][2]*a3, magmom=mags[1]),
                  Atom('O', -pos[8][0]*a1 - pos[8][1]*a2 - pos[8][2]*a3, magmom=mags[1]),
                  Atom('O', (pos[8][0]+0.5)*a1 + pos[8][1]*a2 + (-pos[8][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', pos[8][0]*a1 + (-pos[8][1]+0.5)*a2 + (pos[8][2]+0.5)*a3, magmom=mags[1]),
                  Atom('O', (-pos[8][0]+0.5)*a1 + (pos[8][1]+0.5)*a2 + pos[8][2]*a3, magmom=mags[1])],
                 cell=(a1, a2, a3))
    pos = bulk.get_scaled_positions()
    bulk.set_scaled_positions(pos % [1, 1, 1])
    return bulk

def Mn3O4(y=0.227, z=0.383, a=5.763, c=9.456, mags=[2,0]):
    '''This is the crystal structure of M3O4. It is a tetragonallyd distored spinel.
    See space group 141.'''

    up = mags[0]
    down = -mags[0]
    
    a1 = np.array((a, 0, 0))
    a2 = np.array((0, a, 0))
    a3 = np.array((0, 0, c))

    bulk = Atoms([Atom('Mn', (0, 0, 0), magmom=down, tag=1),
                  Atom('Mn', 0.5*a2 + 0.25*a3, magmom=down, tag=1),
                  Atom('Mn', 0.5*a1 + 0.5*a2 + 0.5*a3, magmom=down, tag=1),
                  Atom('Mn', 0.5*a1 + 0.75*a3, magmom=down, tag=1),

                  Atom('Mn', 0.25*a2 + 5./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.5*a1 + 0.25*a2 + 1./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.75*a1 + 0.5*a2 + 7./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.75*a1 + 3./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.5*a1 + 0.75*a2 + 1./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.75*a2 + 5./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.25*a1 + 3./8*a3, magmom=up, tag=1),
                  Atom('Mn', 0.25*a1 + 0.5*a2 + 7./8*a3, magmom=up, tag=1),

                  Atom('O', (0)*a1 + (y)*a2 + (z)*a3, magmom=mags[1], tag=2),
                  Atom('O', (0.5)*a1 + (-y+0.5)*a2 + (z+0.5)*a3, magmom=mags[1], tag=2),
                  Atom('O', (-y)*a1 + (0.5)*a2 + (z+0.25)*a3, magmom=mags[1], tag=3),
                  Atom('O', (y+0.5)*a1 + (0)*a2 + (z+0.75)*a3, magmom=mags[1], tag=3),
                  Atom('O', (0.5)*a1 + (y)*a2 + (-z+0.75)*a3, magmom=mags[1], tag=2),
                  Atom('O', (0)*a1 + (-y+0.5)*a2 + (-z+0.25)*a3, magmom=mags[1], tag=2),
                  Atom('O', (y+0.5)*a1 + (0.5)*a2 + (-z+0.5)*a3, magmom=mags[1], tag=3),
                  Atom('O', (-y)*a1 + (0)*a2 + (-z)*a3, magmom=mags[1], tag=3),

                  Atom('O', (0.5)*a1 + (y + 0.5)*a2 + (z + 0.5)*a3, magmom=mags[1], tag=2),
                  Atom('O', (0)*a1 + (-y)*a2 + (z)*a3, magmom=mags[1], tag=2),
                  Atom('O', (-y + 0.5)*a1 + (0)*a2 + (z + 0.75)*a3, magmom=mags[1], tag=3),
                  Atom('O', (y)*a1 + (0.5)*a2 + (z + 0.25)*a3, magmom=mags[1], tag=3),
                  Atom('O', (0)*a1 + (y + 0.5)*a2 + (-z + 0.25)*a3, magmom=mags[1], tag=2),
                  Atom('O', (0.5)*a1 + (-y)*a2 + (-z + 0.75)*a3, magmom=mags[1], tag=2),
                  Atom('O', (y)*a1 + (0)*a2 + (-z)*a3, magmom=mags[1], tag=3),
                  Atom('O', (-y + 0.5)*a1 + (0.5)*a2 + (-z + 0.5)*a3, magmom=mags[1], tag=3)],
                 cell=(a1, a2, a3))

    pos = bulk.get_scaled_positions()
    bulk.set_scaled_positions(pos %[1, 1, 1])

    c = []
    for atom in bulk:
        if atom.tag == 1:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(1, 1, 1)))
        elif atom.tag == 2:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(1, 0, 0)))
        elif atom.tag ==3:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0, 1, 0)))
        else:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0, 0, 0)))
    bulk.set_constraint(c)
    return bulk

def rutile(symbols, a=4.59, c=2.958, u=0.305, mags=[2, 0], afm=False):
    '''
    create a rutile structure from the lattice vectors at
    http://cst-www.nrl.navy.mil/lattice/struk/c4.html
    ref Phys Rev B 224112
    spacegroup: 136 P4_2/mnm
    '''
    B = symbols[0]
    X = symbols[1]
    a1 = a*np.array([1.0, 0.0, 0.0])
    a2 = a*np.array([0.0, 1.0, 0.0])
    a3 = c*np.array([0.0, 0.0, 1.0])
    if afm == False:
        up = mags[0]
        down = mags[0]
    else:
        up = mags[0]
        down = -mags[0]
    
    bulk = Atoms([Atom(B, [0., 0., 0.], magmom=up, tag=1),
                   Atom(B, 0.5*a1 + 0.5*a2 + 0.5*a3, magmom=down, tag=1),
                   Atom(X,  u*a1 + u*a2, magmom=mags[1], tag=2),
                   Atom(X, -u*a1 - u*a2, magmom=mags[1], tag=2),
                   Atom(X, (0.5+u)*a1 + (0.5-u)*a2 + 0.5*a3, magmom=mags[1], tag=2),
                   Atom(X, (0.5-u)*a1 + (0.5+u)*a2 + 0.5*a3, magmom=mags[1], tag=2)],
                  cell=[a1, a2, a3])
    c = []
    for atom in bulk:
        if atom.tag == 1:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(1, 1, 1)))
        elif atom.tag == 2:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0, 0, 1)))
        else:
            c.append(FixScaled(cell=bulk.cell, a=atom.index, mask=(0, 0, 0)))
    bulk.set_constraint(c)
    return bulk

def I_43m(symbol='Mn', a=8.9125, xc=0.317, xg1=0.356, zg1=0.042, xg2=0.089, zg2=0.278, mag=0.5):
    '''This space group is exlusively needed to construct the bulk Manganese unit cell'''
    a1 = np.array((a, 0, 0))
    a2 = np.array((0, a, 0))
    a3 = np.array((0, 0, a))
    bulk = Atoms([Atom(symbol, (0., 0., 0.), magmom=mag),
                  Atom(symbol, a1/2. + a2/2. + a3/2., magmom=mag),
                  Atom(symbol, xc*a1 + xc*a2 + xc*a3, magmom=mag),
                  Atom(symbol, -xc*a1 - xc*a2 + xc*a3, magmom=mag),
                  Atom(symbol, -xc*a1 + xc*a2 - xc*a3, magmom=mag),
                  Atom(symbol, xc*a1 - xc*a2 - xc*a3, magmom=mag),
                  Atom(symbol, (xc + 0.5)*a1 + (xc + 0.5)*a2 + (xc + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xc + 0.5)*a1 + (-xc + 0.5)*a2 + (xc + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xc + 0.5)*a1 + (xc + 0.5)*a2 + (-xc + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xc + 0.5)*a1 + (-xc + 0.5)*a2 + (-xc + 0.5)*a3, magmom=mag),
                  Atom(symbol, xg1*a1 + xg1*a2 + zg1*a3, magmom=mag),
                  Atom(symbol, -xg1*a1 + -xg1*a2 + zg1*a3, magmom=mag),
                  Atom(symbol, -xg1*a1 + xg1*a2 + -zg1*a3, magmom=mag),
                  Atom(symbol, xg1*a1 + -xg1*a2 + -zg1*a3, magmom=mag),
                  Atom(symbol, zg1*a1 + xg1*a2 + xg1*a3, magmom=mag),
                  Atom(symbol, zg1*a1 + -xg1*a2 + -xg1*a3, magmom=mag),
                  Atom(symbol, -zg1*a1 + -xg1*a2 + xg1*a3, magmom=mag),
                  Atom(symbol, -zg1*a1 + xg1*a2 + -xg1*a3, magmom=mag),
                  Atom(symbol, xg1*a1 + zg1*a2 + xg1*a3, magmom=mag),
                  Atom(symbol, -xg1*a1 + zg1*a2 + -xg1*a3, magmom=mag),
                  Atom(symbol, xg1*a1 + -zg1*a2 + -xg1*a3, magmom=mag),
                  Atom(symbol, -xg1*a1 + -zg1*a2 + xg1*a3, magmom=mag),
                  Atom(symbol, (xg1 + 0.5)*a1 + (xg1 + 0.5)*a2 + (zg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg1 + 0.5)*a1 + (-xg1 + 0.5)*a2 + (zg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg1 + 0.5)*a1 + (xg1 + 0.5)*a2 + (-zg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xg1 + 0.5)*a1 + (-xg1 + 0.5)*a2 + (-zg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (zg1 + 0.5)*a1 + (xg1 + 0.5)*a2 + (xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (zg1 + 0.5)*a1 + (-xg1 + 0.5)*a2 + (-xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-zg1 + 0.5)*a1 + (-xg1 + 0.5)*a2 + (xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-zg1 + 0.5)*a1 + (xg1 + 0.5)*a2 + (-xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xg1 + 0.5)*a1 + (zg1 + 0.5)*a2 + (xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg1 + 0.5)*a1 + (zg1 + 0.5)*a2 + (-xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xg1 + 0.5)*a1 + (-zg1 + 0.5)*a2 + (-xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg1 + 0.5)*a1 + (-zg1 + 0.5)*a2 + (xg1 + 0.5)*a3, magmom=mag),
                  Atom(symbol, xg2*a1 + xg2*a2 + zg2*a3, magmom=mag),
                  Atom(symbol, -xg2*a1 + -xg2*a2 + zg2*a3, magmom=mag),
                  Atom(symbol, -xg2*a1 + xg2*a2 + -zg2*a3, magmom=mag),
                  Atom(symbol, xg2*a1 + -xg2*a2 + -zg2*a3, magmom=mag),
                  Atom(symbol, zg2*a1 + xg2*a2 + xg2*a3, magmom=mag),
                  Atom(symbol, zg2*a1 + -xg2*a2 + -xg2*a3, magmom=mag),
                  Atom(symbol, -zg2*a1 + -xg2*a2 + xg2*a3, magmom=mag),
                  Atom(symbol, -zg2*a1 + xg2*a2 + -xg2*a3, magmom=mag),
                  Atom(symbol, xg2*a1 + zg2*a2 + xg2*a3, magmom=mag),
                  Atom(symbol, -xg2*a1 + zg2*a2 + -xg2*a3, magmom=mag),
                  Atom(symbol, xg2*a1 + -zg2*a2 + -xg2*a3, magmom=mag),
                  Atom(symbol, -xg2*a1 + -zg2*a2 + xg2*a3, magmom=mag),
                  Atom(symbol, (xg2 + 0.5)*a1 + (xg2 + 0.5)*a2 + (zg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg2 + 0.5)*a1 + (-xg2 + 0.5)*a2 + (zg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg2 + 0.5)*a1 + (xg2 + 0.5)*a2 + (-zg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xg2 + 0.5)*a1 + (-xg2 + 0.5)*a2 + (-zg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (zg2 + 0.5)*a1 + (xg2 + 0.5)*a2 + (xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (zg2 + 0.5)*a1 + (-xg2 + 0.5)*a2 + (-xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-zg2 + 0.5)*a1 + (-xg2 + 0.5)*a2 + (xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-zg2 + 0.5)*a1 + (xg2 + 0.5)*a2 + (-xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xg2 + 0.5)*a1 + (zg2 + 0.5)*a2 + (xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg2 + 0.5)*a1 + (zg2 + 0.5)*a2 + (-xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (xg2 + 0.5)*a1 + (-zg2 + 0.5)*a2 + (-xg2 + 0.5)*a3, magmom=mag),
                  Atom(symbol, (-xg2 + 0.5)*a1 + (-zg2 + 0.5)*a2 + (xg2 + 0.5)*a3, magmom=mag)],
                 cell=(a1, a2, a3))
    return bulk

def MO2H2(symbols, a=3.184, c=4.617, u=0.43, z=0.22, mags=(0.5, 0), afm=False):
    '''Returns a brucite crystal structure. This is a common crystal structure
    of Ni(OH)2
    
    Parameters
    ----------
    symbols: tuple
        The atoms of the unit cell. The hydrogen atom is implied.
    a: flt
        The a lattice parameter of the unit cell
    c: flt
        The c lattice parameter of the unit cell
    u: flt
        The hydrogen position coordinate 
    z: flt
        The oxygen position coordinate
    mags: tuple
        The initial magnetic moment of the metal and oxygen ion
    '''
    
    a1 = np.array((0.5*a, -0.5*3**0.5*a, 0))
    a2 = np.array((0.5*a, 0.5*3**0.5*a, 0))
    a3 = np.array((0, 0, c))

    if afm == True:
        atoms = Atoms([Atom(symbols[0], np.array((0, 0, 0)), magmom=mags[0]),
                       Atom('H', 1./3*a1 + 2./3*a2 + u*a3, magmom=0),
                       Atom('H', 2./3*a1 + 1./3*a2 + (1-u)*a3, magmom=0),
                       Atom(symbols[1], 1./3*a1 + 2./3*a2 + z*a3, magmom=mags[1]),
                       Atom(symbols[1], 2./3*a1 + 1./3*a2 + (1-z)*a3, magmom=mags[1]),
                       Atom(symbols[0], a3, magmom=-mags[0]),
                       Atom('H', 1./3*a1 + 2./3*a2 + (1+u)*a3, magmom=0),
                       Atom('H', 2./3*a1 + 1./3*a2 + (2-u)*a3, magmom=0),
                       Atom(symbols[1], 1./3*a1 + 2./3*a2 + (1+z)*a3, magmom=mags[1]),
                       Atom(symbols[1], 2./3*a1 + 1./3*a2 + (2-z)*a3, magmom=mags[1]),],
                      cell=(a1, a2, 2*a3))
    else:
        atoms = Atoms([Atom(symbols[0], np.array((0, 0, 0)), magmom=mags[0]),
                       Atom('H', 1./3*a1 + 2./3*a2 + u*a3, magmom=0),
                       Atom('H', 2./3*a1 + 1./3*a2 + (1-u)*a3, magmom=0),
                       Atom(symbols[1], 1./3*a1 + 2./3*a2 + z*a3, magmom=mags[1]),
                       Atom(symbols[1], 2./3*a1 + 1./3*a2 + (1-z)*a3, magmom=mags[1])],
                      cell=(a1, a2, a3))
    
    c = []
    for atom in atoms:
        if atom.symbol == symbols[0]:
            c.append(FixScaled(cell=atoms.cell, a=atom.index, mask=(1, 1, 1)))
        elif atom.symbol in (symbols[1], 'H'):
            c.append(FixScaled(cell=atoms.cell, a=atom.index, mask=(1, 1, 0)))
        else:
            c.append(FixScaled(cell=atoms.cell, a=atom.index, mask=(0, 0, 0)))
    atoms.set_constraint(c)
    return atoms

def MOOH(symbols, a=3.184, c=4.617, z=0.22, mags=(0.5, 0), afm=False):
    '''Returns a brucite crystal structure. This is a common crystal structure
    of Ni(OH)2
    
    Parameters
    ----------
    symbols: tuple
        The atoms of the unit cell. The hydrogen atom is implied.
    a: flt
        The a lattice parameter of the unit cell
    c: flt
        The c lattice parameter of the unit cell
    z: flt
        The oxygen position coordinate
    mags: tuple
        The initial magnetic moment of the metal and oxygen ion
    '''
    
    a1 = np.array((0.5*a, -0.5*3**0.5*a, 0))
    a2 = np.array((0.5*a, 0.5*3**0.5*a, 0))
    a3 = np.array((0, 0, c))
    atoms = Atoms([Atom('H', 2./3*a1 + 1./3*a2, magmom=0),
                   Atom(symbols[1], 2./3*a1 + 1./3*a2 + (0.5-z)*a3, magmom=mags[1]),               
                   Atom(symbols[0], 0.5*a3, magmom=mags[0]),
                   Atom(symbols[1], 1./3*a1 + 2./3*a2 + (0.5+z)*a3, magmom=mags[1]),
                   
                   Atom('H', 1./3*a1 + 2./3*a2 + a3, magmom=0),
                   Atom(symbols[1], 1./3*a1 + 2./3*a2 + (1.5-z)*a3, magmom=mags[1]),               
                   Atom(symbols[0], 2./3*a1 + 1./3*a2 + 1.5*a3, magmom=mags[0]),
                   Atom(symbols[1], (1.5+z)*a3, magmom=mags[1]),
                   
                   Atom('H', 2*a3, magmom=0),
                   Atom(symbols[1], (2.5-z)*a3, magmom=mags[1]),
                   Atom(symbols[0], 1./3*a1 + 2./3*a2 + 2.5*a3, magmom=mags[0]),
                   Atom(symbols[1], 2./3*a1 + 1./3*a2 + (2.5+z)*a3, magmom=mags[1])],
                  cell=(a1, a2, 3*a3))

    
    c = []
    for atom in atoms:
        if atom.symbol in (symbols[0], 'H'):
            c.append(FixScaled(cell=atoms.cell, a=atom.index, mask=(1, 1, 1)))
        elif atom.symbol == symbols[1]:
            c.append(FixScaled(cell=atoms.cell, a=atom.index, mask=(1, 1, 0)))
        else:
            c.append(FixScaled(cell=atoms.cell, a=atom.index, mask=(0, 0, 0)))
    atoms.set_constraint(c)
    return atoms

def gMOOH_old(symbols, a=2.8295, c=20.9472, z=0.612, mags=(0.5, 0), electrolyte='K'):
    A = symbols[0]
    B = symbols[1]
    C = electrolyte
    d = 0.9
    hx = 0.7
    hz = 0.105
    h2x = .27
    h2z = .03
    a1 = np.array((0.5*a, -0.5*3**0.5*a, 0))
    a2 = np.array((0.5*a, 0.5*3**0.5*a, 0))
    a3 = np.array((0, 0, c))
    b1 = 2 * a1 + a2
    b2 = 2 * a2 + a1
    c1 = a1 + a2
    c2 = 2*a1 + 2*a2
    r1 = 2./3.*a1 + 1./3.*a2 + 1./3.*a3
    r2 = 1./3.*a1 + 2./3.*a2 + 2./3.*a3

    atoms = Atoms([Atom(A, np.array((0, 0, 0))),
                   Atom(C, 1./2.*a1 + 1./2.*a2 + 1./2.*a3),
                   Atom(B, (1./2. + d)*a1 + (1./2. + d)*a2 + 1./2.*a3),
                   Atom('H', (1./2. + hx)*a1 + (1./2. + hx)*a2 + (1./2. + hz/3)*a3),
                   Atom('H', (1./2. + hx)*a1 + (1./2. + hx)*a2 + (1./2. - hz/3)*a3),
                   Atom(B, (1./2. - d)*a1 + (1./2. - d)*a2 + 1./2.*a3 + b1 + b2),
                   Atom('H', (1./2. - hx)*a1 + (1./2. - hx)*a2 + (1./2. + hz/3)*a3 + b1 + b2),
                   Atom('H', (1./2. - hx)*a1 + (1./2. - hx)*a2 + (1./2. - hz/3)*a3 + b1 + b2),
                   Atom(A, r1),
                   Atom(A, r2),
                   Atom(B, z * a3),
                   Atom(B, (1 - z) * a3),
                   Atom(B, z * a3 + r1),
                   Atom('H', -(h2x*a1 + h2x*a2) + (z - h2z) * a3 + r1),
                   Atom('H', (h2x*a1 + h2x*a2) + (z - h2z) * a3 + r1),
                   Atom('H', -(h2x*a1 + h2x*a2) + (1 - z + h2z) * a3 + r1),
                   Atom('H', (h2x*a1 + h2x*a2) + (1 - z + h2z) * a3 + r1),
                   Atom(B, (1 - z) * a3 + r1),
                   Atom(B, (z - 1) * a3 + r2),
                   Atom(B, (-z) * a3 + r2),
                   
                   Atom(A, np.array((0, 0, 0)) + c1),
                   Atom(A, r1 + c1),
                   Atom(C, 1./2.*a1 + 1./2.*a2 + 1./2.*a3 + r1 + c1),
                   Atom(B, (1./2. + d)*a1 + (1./2. + d)*a2 + 1./2.*a3 + r1 + c1),
                   Atom('H', (1./2. + hx)*a1 + (1./2. + hx)*a2 + (1./2. + hz/3)*a3 + r1 + c1),
                   Atom('H', (1./2. + hx)*a1 + (1./2. + hx)*a2 + (1./2. - hz/3)*a3 + r1 + c1),
                   Atom(B, (1./2. - d)*a1 + (1./2. - d)*a2 + 1./2.*a3 + r1 + c1),
                   Atom('H', (1./2. - hx)*a1 + (1./2. - hx)*a2 + (1./2. + hz/3)*a3 + r1 + c1),
                   Atom('H', (1./2. - hx)*a1 + (1./2. - hx)*a2 + (1./2. - hz/3)*a3 + r1 + c1),
                   Atom(A, r2 + c1),
                   Atom(B, z * a3 + c1),
                   Atom(B, (1 - z) * a3 + c1),
                   Atom(B, z * a3 + r1 + c1),
                   Atom(B, (1 - z) * a3 + r1 + c1),
                   Atom(B, (z - 1) * a3 + r2 + c1),
                   Atom('H', -(h2x*a1 + h2x*a2) + (z - 1 - h2z) * a3 + r2 + c1),
                   Atom('H', (h2x*a1 + h2x*a2) + (z - 1 - h2z) * a3 + r2 + c1),
                   Atom('H', -(h2x*a1 + h2x*a2) + (-z + h2z) * a3 + r2 + c1),
                   Atom('H', (h2x*a1 + h2x*a2) + (-z + h2z) * a3 + r2 + c1),
                   Atom(B, (-z) * a3 + r2 + c1),
                   
                   Atom(A, np.array((0, 0, 0)) + c2),
                   Atom(A, r1 + c2),
                   Atom(A, r2 + c2),
                   Atom(C, 1./2.*a1 + 1./2.*a2 + 1./2.*a3 + r2 - a3 + c2 - b2),
                   Atom(B, (1./2. + d)*a1 + (1./2. + d)*a2 + 1./2.*a3 + r2 - a3 + c2 - b2 - b1),
                   Atom('H', (1./2. + hx)*a1 + (1./2. + hx)*a2 + (1./2. + hz/3)*a3 + r2 - a3 + c2 - b2 - b1),
                   Atom('H', (1./2. + hx)*a1 + (1./2. + hx)*a2 + (1./2. - hz/3)*a3 + r2 - a3 + c2 - b2 - b1),
                   Atom(B, (1./2. - d)*a1 + (1./2. - d)*a2 + 1./2.*a3 + r2 - a3 + c2),
                   Atom('H', (1./2. - hx)*a1 + (1./2. - hx)*a2 + (1./2. + hz/3)*a3 + r2 - a3 + c2),
                   Atom('H', (1./2. - hx)*a1 + (1./2. - hx)*a2 + (1./2. - hz/3)*a3 + r2 - a3 + c2),
                   Atom(B, z * a3 + c2),
                   Atom('H', -(h2x*a1 + h2x*a2) + (z - h2z) * a3 + c2),
                   Atom('H', (h2x*a1 + h2x*a2) + (z - h2z) * a3 + c2),
                   Atom('H', -(h2x*a1 + h2x*a2) + (1 - z + h2z) * a3 + c2),
                   Atom('H', (h2x*a1 + h2x*a2) + (1 - z + h2z) * a3 + c2),
                   Atom(B, (1 - z) * a3 + c2),
                   Atom(B, z * a3 + r1 + c2),
                   Atom(B, (1 - z) * a3 + r1 + c2),
                   Atom(B, (z - 1) * a3 + r2 + c2),
                   Atom(B, (-z) * a3 + r2 + c2)],
                  cell=(b1, b2, a3))

    return atoms

def gMOOH(symbols, a=5.1, b=9.1, c=6.84, mag=0.5, electrolyte='K'):
    A = symbols[0]
    B = symbols[1]
    C = electrolyte

    a1 = np.array([a, 0, 0])
    a2 = np.array([0, b, 0])
    a3 = np.array([c * np.cos(74.3 * np.pi / 180), 0, 
                   c * np.sin(74.3 * np.pi / 180)])
    r1 = 0.5 * a1 + 0.5 * a2
    
    atoms = Atoms([Atom(A, np.array([0, 0, 0]), magmom=mag),
                   Atom(A, r1, magmom=mag),
                   Atom(A, 0.2891 * a2, magmom=mag),
                   Atom(A, -0.2891 * a2, magmom=mag),
                   Atom(A, 0.2891 * a2 + r1, magmom=mag),
                   Atom(A, -0.2891 * a2 + r1, magmom=mag),
                   Atom(B, 0.2939 * a2 + 0.5 * a3),
                   Atom(B, -0.2939 * a2 + 0.5 * a3),
                   Atom(B, 0.2939 * a2 + 0.5 * a3 + r1),
                   Atom(B, -0.2939 * a2 + 0.5 * a3 + r1),
                   Atom(B, 0.3739 * a1 + 0.2491 * a3),
                   Atom(B, -0.3739 * a1 + -0.2491 * a3),
                   Atom(B, 0.3739 * a1 + 0.2491 * a3 + r1),
                   Atom(B, -0.3739 * a1 + -0.2491 * a3 + r1),
                   Atom(B, 0.3718 * a1 + 0.6564 * a2 + 0.1480 * a3),
                   Atom(B, -0.3718 * a1 + 0.6564 * a2 + -0.1480 * a3),
                   Atom(B, -0.3718 * a1 + -0.6564 * a2 + -0.1480 * a3),
                   Atom(B, 0.3718 * a1 + -0.6564 * a2 + 0.1480 * a3),
                   Atom(B, 0.3718 * a1 + 0.6564 * a2 + 0.1480 * a3 + r1),
                   Atom(B, -0.3718 * a1 + 0.6564 * a2 + -0.1480 * a3 + r1),
                   Atom(B, -0.3718 * a1 + -0.6564 * a2 + -0.1480 * a3 + r1),
                   Atom(B, 0.3718 * a1 + -0.6564 * a2 + 0.1480 * a3 + r1),
                   
                   Atom('H', 0.4620 * a1 + 0.2643 * a2 + 0.3651 * a3),
                   Atom('H', -0.4620 * a1 + 0.2643 * a2 + -0.3651 * a3),
                   Atom('H', -0.4620 * a1 + -0.2643 * a2 + -0.3651 * a3),
                   Atom('H', 0.4620 * a1 + -0.2643 * a2 + 0.3651 * a3),
                   Atom('H', 0.4620 * a1 + 0.2643 * a2 + 0.3651 * a3 + r1),
                   Atom('H', -0.4620 * a1 + 0.2643 * a2 + -0.3651 * a3 + r1),
                   Atom('H', -0.4620 * a1 + -0.2643 * a2 + -0.3651 * a3 + r1),
                   Atom('H', 0.4620 * a1 + -0.2643 * a2 + 0.3651 * a3 + r1),
                   
                   Atom('H', 0.4042 * a1 + 0.0802 * a2 + 0.3435 * a3),
                   Atom('H', -0.4042 * a1 + 0.0802 * a2 + -0.3435 * a3),
                   Atom('H', -0.4042 * a1 + -0.0802 * a2 + -0.3435 * a3),
                   Atom('H', 0.4042 * a1 + -0.0802 * a2 + 0.3435 * a3),
                   Atom('H', 0.4042 * a1 + 0.0802 * a2 + 0.3435 * a3 + r1),
                   Atom('H', -0.4042 * a1 + 0.0802 * a2 + -0.3435 * a3 + r1),
                   Atom('H', -0.4042 * a1 + -0.0802 * a2 + -0.3435 * a3 + r1),
                   Atom('H', 0.4042 * a1 + -0.0802 * a2 + 0.3435 * a3 + r1),
                   
                   Atom(C, 0.5 * a3),
                   Atom(C, 0.5 * a3 + r1)],
                  
                  cell=(a1, a2, a3))
    
    pos = atoms.get_scaled_positions()
    atoms.set_scaled_positions(pos % [1, 1, 1])
    
    return atoms

                  
def set_volume(self, vol):
    '''This is short function to quickly isotropically expand/reduce the
    volume'''

    factor = vol/self.get_volume()
    cell_factor = factor ** (1. / 3.)
    cell0 = self.get_cell()
    self.set_cell(cell0 * cell_factor, scale_atoms=True)

    return

Atoms.set_volume = set_volume
