#!/usr/bin/env python
from os.path import isfile, getsize
import shutil
from subprocess import Popen, PIPE

default_name = os.getcwd + '-DDEC'

def qcalculate(name=default_name, time='long'):
    '''Function uses the densite derived electorstatic chargs (DDEC) method
    for calculating net atomic charges

    Requires to be in the same folder as these VASP output files
    AECCAR0, AECCAR2, CHG, and POTCAR
    Specify these files to be written by putting in the INCAR
    LCHARG = .TRUE.
    LAECHG = .TRUE.
    PREC = Accurate
    '''
    files = ('AECCAR0', 'AECCAR2', 'CHG', 'POTCAR')
    for f in files:
        if isfile(f) != True or getsize(f) == 0:
            raise LookupError('Could not read file' + f)
    
    CWD = os.getcwd()
    charge_mol_path = '/home/zhongnanxu/opt/chargemol_03_28_2012/chargemol_job_master.m'
    shutil.copy(charge_mol_path, CWD)
    run_script = '''#!/bin/bash
cd $PBS_O_WORKDIR

echo "Nodes chosen are:"
cat $PBS_NODEFILE

matlab -nodesktop -r chargemol_job > chargemol_output.txt'''

    if time == 'short':
        hours = 24
    else:
        hours = 168
    resources = '-l walltime=%i:00:00' % hours
    p = Popen(['qsub', '-joe', '-N', '%s' % name, resources],
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate(script)
    return
