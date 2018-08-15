# coding: utf-8

# In[1]:

from __future__ import division, print_function
import pycuda.autoinit

import numpy as np
import pdb
import cPickle as pkl
# In[2]:

import sys
#sys.path.append('PyHEADTAIL/')


# In[3]:

# need to initiate GPU before first import of PyHEADTAIL
try:
    import pycuda.autoinit
except ImportError:
    print ('No GPU available.')


# In[5]:

from PyHEADTAIL.general.contextmanager import CPU, GPU
from PyHEADTAIL.sixtracklib_interface.ctypes_interface import *

# In[6]:

#from PyCERNmachines.CERNmachines import PS
sys.path.append('/home/mmadhyas/PyCERNmachines')
from CERNmachines import PS, SPS


# In[7]:

n_macroparticles = int(1e4)
n_slices = 100
n_segments = 1 #10

intensity = 1.6e12
epsn_z = 1.3 # in eV*s
epsn_x = epsn_y = 2e-6 # in m*rad

machine_configuration = 'LHCbeam_h7'
longitudinal_focusing = 'non-linear'


# In[8]:

machine = PS(
    n_segments=n_segments,
    machine_configuration=machine_configuration,
    gamma=2.49,
    longitudinal_focusing=longitudinal_focusing,
)

machine.longitudinal_map.pop_kick(1)

machine.longitudinal_map.phi_offsets[0] = np.pi # below transition


# In[9]:
beam = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, epsn_z)

with GPU(beam) as cmg:

    momenta_coords_dict = beam.get_coords_n_momenta_dict()
    print('keys:', momenta_coords_dict.keys())
    #pdb.set_trace()
    particle_coordinates_sixtrack_struct(momenta_coords_dict, fname="/home/mmadhyas/sixtracklib_gsoc18/studies/study10/build/libsample_fodod.so", n_part=10000)
    #new_particle_coordinates_sixtrackgpu(momenta_coords_dict, fname="/home/mmadhyas/sixtracklib_gsoc18/studies/study10/build/libsample_fodod.so", n_part=10000)

'''
momenta_coords_dict = beam.get_coords_n_momenta_dict()
res =  (new_particle_coordinates_sixtrack(momenta_coords_dict, fname="/home/mmadhyas/sixtracklib_gsoc18/studies/study10/build/libsample_fodo.so", n_part=10000))
'''
