import os
import numpy as np
from ctypes import *
import ctypes
from numpy.ctypeslib import ndpointer

class ParticleDataCtypes(ctypes.Structure):
     _fields_ = [("npart", c_int),
                ("x", POINTER(c_double)),
                ("xp", POINTER(c_double)),
                ("y", POINTER(c_double)),
                ("yp", POINTER(c_double)),
                ("z", POINTER(c_double)),
                ("dp", POINTER(c_double)),
                ("q0", POINTER(c_double)),
                ("mass0", POINTER(c_double)),
                ("beta0", POINTER(c_double)),
                ("gamma0", POINTER(c_double)),
                ("p0c", POINTER(c_double))
                ]


def particle_coordinates_sixtrack_struct(coord_momenta_dict, fname, n_part):
    """
    Pass the particle coordinates to sixtracklib's tracking method.
    """
    mylib = cdll.LoadLibrary(fname)
    coord_momenta_ptr_dict = {}
    proc_flag = 'g'
    for attr in coord_momenta_dict.keys():
        vals = coord_momenta_dict[attr]
        if type(vals) is np.ndarray:
            proc_flag = 'c'
            coord_momenta_ptr_dict[attr] = np.ctypeslib.as_ctypes(vals)
        else:
            coord_momenta_ptr_dict[attr] = ctypes.cast(vals.ptr, ctypes.POINTER(c_double))

    data = ParticleDataCtypes(c_int(n_part),
            coord_momenta_ptr_dict['x'],
            coord_momenta_ptr_dict['xp'],
            coord_momenta_ptr_dict['y'],
            coord_momenta_ptr_dict['yp'],
            coord_momenta_ptr_dict['z'],
            coord_momenta_ptr_dict['dp'],
            coord_momenta_ptr_dict['q0'],
            coord_momenta_ptr_dict['mass0'],
            coord_momenta_ptr_dict['beta0'],
            coord_momenta_ptr_dict['gamma0'],
            coord_momenta_ptr_dict['p0c']
            )

    mylib.run(byref(data), c_char(proc_flag))
    return coord_momenta_dict
