'''
@date:   30/09/2015
@author: Stefan Hegglin
'''
from __future__ import division

import sys, os
BIN = os.path.dirname(__file__) # ./PyHEADTAIL/testing/unittests/
BIN = os.path.abspath( BIN ) # absolute path to unittests
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/testing/
BIN = os.path.dirname( BIN ) # ../ -->  ./PyHEADTAIL/
BIN = os.path.dirname( BIN ) # ../ -->  ./
sys.path.append(BIN)

import unittest
import numpy as np
from scipy.constants import c, e, m_p
import copy
# try to import pycuda, if not available --> skip this test file
try:
    import pycuda.autoinit
    import pycuda.gpuarray
except ImportError:
    has_pycuda = False
else:
    has_pycuda = True

import PyHEADTAIL.general.pmath as pm
from PyHEADTAIL.particles.particles import Particles
from PyHEADTAIL.particles.slicing import UniformBinSlicer



class TestDispatch(unittest.TestCase):
    '''Test Class for the function dispatch functionality in general.pmath'''
    def setUp(self):
        self.available_CPU = pm._CPU_numpy_func_dict.keys()
        self.available_GPU = pm._GPU_func_dict.keys()

    def test_set_CPU(self):
        pm.update_active_dict(pm._CPU_numpy_func_dict)
        self.assertTrue(
            set(self.available_CPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to CPU fails. Not all CPU functions ' +
            'were spilled to pm.globals()'
            )

    def test_set_GPU(self):
        pm.update_active_dict(pm._GPU_func_dict)
        self.assertTrue(
            set(self.available_GPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to GPU fails. Not all GPU functions ' +
            'were spilled to pm.globals()'
            )
        self.assertFalse(
            set(self.available_CPU).issubset(set(pm.__dict__.keys())),
            'Setting the active dict to GPU fails. Not all CPU functions ' +
            'were deleted from pm.globals() when switching to GPU.'
            )

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_equivalency_CPU_GPU_functions(self):
        '''
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions. Only single param funnctions.
        Use a large sample size to account for std/mean fluctuations due to
        different algorithms (single pass/shifted/...)
        '''
        multi_param_fn = ['emittance', 'apply_permutation', 'mean_per_slice',
            'std_per_slice', 'emittance_per_slice', 'particles_within_cuts',
            'macroparticles_per_slice']
        np.random.seed(0)
        parameter_cpu = np.random.normal(loc=1., scale=1., size=100000)
        parameter_gpu = pycuda.gpuarray.to_gpu(parameter_cpu)
        common_functions = [fn for fn in self.available_CPU
                            if fn in self.available_GPU]
        for fname in common_functions:
            if fname not in multi_param_fn:
                res_cpu = pm._CPU_numpy_func_dict[fname](parameter_cpu)
                res_gpu = pm._GPU_func_dict[fname](parameter_gpu)
                if isinstance(res_gpu, pycuda.gpuarray.GPUArray):
                    res_gpu = res_gpu.get()
                self.assertTrue(np.allclose(res_cpu, res_gpu),
                    'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_emittance_computation(self):
        '''
        Emittance computation only, requires a special funcition call.
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        Use a large number of samples (~500k). The CPU and GPU computations
        are not exactly the same due to differences in the algorithms (i.e.
        biased/unbiased estimator)
        '''
        fname = 'emittance'
        np.random.seed(0)
        parameter_cpu_1 = np.random.normal(loc=1., scale=.1, size=500000)
        parameter_cpu_2 = np.random.normal(loc=1., scale=1., size=500000)
        parameter_cpu_3 = np.random.normal(loc=1., scale=1., size=500000)
        parameter_gpu_1 = pycuda.gpuarray.to_gpu(parameter_cpu_1)
        parameter_gpu_2 = pycuda.gpuarray.to_gpu(parameter_cpu_2)
        parameter_gpu_3 = pycuda.gpuarray.to_gpu(parameter_cpu_3)
        params_cpu = [parameter_cpu_1, parameter_cpu_2, parameter_cpu_3]
        params_gpu = [parameter_gpu_1, parameter_gpu_2, parameter_gpu_3]
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        if isinstance(res_gpu, pycuda.gpuarray.GPUArray):
            res_gpu = res_gpu.get()
        self.assertTrue(np.allclose(res_cpu, res_gpu),
            'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_apply_permutation_computation(self):
        '''
        apply_permutation only, requires a special function call.
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = 'apply_permutation'
        np.random.seed(0)
        n = 10
        parameter_cpu_tosort = np.random.normal(loc=1., scale=.1, size=n)
        parameter_gpu_tosort = pycuda.gpuarray.to_gpu(parameter_cpu_tosort)
        parameter_cpu_perm = np.array(np.random.permutation(n), dtype=np.int32)
        parameter_gpu_perm = pycuda.gpuarray.to_gpu(parameter_cpu_perm)
        params_cpu = [parameter_cpu_tosort, parameter_cpu_perm]
        params_gpu = [parameter_gpu_tosort, parameter_gpu_perm]
        res_cpu = pm._CPU_numpy_func_dict[fname](*params_cpu)
        res_gpu = pm._GPU_func_dict[fname](*params_gpu)
        self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
            'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_per_slice_stats(self):
        '''
        All per_slice functions (mean, cov, ?emittance)
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fnames = ['mean_per_slice', 'std_per_slice']
        np.random.seed(0)
        n = 99999
        b = self.create_gaussian_bunch(n)
        b.sort_for('z')
        slicer = UniformBinSlicer(n_slices=777, n_sigma_z=None)
        s_set = b.get_slices(slicer)
        z_cpu = b.z.copy()
        z_gpu = pycuda.gpuarray.to_gpu(z_cpu)
        sliceset_cpu = s_set
        sliceset_gpu = copy.deepcopy(s_set)
        sliceset_gpu.slice_index_of_particle = pycuda.gpuarray.to_gpu(
            s_set.slice_index_of_particle
        )
        for fname in fnames:
            res_cpu = pm._CPU_numpy_func_dict[fname](sliceset_cpu, z_cpu)
            res_gpu = pm._GPU_func_dict[fname](sliceset_gpu,z_gpu)
            self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
                'CPU/GPU version of ' + fname + ' dont yield the same result')
        fnames = ['emittance_per_slice']
        v_cpu = b.x
        v_gpu = pycuda.gpuarray.to_gpu(v_cpu)
        dp_cpu = z_cpu + np.arange(n)/n
        dp_gpu = pycuda.gpuarray.to_gpu(dp_cpu)
        for fname in fnames:
            res_cpu = pm._CPU_numpy_func_dict[fname](sliceset_cpu, z_cpu, v_cpu, dp_cpu)
            res_gpu = pm._GPU_func_dict[fname](sliceset_gpu, z_gpu, v_gpu, dp_gpu)
            # only check things which aren't nan/None. Ignore RuntimeWarning!
            with np.errstate(invalid='ignore'):
                res_cpu = res_cpu[res_cpu>1e-10]
                res_gpu = res_gpu.get()[res_gpu.get()>1e-10]
            self.assertTrue(np.allclose(res_cpu, res_gpu),
                'CPU/GPU version of ' + fname + ' dont yield the same result')

    @unittest.skipUnless(has_pycuda, 'pycuda not found')
    def test_sliceset_computations(self):
        '''
        macroparticles per slice, particles_within_cuts
        require a sliceset as a parameter
        Check that CPU/GPU functions yield the same result (if both exist)
        No complete tracking, only bare functions.
        '''
        fname = ['particles_within_cuts', 'macroparticles_per_slice']
        pm.update_active_dict(pm._CPU_numpy_func_dict)
        np.random.seed(0)
        n = 999
        b = self.create_gaussian_bunch(n)
        b.sort_for('z')
        slicer = UniformBinSlicer(n_slices=20, n_sigma_z=2)
        s_set = b.get_slices(slicer)
        z_cpu = b.z.copy()
        z_gpu = pycuda.gpuarray.to_gpu(z_cpu)
        sliceset_cpu = s_set
        sliceset_gpu = copy.deepcopy(s_set)
        sliceset_gpu.slice_index_of_particle = pycuda.gpuarray.to_gpu(
            s_set.slice_index_of_particle
        )
        params_cpu = [sliceset_cpu]
        params_gpu = [sliceset_gpu]
        for f in fname:
            res_cpu = pm._CPU_numpy_func_dict[f](*params_cpu)
            res_gpu = pm._GPU_func_dict[f](*params_gpu)
            self.assertTrue(np.allclose(res_cpu, res_gpu.get()),
                'CPU/GPU version of ' + f + ' dont yield the same result')



    def create_all1_bunch(self, n_macroparticles):
        np.random.seed(1)
        x = np.ones(n_macroparticles)
        y = x.copy()
        z = x.copy()
        xp = x.copy()
        yp = x.copy()
        dp = x.copy()
        coords_n_momenta_dict = {
            'x': x, 'y': y, 'z': z,
            'xp': xp, 'yp': yp, 'dp': dp
        }
        return Particles(
            macroparticlenumber=len(x), particlenumber_per_mp=100, charge=e,
            mass=m_p, circumference=100, gamma=10,
            coords_n_momenta_dict=coords_n_momenta_dict
        )

    def create_gaussian_bunch(self, n_macroparticles):
        P = self.create_all1_bunch(n_macroparticles)
        P.x = np.random.randn(n_macroparticles)
        P.y = np.random.randn(n_macroparticles)
        P.z = np.random.randn(n_macroparticles)
        P.xp = np.random.randn(n_macroparticles)
        P.yp = np.random.randn(n_macroparticles)
        P.dp = np.random.randn(n_macroparticles)
        return P

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
