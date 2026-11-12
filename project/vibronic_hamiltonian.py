# system imports
import os
from os.path import join
import itertools as it
import types
import sys

# third party imports
import scipy as scp
# from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
# import matplotlib.animation as anim
import parse  # used for loading data files
import opt_einsum as oe
import pandas as pd


# local imports
from project.log_conf import log
from project.two_mode_model import model_two_mode
from project.vibronic import vIO, VMK
from project.residual_equations import create_truncation_order_object
from project.residual_equations.symmetrize import symmetrize_tensor

# testing generated equations
from project.residual_equations import eT_zhz_eqs_H_2_P_4_T_1_exp_4_Z_1 as z_one_eqns
# create correct residual equations
from project.residual_equations import eT_zhz_eqs_H_2_P_4_T_1_exp_4_Z_2 as z_two_eqns
from project.residual_equations import eT_zhz_eqs_H_2_P_4_T_1_exp_4_Z_3 as z_three_eqns
from project.residual_equations import eT_zhz_eqs_RHS_H_4_P_4_T_1_exp_1_Z_4 as z_four_eqns
# tested modified integrator
from project.temporary_trial_solver import new_solve_ivp

# Force any printing of numpy arrays to be at a set precision
# to improve reading of output/debug print statements/logs
np.set_printoptions(precision=10, suppress=True, linewidth=200)

# factoring out the debug statements
print_force_constants = False
print_full_hamiltonian = False
print_U_matrix_and_percent_progress = True
check_hermicity_of_H_SOS = True
calculating_hot_bands = False

tab = " "*4

optimized = False


class vibronic_hamiltonian(object):
    """ vibronic_hamiltonian is python object """
    
    
    def __init__(
            self, model, model_name, build_H=False, HO_size=40, cc_truncation_order=1, hamiltonian_truncation_order=1,
            compare_with_mctdh=True, comparing_to_test_models=False,
            trans=False, theta=0, W_truncation_order=0, solve_W=False, solve_Z=False,
            cal_Z0=False, cal_t0=True, solve_t=False, Z_truncation_order=2, T_truncation_order=1,
            selected_surface=[],
            force_use_optimize=False,
            calculate_population_flag=True,normal_order= True,init_disp = 0, disps = [],
            correlation_funcs = False, z_check = False, include_norm = False
    ):
        # 
        assert not comparing_to_test_models
        
        # model name
        self.model_name = model_name

        # number of potential energy surfaces
        self.A = model[VMK.A]

        # number of normal modes
        self.N = model[VMK.N]

        # using optimized paths only pays off for larger systems
        # or for higher order terms
        nof_dimensions = self.A + self.N

        self.op_einsum_flag = False

        if force_use_optimize or (nof_dimensions >= 15):
            self.op_einsum_flag = True
        else:
            self.op_einsum_flag = False

        self.calculate_population_flag = bool(calculate_population_flag) # maybe get rid of 

        # Vibronic model
        self.model = model

        # initialize truncation order of T, Z tensor
        self.Z_truncation_order = Z_truncation_order
        self.T_truncation_order = T_truncation_order

        if self.Z_truncation_order > 4:
            raise Exception('Only supports up to Z4 truncation, not Z5 or higher.')

        if self.T_truncation_order > 1:
            raise Exception('Only supports up to T1 truncation, not T2 or higher.')

        # ---------------------------------------------------------------------
        # make the truncation objects for the generated code
        # ---------------------------------------------------------------------
        # create the ground state/excited state ansatz
        self.ansatz = types.SimpleNamespace()
        self.ansatz.ground_state = True

        # create the truncation obj
        self.gen_trunc = types.SimpleNamespace()

        self.gen_trunc.h_at_least_linear = False
        self.gen_trunc.h_at_least_quadratic = False
        self.gen_trunc.h_at_least_cubic = False
        self.gen_trunc.h_at_least_quartic = False
        
        if hamiltonian_truncation_order >= 1:
            self.gen_trunc.h_at_least_linear = True
        if hamiltonian_truncation_order >= 2:
            self.gen_trunc.h_at_least_quadratic = True
        if hamiltonian_truncation_order >= 3:
            self.gen_trunc.h_at_least_cubic = True
        if hamiltonian_truncation_order >= 4:
            self.gen_trunc.h_at_least_quartic = True

        if hamiltonian_truncation_order > 4:
            assert False, 'Hamiltonians of this <4 order are not supported yet'

        self.gen_trunc.t_singles = False

        if T_truncation_order >= 1:
            self.gen_trunc.t_singles = True
        if T_truncation_order >= 2:
            assert False, 'T_2 not supported yet'

        self.gen_trunc.z_at_least_linear = False
        self.gen_trunc.z_at_least_quadratic = False
        self.gen_trunc.z_at_least_cubic = False
        self.gen_trunc.z_at_least_quartic = False

        if Z_truncation_order >= 1:
            self.gen_trunc.z_at_least_linear = True
        if Z_truncation_order >= 2:
            self.gen_trunc.z_at_least_quadratic = True
        if Z_truncation_order >= 3:
            self.gen_trunc.z_at_least_cubic = True
        if Z_truncation_order >= 4:
            self.gen_trunc.z_at_least_quartic = True
        
        if Z_truncation_order > 4:
            assert False, 'Hamiltonians of this <4 order are not supported yet'
            

        # these functions probably need to be re-worked
        self.gen_trunc.confirm_at_least_singles = lambda: True
        self.gen_trunc.confirm_at_least_doubles = lambda: True
        self.gen_trunc.confirm_at_least_triples = lambda: True

        # ---------------------------------------------------------------------
         
        A, N = self.A, self.N

        # initialize derivative tensors
        self.dT = {
            0: np.zeros(A, dtype=complex),
        }

        if self.T_truncation_order >= 1:
            self.dT[1] = np.zeros((A, N), dtype=complex)


        self.dZ = {
            0: np.zeros((A, A), dtype=complex),
        }

        if self.Z_truncation_order >= 1:
            self.dZ[1] = np.zeros((A, A, N), dtype=complex)

        if self.Z_truncation_order >= 2:
            self.dZ[2] = np.zeros((A, A, N, N), dtype=complex)

        if self.Z_truncation_order >= 3:
            self.dZ[3] = np.zeros((A, A, N, N, N), dtype=complex)
            
        if self.Z_truncation_order >= 4:
            self.dZ[4] = np.zeros((A, A, N, N, N, N), dtype=complex)

        # ---------------------------------------------------------------------
        # initialize Lagrange multiplier
        self.la = np.zeros(self.A, dtype=complex)

        # initialize norm
        # self.Norm = []

        # initialize state property
        if self.calculate_population_flag:
            self.State_pop_DB = []
            self.State_pop_AB = []

        # set up step size
        self.step_size = 5e-3

        # users defined selected surfaces to be decoupled
        self.selected_surface = selected_surface
        # breakpoint()
        # create the `truncation_order_namedtuple` used in calculation residuals and W operators
        self.trunc = create_truncation_order_object(cc_truncation_order, hamiltonian_truncation_order)

        # print information related to truncation
        self._check_truncation_info()

        # electronic
        self.E_tdm = self.model[VMK.etdm].copy()
        # magnetic
        self.M_tdm = self.model[VMK.mtdm].copy()

        log.info("Electronic TDM:{:}".format(self.E_tdm.shape))
        log.info("Magnetic TDM:{:}".format(self.M_tdm.shape))

        self.HO_size, self.trans, self.theta = HO_size, trans, theta

        if print_force_constants:
            vIO.print_model(self.model)

        # save the flags for `initialize_hamiltonian`
        self.compare_with_mctdh = compare_with_mctdh
        self.comparing_to_test_models = comparing_to_test_models

        # the majority of the work
        self.normal_order = normal_order
        self._initialize_hamiltonian()
        self._init_qs_1d()
        
        # precompute optimal einsum paths to speed up integration
        if self.op_einsum_flag:
            # self._initialize_einsum_paths()
            self._initialize_opt_einsum_paths()
            self._initialize_new_scheme_D_einsum_paths()
            self._initialize_cal_dT_Tran_C_einsum_paths()
            self._initialize_cal_H_bar_tilde_einsum_paths()
            self._initialize_compute_C_matrix_einsum_paths()

        # these lines are only for testing against
        # Full configuration-interaction (FCI)
        if build_H:
            self._construct_debug_hamiltonian_for_FCI()

        # just double check that quadratic terms are symmetric
        for key in self.h.keys():
            rank = key[0] + key[1]
            if rank == 2:
                assert np.allclose(self.h[key], np.transpose(self.h[key], (0, 1, 3, 2))), (
                    f"{key = } is not symmetric"
                )
        
        # new features
        self.correlation_funcs = correlation_funcs
        self.z_check = z_check 
        self.include_norm = include_norm
        
        # if self.correlation_funcs == True: 
        #     assert self.N == 1, 'correlation functions only calculated for single mode systems'
        
        self.init_t = (init_disp)/np.sqrt(2)
        if len(disps) != 0:
            temp_t = np.zeros([self.A, self.N], dtype=complex)[0]
            tempt = np.array(disps,dtype=complex)
            self.init_t = temp_t+tempt/np.sqrt(2)
        
        self.wavenumber_units = False

        return


    def _check_truncation_info(self):
        """ Some debugging and output just to make sure we didn't pick conflicting values """
        hamiltonian_order = vIO.extract_maximum_order_of_model(self.model)
        truncation_info = (
            '\n'
            f"The Hamiltonian provided contains at most {hamiltonian_order} order terms\n"
            # f"The calculation will considering up to    {self.trunc.hamiltonian_truncation_order} order terms\n"
            # f"Our Coupled Cluster calculation is: {self.trunc.cc_abbreviation()}\n")
        )

        truncation_info += f"Z truncation level is :{self.Z_truncation_order}\n"
        truncation_info += f"T truncation level is :{self.T_truncation_order}\n"

        log.info(truncation_info)

        # make sure we have sufficient CC truncation to calculate model to the order requested
        # assert self.trunc.cc_truncation_order >= self.trunc.hamiltonian_truncation_order, truncation_info
        # make sure that the Hamiltonian has sufficient terms to calculate `highest_order`
        
        # Temporarily off to test H=0
        # assert hamiltonian_order >= self.trunc.hamiltonian_truncation_order, truncation_info

        # self.highest_order = highest_order
        # self.mode_2_hamiltonian_order = hamiltonian_order
        return

    def _initialize_einsum_paths(self):
        """ Precompute einsum paths for the most computationally expensive parts of the integration.

        With optimized einsum paths runtime can be half as long.
        However since the paths are computed on every integration step, the longer the propagation time
        the more savings can be gained by precomputing static paths ahead of time.

        May also want to consult
        https://github.com/dgasmith/opt_einsum/blob/c826bb7df16f470a69f7bf90598fc27586209d11/docs/source/path_finding.rst
        """

        # ------------------------------------------------------------------------------------------------
        A, N = self.A, self.N

        # ostr = 'auto-hq'
        ostr = 'optimal'

        # ------------------------------------------------------------------------------------------------
        # constant z
        # ------------------------------------------------------------------------------------------------
        self.f_z_0_op_list = [
            oe.contract_expression('k,yx,xk->y', (N, ), (A, A), (A, N), optimize=ostr),
            oe.contract_expression('k,yxk,x->y', (N, ), (A, A, N), (A, ), optimize=ostr),
            oe.contract_expression('k,yxlk,xl->y', (N, ), (A, A, N, N), (A, N), optimize=ostr),
            oe.contract_expression('k,l,yxl,xk->y', (N, ), (N, ), (A, A, N), (A, N), optimize=ostr),
            oe.contract_expression('k,l,yxkl,x->y', (N, ), (N, ), (A, A, N, N), (A, ), optimize=ostr),
            oe.contract_expression('k,l,m,yxlm,xk->y', (N, ), (N, ), (N, ), (A, A, N, N), (A, N), optimize=ostr),
        ]

        if self.Z_truncation_order >= 2:
            self.f_z_0_op_list.extend([
                oe.contract_expression('k,yxl,xkl->y', (N, ), (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,yx,xkl->y', (N, ), (N, ), (A, A), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,yxml,xkm->y', (N, ), (N, ), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,m,yxm,xkl->y', (N, ), (N, ), (N, ), (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('i,j,k,l,yxij,xkl->y', (N, ), (N, ), (N, ), (N, ), (A, A, N, N), (A, N, N), optimize=ostr),

            ])

        # ------------------------------------------------------------------------------------------------
        # linear z
        # ------------------------------------------------------------------------------------------------
        self.f_z_i_op_list = [
            oe.contract_expression('yxi,x->yi', (A, A, N), (A,), optimize=ostr),
            oe.contract_expression('yxki,xk->yi', (A, A, N, N), (A, N), optimize=ostr),
            oe.contract_expression('yx,xi->yi', (A, A), (A, N), optimize=ostr),
            # adding terms associated e^T_dagger
            oe.contract_expression('k,yxi,xk->yi', (N,), (A, A, N), (A, N), optimize=ostr),
            oe.contract_expression('k,yxk,xi->yi', (N,), (A, A, N), (A, N), optimize=ostr),
            oe.contract_expression('k,yxki,x->yi', (N,), (A, A, N, N), (A,), optimize=ostr),
            oe.contract_expression('k,l,yxli,xk->yi', (N,), (N,), (A, A, N, N), (A, N), optimize=ostr),
            oe.contract_expression('k,l,yxkl,xi->yi', (N,), (N,), (A, A, N, N), (A, N), optimize=ostr)
        ]

        if self.Z_truncation_order >= 2:
            self.f_z_i_op_list.extend([
                oe.contract_expression('yxk,xki->yi', (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,yx,xki->yi', (N,), (A, A), (A, N, N), optimize=ostr),
                oe.contract_expression('k,yxli,xkl->yi', (N,), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,yxlk,xli->yi', (N,), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,yxl,xki->yi', (N,), (N,), (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,yxi,xkl->yi', (N,), (N,), (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,m,yxmi,xkl->yi', (N,), (N,), (N,), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,m,yxlm,xki->yi', (N,), (N,), (N,), (A, A, N, N), (A, N, N), optimize=ostr)
            ])

        # still testing different versions
        if False and self.T_truncation_order >= 2:
            self.f_z_i_op_list.extend([
                oe.contract_expression('kl,yxki,xl->yi', (N, N), (A, A, N, N), (A, N), optimize=ostr),
                oe.contract_expression('kl,yxkl,xi->yi', (N, N), (A, A, N, N), (A, N), optimize=ostr),
            ])

        # still testing different versions
        if False and self.Z_truncation_order >= 2 and self.T_truncation_order >= 2:
            self.f_z_i_op_list.extend([
                oe.contract_expression('kl,yxk,xli->yi', (N, N), (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('kl,yxi,xkl->yi', (N, N), (A, A, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,lm,yxli,xkm->yi', (N, ), (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,lm,yxkl,xmi->yi', (N, ), (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,lm,yxki,xlm->yi', (N, ), (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,lm,yxlm,xki->yi', (N, ), (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
            ])

        # ------------------------------------------------------------------------------------------------
        # quadratic z
        # ------------------------------------------------------------------------------------------------
        self.f_z_ij_op_list = [
            oe.contract_expression('yx,xij->yij', (A, A), (A, N, N), optimize=ostr),
            oe.contract_expression('yxi,xj->yij', (A, A, N), (A, N), optimize=ostr),
            oe.contract_expression('yxij,x->yij', (A, A, N, N), (A, ), optimize=ostr),
            oe.contract_expression('yxki,xkj->yij', (A, A, N, N), (A, N, N), optimize=ostr),
            # adding terms associated e^T_dagger
            oe.contract_expression('k,yxi,xkj->yij', (N, ), (A, A, N), (A, N, N), optimize=ostr),
            oe.contract_expression('k,yxk,xij->yij', (N, ), (A, A, N), (A, N, N), optimize=ostr),
            oe.contract_expression('k,yxij,xk->yij', (N, ), (A, A, N, N), (A, N), optimize=ostr),
            oe.contract_expression('k,yxki,xj->yij', (N, ), (A, A, N, N), (A, N), optimize=ostr),
            oe.contract_expression('k,l,yxli,xkj->yij', (N, ), (N, ), (A, A, N, N), (A, N, N), optimize=ostr),
            oe.contract_expression('k,l,yxij,xkl->yij', (N, ), (N, ), (A, A, N, N), (A, N, N), optimize=ostr),
            oe.contract_expression('k,l,yxkl,xij->yij', (N, ), (N, ), (A, A, N, N), (A, N, N), optimize=ostr),
        ]

        # still testing different versions
        if False and self.T_truncation_order >= 2:
            self.f_z_ij_op_list.extend([
                oe.contract_expression('kl,yxki,xlj->yij', (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('kl,yxkj,xli->yij', (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('kl,yxij,xkl->yij', (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('kl,yxkl,xij->yij', (N, N), (A, A, N, N), (A, N, N), optimize=ostr),
            ])

        # linear t
        # self.f_t[1]_op_list = [
        #     oe.contract_expression('abk,bki->abi', (A, A, N), (A, N, N)),
        #     oe.contract_expression('abkl,bk,bli->abi', (A, A, N, N), (A, N), (A, N, N)),
        # ]

        # # quadratic t
        # self.f_t[2]_op_list = [
        #     oe.contract_expression('abkj,bki->abij', (A, A, N, N), (A, N, N)),
        #     oe.contract_expression('abki,bkj->abij', (A, A, N, N), (A, N, N)),
        #     oe.contract_expression('abkl,bli,bkj->abij', (A, A, N, N), (A, N, N), (A, N, N)),
        # ]

        # ------------------------------------------------------------------------------------------------
        # constant z residual
        # ------------------------------------------------------------------------------------------------
        self.f_z_0_residual_list = []

        if self.Z_truncation_order >= 2:
            self.f_z_0_residual_list.extend([
                oe.contract_expression('l,m,ylm->y', (N, ), (N, ), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,ykl->y', (N, ), (N, ), (A, N, N), optimize=ostr),
            ])

        if self.T_truncation_order >= 2:
            self.f_z_0_residual_list.extend([
                oe.contract_expression('k,l,kl,y->y', (N, ), (N, ), (N, N), (A, ), optimize=ostr),
                oe.contract_expression('k,l,m,lm,yk->y', (N, ), (N, ), (N, ), (N, N), (A, N), optimize=ostr),
            ])

        if self.T_truncation_order >= 2 and self.Z_truncation_order >= 2:
            self.f_z_0_residual_list.extend([
                oe.contract_expression('k,l,m,n,kl,ymn->y', (N, ), (N, ), (N, ), (N, ), (N, N), (A, N, N), optimize=ostr),
            ])

        # ------------------------------------------------------------------------------------------------
        # linear z residual
        # ------------------------------------------------------------------------------------------------
        self.f_z_1_residual_list = []

        if self.Z_truncation_order >= 2:
            self.f_z_1_residual_list.extend([
                oe.contract_expression('l,yli->yi', (N, ), (A, N, N), optimize=ostr),
                oe.contract_expression('k,yki->yi', (N, ), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,i,ykl->yi', (N, ), (N, ), (N, ), (A, N, N), optimize=ostr),
            ])

        if self.T_truncation_order >= 2:
            self.f_z_1_residual_list.extend([
                oe.contract_expression('k,ki,y->yi', (N, ), (N, N), (A, ), optimize=ostr),
                oe.contract_expression('k,l,ki,yl->yi', (N, ), (N, ), (N, N), (A, N), optimize=ostr),
                oe.contract_expression('k,l,kl,yi->yi', (N, ), (N, ), (N, N), (A, N), optimize=ostr),
            ])

        if self.T_truncation_order >= 2 and self.Z_truncation_order >= 2:
            self.f_z_1_residual_list.extend([
                oe.contract_expression('k,l,m,mi,ykl->yi', (N, ), (N, ), (N, ), (N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,m,lm,yki->yi', (N, ), (N, ), (N, ), (N, N), (A, N, N), optimize=ostr),
            ])

        # ------------------------------------------------------------------------------------------------
        # quadratic z residual
        # ------------------------------------------------------------------------------------------------
        self.f_z_2_residual_list = [
            oe.contract_expression('i,yj->yij', (N, ), (A, N), optimize=ostr),
        ]

        if self.Z_truncation_order >= 2:
            self.f_z_2_residual_list.extend([
                oe.contract_expression('k,i,ykj->yij', (N, ), (N, ), (A, N, N), optimize=ostr),
            ])

        if self.T_truncation_order >= 2:
            self.f_z_2_residual_list.extend([
                oe.contract_expression('k,ki,yj->yij', (N, ), (N, N), (A, N), optimize=ostr),
                oe.contract_expression('ij,y->yij', (N, N), (A, ), optimize=ostr),
                oe.contract_expression('k,ij,yk->yij', (N, ), (N, N), (A, N), optimize=ostr),
            ])

        if self.T_truncation_order >= 2 and self.Z_truncation_order >= 2:
            self.f_z_2_residual_list.extend([
                oe.contract_expression('k,l,ki,ylj->yij', (N, ), (N, ), (N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,kl,yij->yij', (N, ), (N, ), (N, N), (A, N, N), optimize=ostr),
                oe.contract_expression('k,l,ij,ykl->yij', (N, ), (N, ), (N, N), (A, N, N), optimize=ostr),
            ])

        return

    def _initialize_opt_einsum_paths(self):
        """Store optimized paths as member of `self` for use in the `eT_zhz_eqs_***` module during integration."""

        if self.Z_truncation_order == 1:
            self.all_opt_paths = z_one_eqns.compute_all_optimized_paths(self.A, self.N, self.ansatz, self.gen_trunc)
        if self.Z_truncation_order == 2:
            self.all_opt_paths = z_two_eqns.compute_all_optimized_paths(self.A, self.N, self.ansatz, self.gen_trunc)
        if self.Z_truncation_order == 3:
            self.all_opt_paths = z_three_eqns.compute_all_optimized_paths(self.A, self.N, self.ansatz, self.gen_trunc)

        return

    def _initialize_new_scheme_D_einsum_paths(self):
        """Store optimized paths as member of `self` for use in the scheme
        This function should eventually  be replaced with someone more permanent.
        But for testing purposes for now its sufficient
        """
        A, N = self.A, self.N

        self.d0_opt_paths = [
            oe.contract_expression('k,yk->y', (N,), (A, N), optimize='auto-hq'),
            oe.contract_expression('k,l,ykl->y', (N,), (N,), (A, N, N), optimize='auto-hq'),
            oe.contract_expression('k,l,m,yklm->y', (N,), (N,), (N,), (A, N, N, N), optimize='auto-hq'),
        ]

        if self.Z_truncation_order >= 1:
            self.d1_opt_paths = [
                oe.contract_expression('k,yki->yi', (N,), (A, N, N), optimize='auto-hq'),
                oe.contract_expression('k,l,ykli->yi', (N,), (N,), (A, N, N, N), optimize='auto-hq'),
            ]

        if self.Z_truncation_order >= 2:
            self.d2_opt_paths = [
                oe.contract_expression('l,ylij->yij', (N,), (A, N, N, N), optimize='auto-hq'),
            ]

        return

    def _initialize_cal_dT_Tran_C_einsum_paths(self):
        """Store optimized paths as member of `self` for use in `_cal_dT_Tran_C`
        This function should eventually be replaced with someone more permanent.
        But for optimization for calculating Fe(CO)5 it is fine.
        """
        A, N = self.A, self.N

        self.cal_dT_Tran_C_paths = [
            oe.contract_expression('i,y->yi', (N,), (A,), optimize='auto-hq'),
            oe.contract_expression('i,yj->yij', (N,), (A, N,), optimize='auto-hq'),
            oe.contract_expression('i,yjk->yijk', (N,), (A, N, N,), optimize='auto-hq'),
        ]

        return

    def _initialize_cal_H_bar_tilde_einsum_paths(self):
        """Store optimized paths as member of `self` for use in `_cal_H_bar_tilde`
        This function should eventually be replaced with someone more permanent.
        But for optimization for calculating Fe(CO)5 it is fine.
        """
        A, N = self.A, self.N

        self.H_bar_tilde_paths = [
            oe.contract_expression('k,abk->ab', (N,), (A, A, N), optimize='auto-hq'),
            oe.contract_expression('k,l,abkl->ab', (N,), (N,), (A, A, N, N), optimize='auto-hq'),
            oe.contract_expression('k,abik->abi', (N,), (A, A, N, N), optimize='auto-hq'),
            oe.contract_expression('k,abki->abi', (N,), (A, A, N, N), optimize='auto-hq'),
        ]

        return

    def _initialize_compute_C_matrix_einsum_paths(self):
        """Store optimized paths as member of `self` for use in `_compute_C_matrix`
        This function should eventually be replaced with someone more permanent.
        But for optimization for calculating Fe(CO)5 it is fine.
        """
        A, N = self.A, self.N

        self.Cmat_Z0_opt_paths = [
            oe.contract_expression('k,xk->x', (N,), (A, N), optimize='auto-hq'),
        ]

        if self.Z_truncation_order >= 2:

            self.Cmat_Z2_opt_paths = [
                # s0
                oe.contract_expression('k,l,xkl->x', (N,), (N,), (A, N, N), optimize='auto-hq'),
                # s1
                oe.contract_expression('k,xik->xi', (N,), (A, N, N), optimize='auto-hq'),
                # no s2
            ]

        if self.Z_truncation_order >= 3:

            self.Cmat_Z3_opt_paths = [
                # s0
                oe.contract_expression('k,l,m,xklm->x', (N,), (N,), (N,), (A, N, N, N), optimize='auto-hq'),
                # s1
                oe.contract_expression('k,l,xikl->xi', (N,), (N,), (A, N, N, N), optimize='auto-hq'),
                # s2
                oe.contract_expression('k, xkij->xij', (N,), (A, N, N, N), optimize='auto-hq'),
            ]

        return

    def _initialize_hamiltonian(self, build_H=False):
        """ Express the Hamiltonian in a finite H.O. basis for comparing to FCI
        The basis size is defined by self.HO_size

        Explanation of the mapping between the code and the theory:
        H with a single creation operator is represented by
            - self.h[(1, 0)]
        H with a single anhilation operator is represented by
            - self.h[(0, 1)]
        """ 
        N, A = self.N, self.A  # for brevity

        # define coefficient tensors
        self.h = {
            (0, 0): np.zeros((A, A), dtype=complex),
            (0, 1): np.zeros((A, A, N), dtype=complex),
            (1, 0): np.zeros((A, A, N), dtype=complex),
            (1, 1): np.zeros((A, A, N, N), dtype=complex),
            (0, 2): np.zeros((A, A, N, N), dtype=complex),
            (2, 0): np.zeros((A, A, N, N), dtype=complex),
            
            # NEW 3 hamiltonian 
            (0, 3): np.zeros((A, A, N, N, N), dtype=complex),
            (3, 0): np.zeros((A, A, N, N, N), dtype=complex),
            (1, 2): np.zeros((A, A, N, N, N), dtype=complex),
            (2, 1): np.zeros((A, A, N, N, N), dtype=complex),
            
            # New 4 Hamiltonian
            (1, 3): np.zeros((A, A, N, N, N, N), dtype=complex),
            (3, 1): np.zeros((A, A, N, N, N, N), dtype=complex),
            (0, 4): np.zeros((A, A, N, N, N, N), dtype=complex),
            (4, 0): np.zeros((A, A, N, N, N, N), dtype=complex),
            (2, 2): np.zeros((A, A, N, N, N, N), dtype=complex),
        }
        # read in input parameters
        # --------------------------------------------------------------------------
        # specific prefactors for comparing with MCTDH
        # breakpoint()
        # This is the second quantization of the Hamiltonian from the Q-basis  
        if self.compare_with_mctdh:
            log.info("parameter of the Hamiltonian are entered properly!")

            # energy
            self.h[(0, 0)] += self.model[VMK.E].copy()

            # H.O. ground state energy
            for a in range(A):
                self.h[(0, 0)][a, a] += 0.5 * np.sum(self.model[VMK.w])

            log.info("zero point energy: {:.8f} ev".format(0.5 * np.sum(self.model[VMK.w])))

            # frequencies
            for a, j in it.product(range(A), range(N)):
                self.h[(1, 1)][a, a, j, j] += self.model[VMK.w][j].copy()

            # linear terms
            if self.trunc.at_least_linear:
                self.h[(0, 1)] += self.model[VMK.G1].copy() / np.sqrt(2)
                self.h[(1, 0)] += self.model[VMK.G1].copy() / np.sqrt(2)

            
            # quadratic terms
            if self.trunc.at_least_quadratic:
                # quadratic correction to H.O. ground state energy
                self.h[(0, 0)] += 0.5 * np.trace(self.model[VMK.G2], axis1=2, axis2=3)

                # quadratic terms
                self.h[(1, 1)] += self.model[VMK.G2].copy()
                self.h[(2, 0)] += 0.5*self.model[VMK.G2].copy() #CHANGE: could be 0.5
                self.h[(0, 2)] += 0.5*self.model[VMK.G2].copy() #CHANGE: could be 0.5
            
            
            # Cubic 
            if self.trunc.at_least_cubic:
                # breakpoint()
                self.h[(1,0)] += ((2*np.sqrt(2))**-1)*3*np.einsum('xyjkk -> xyj',self.model[VMK.G3].copy()) # figure out einsum 
                self.h[(0,1)] += ((2*np.sqrt(2))**-1)*3*np.einsum('xyjkk -> xyj',self.model[VMK.G3].copy())  

                self.h[(3,0)] += ((2*np.sqrt(2))**-1)*self.model[VMK.G3].copy()
                self.h[(0,3)] += ((2*np.sqrt(2))**-1)*self.model[VMK.G3].copy()
                
                self.h[(2,1)] += ((2*np.sqrt(2))**-1)*3*self.model[VMK.G3].copy()
                self.h[(1,2)] += ((2*np.sqrt(2))**-1)*3*self.model[VMK.G3].copy()
            
            # Quartic 
            if self.trunc.at_least_quartic:
                self.h[(0,0)] += 0.25*3*np.einsum('xyiijj -> xy',self.model[VMK.G4].copy())
                # breakpoint()
                self.h[(1,1)] += 0.25*12*np.einsum('xyijkk -> xyij',self.model[VMK.G4].copy())
                self.h[(2,0)] += 0.25*6*np.einsum('xyijkk -> xyij',self.model[VMK.G4].copy())
                self.h[(0,2)] += 0.25*6*np.einsum('xyijkk -> xyji',self.model[VMK.G4].copy())
                
                self.h[(4,0)] += 0.25*self.model[VMK.G4].copy()
                self.h[(0,4)] += 0.25*self.model[VMK.G4].copy()
                
                self.h[(3,1)] += 0.25*4*self.model[VMK.G4].copy()
                self.h[(1,3)] += 0.25*4*self.model[VMK.G4].copy()
                
                self.h[(2,2)] += 0.25*6*self.model[VMK.G4].copy()
            # breakpoinct()
            # Change coeeficents to match normal ordered form of the hamiltonian 
            if self.normal_order == True: 
                self.h[(0, 0)] *= 1
                self.h[(0, 1)] *= 1
                self.h[(1, 0)] *= 1
                self.h[(1, 1)] *= 1
                
                self.h[(0, 2)] *= 2
                self.h[(2, 0)] *= 2
                
                self.h[(0, 3)] *= 6
                self.h[(3, 0)] *= 6
                self.h[(1, 2)] *= 2
                self.h[(2, 1)] *= 2
                
                self.h[(1, 3)] *= 6
                self.h[(3, 1)] *= 6
                self.h[(0, 4)] *= 24
                self.h[(4, 0)] *= 24
                self.h[(2, 2)] *= 4            
                # breakpoint()
        else:
            Exception("You shouldn't reach this point")
                
        return
    
    def single_mode_q_CI_basis(self,HO_basis_size):
        Q = np.zeros([HO_basis_size, HO_basis_size], dtype=complex)
        for a_1, b_1 in it.product(range(HO_basis_size),range(HO_basis_size)):
            index = (a_1,b_1)
            
            if a_1 == b_1 + (1):
                Q[index] += np.sqrt(b_1 + 1)
            
            if a_1 == b_1 + (-1):
                Q[index] += np.sqrt(b_1 + 0)            
        
        Q *= 1/np.sqrt(2)  
        # breakpoint()

        return Q

    def single_mode_q2_CI_basis(self,HO_basis_size):
        Q2 = np.zeros([HO_basis_size, HO_basis_size], dtype=complex)
        for a_1, b_1 in it.product(range(HO_basis_size),range(HO_basis_size)):
            index = (a_1,b_1)
            
            if a_1 == b_1 + (2):
                Q2[index] += np.sqrt(b_1 + 1)*np.sqrt(b_1 + 2)
            
            if a_1 == b_1 + (-2):
                Q2[index] += np.sqrt(b_1 + 0)*np.sqrt(b_1 - 1)
            
            if a_1 == b_1 + (0):
                Q2[index] += 2*b_1+1            
        
        Q2 *= 1/(2)  
        # breakpoint()

        return Q2

    # def acf_eigenvalue_ci(self,eigvals,intensities):
    #     hbar = 0.658211
    #     finaltime = 100
    #     stepsize = 0.1
        
    #     acl_time = np.arange(0,finaltime+stepsize,stepsize)
        
    #     # Transform q to the Hamiltonian basis 
    #     # q_static = eigvects.T@q_op_in_HO@eigvects 
    #     # amplitudes = eigvects[0,:]
        
    #     phase = np.exp(-1j*np.outer(eigvals,acl_time)/hbar)
    #     acf = intensities[:,None]*phase
        
    #     # time_dependent_wave_function[-1] *= 1/np.sqrt(2)
    #     # breakpoint()
    #     # exp_val = np.einsum('kt,kl,lt->t', time_dependent_wave_function.conj(),
    #     #                             q_static, time_dependent_wave_function)
            
    #     return(acl_time,acf)
    
    def single_mode_q_exp_val(self,eigvals,eigvects,q_op_in_HO,stepsize,finaltime):
        # hbar = 0.658211
        hbar= 0.6582119569
        acl_time = np.arange(0,finaltime+stepsize*2,stepsize*2)
        # breakpoint()
        # Transform q to the Hamiltonian basis 
        q_static = eigvects.T@q_op_in_HO@eigvects 
        amplitudes = eigvects[0,:]
        
        phase = np.exp(-1j*np.outer(eigvals,acl_time)/hbar)
        time_dependent_wave_function = amplitudes[:,None]*phase
        # time_dependent_wave_function[-1] *= 1/np.sqrt(2)
        # breakpoint()
        exp_val = np.einsum('kt,kl,lt->t', time_dependent_wave_function.conj(),
                                    q_static, time_dependent_wave_function)
        # breakpoint()
        return(acl_time,exp_val)

    def single_mode_q2_exp_val(self,eigvals,eigvects,q2_op_in_HO,stepsize,finaltime):
        # hbar = 0.658211
        hbar = 0.6582119569 
        acl_time = np.arange(0,finaltime+stepsize*2,stepsize*2)
        # breakpoint()
        # Transform q to the Hamiltonian basis 
        q_static = eigvects.T@q2_op_in_HO@eigvects 
        amplitudes = eigvects[0,:]
        
        phase = np.exp(-1j*np.outer(eigvals,acl_time)/hbar)
        time_dependent_wave_function = amplitudes[:,None]*phase
        # time_dependent_wave_function[-1] *= 1/np.sqrt(2)
        # breakpoint()
        exp_val = np.einsum('kt,kl,lt->t', time_dependent_wave_function.conj(),
                                    q_static, time_dependent_wave_function)
        # breakpoint()
        return(acl_time,exp_val)

    def ED_subroutine(self,HO_basis_size,tfinal=10,stepsize=.1,subspace_calc=False,q_eval=False):
        
        # Single mode results 
        if self.N == 1:
            H_mat_HO_basis = self._single_mode_FCI_exact_standard_hamiltonian(HO_basis_size)
        
        # Double mode results
        if self.N == 2:
            H_mat_HO_basis = self._double_mode_FCI_exact_standard_hamiltonian(HO_basis_size)
            breakpoint()
            if self.trunc.at_least_quadratic:
                hamiltonian_order = 2
            if self.trunc.at_least_quartic:
                hamiltonian_order = 4
            if subspace_calc:
                H_mat_HO_basis = self._project_subspace(HO_basis_size,H_mat_HO_basis,hamiltonian_order)
                        
        # Create eigval and eigvec data for acf  
        eigvals,eigvecs = np.linalg.eigh(H_mat_HO_basis) 
        # Create ACF
        time, acf,intensities = self._ed_acf(tfinal,stepsize,eigvecs,eigvals)
        
        if self.N == 1 and q_eval:
           Q_mat = self.single_mode_q_CI_basis(HO_basis_size)
           Q2_mat = self.single_mode_q2_CI_basis(HO_basis_size)
           eig_q_results, exact_animation = self.wavefunction_evolution_dvr(Q_mat,H_mat_HO_basis,tfinal,stepsize)
           time,Q_exp_val = self.single_mode_q_exp_val(eigvals,eigvecs,Q_mat,stepsize,finaltime=tfinal)
           time,Q2_exp_val = self.single_mode_q2_exp_val(eigvals,eigvecs,Q2_mat,stepsize,finaltime=tfinal)
           time,H_exp_val,ex_norm = self.exact_energy_norm(eigvals,eigvecs,H_mat_HO_basis,stepsize,finaltime=tfinal)
           return(eigvals,time,acf,Q_exp_val,Q2_exp_val,H_mat_HO_basis,intensities, eig_q_results, exact_animation,
                  H_exp_val,ex_norm)
     
        return(eigvals,time,acf,H_mat_HO_basis,intensities)
    
    def exact_energy_norm(self,energy_evals,energy_evecs,H_mat,stepsize,finaltime):
        hbar = 0.6582119569 
        acl_time = np.arange(0,finaltime+stepsize*2,stepsize*2)
        
        # ground harmonic oscilator state in eigenbasis
        C_E_0 = energy_evecs[0,:]
        
        # C_E_0[0] = 1
        # *0 # ground state of hamiltonian in Energy basis
        
        exact_exp_H = []
        exact_norm = []
        for t in acl_time:
            psi_t = np.exp(-1j*energy_evals*t/hbar)*C_E_0
            # transform into HO basis
            psi_t_HO = energy_evecs@psi_t
            exact_norm_t = np.vdot(psi_t_HO,psi_t_HO)
            exact_exp_H_t = psi_t_HO.conj()@H_mat@psi_t_HO
            exact_exp_H.append(exact_exp_H_t)
            exact_norm.append(exact_norm_t)
             
        
        return(acl_time,np.array(exact_exp_H),np.array(exact_norm))

    def _single_mode_FCI_exact_standard_hamiltonian(self,HO_basis_size):
        H = np.zeros([HO_basis_size, HO_basis_size], dtype=complex)
        for a_1, b_1 in it.product(range(HO_basis_size),range(HO_basis_size)):
            index = (a_1,b_1)

            if self.trunc.at_least_linear:
                # hamiltonian terms associated with H(1, 0)
                if a_1 == b_1 + (1):
                    H[index] += self.h[(1, 0)][0,0,0]*np.sqrt(b_1 + 1)
                # hamiltonian terms associated with H(0, 1)
                if a_1 == b_1 + (-1):
                    H[index] += self.h[(0, 1)][0,0,0]*np.sqrt(b_1 + 0)            
                # hamiltonian terms associated with H(1, 1), H(0, 0) 
                if a_1 == b_1 + (0):

                    H[index] += self.h[(0,0)][0,0] 
                    H[index] += 1*self.h[(1, 1)][0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0)
            # breakpoint()
            if self.trunc.at_least_quadratic:
                
                # hamiltonian terms associated with H(2, 0)
                if a_1 == b_1 + (2):
                    H[index] += 0.5*self.h[(2, 0)][0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)
                # hamiltonian terms associated with H(0, 2)
                if a_1 == b_1 + (-2):
                    H[index] += 0.5*self.h[(0, 2)][0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1)
            
            if self.trunc.at_least_cubic:
                # hamiltonian terms associated with H(3, 0)
                if a_1 == b_1 + (3):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][0,0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3)
                # hamiltonian terms associated with H(0, 3)
                if a_1 == b_1 + (-3):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2)
                # hamiltonian terms associated with H(1, 2)
                if a_1 == b_1 + (-1):
                    H[index] += 0.5*self.h[(1, 2)][0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1)
                # hamiltonian terms associated with H(2, 1)
                if a_1 == b_1 + (1):
                    H[index] += 0.5*self.h[(2, 1)][0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1)
            
            if self.trunc.at_least_quartic:
                # hamiltonian terms associated with H(4, 0)
                if a_1 == b_1 + (4):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][0,0,0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_1 + 4)
                # hamiltonian terms associated with H(0, 4)
                if a_1 == b_1 + (-4):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -3)
                # hamiltonian terms associated with H(2, 2)
                if a_1 == b_1 + (0):
                    H[index] += 0.25*self.h[(2, 2)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * (b_1 + -1) * np.sqrt(b_1 + 0)
                # hamiltonian terms associated with H(1, 3)
                if a_1 == b_1 + (-2):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -2)
                # hamiltonian terms associated with H(3, 1)
                if a_1 == b_1 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)

        self.single_mode_exact_eigvals = np.linalg.eigvalsh(H)
        vals,vecs = np.linalg.eigh(H)
         
        return(H)


    def _single_mode_FCI_exact_standard_hamiltonian_additional(self,H_dict,HO_basis_size):
        H = np.zeros([HO_basis_size, HO_basis_size], dtype=complex)
        for a_1, b_1 in it.product(range(HO_basis_size),range(HO_basis_size)):
            index = (a_1,b_1)

            if self.trunc.at_least_linear:
                # hamiltonian terms associated with H(1, 0)
                if a_1 == b_1 + (1):
                    H[index] += H_dict[(1, 0)][0,0,0]*np.sqrt(b_1 + 1)
                # hamiltonian terms associated with H(0, 1)
                if a_1 == b_1 + (-1):
                    H[index] += H_dict[(0, 1)][0,0,0]*np.sqrt(b_1 + 0)            
                # hamiltonian terms associated with H(1, 1), H(0, 0) 
                if a_1 == b_1 + (0):

                    H[index] += H_dict[(0,0)][0,0] 
                    H[index] += 1*H_dict[(1, 1)][0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0)
            # breakpoint()
            if self.trunc.at_least_quadratic:
                
                # hamiltonian terms associated with H(2, 0)
                if a_1 == b_1 + (2):
                    H[index] += 0.5*H_dict[(2, 0)][0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)
                # hamiltonian terms associated with H(0, 2)
                if a_1 == b_1 + (-2):
                    H[index] += 0.5*H_dict[(0, 2)][0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1)
            
            if self.trunc.at_least_cubic:
                # hamiltonian terms associated with H(3, 0)
                if a_1 == b_1 + (3):
                    H[index] += 0.16666666666666666*H_dict[(3, 0)][0,0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3)
                # hamiltonian terms associated with H(0, 3)
                if a_1 == b_1 + (-3):
                    H[index] += 0.16666666666666666*H_dict[(0, 3)][0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2)
                # hamiltonian terms associated with H(1, 2)
                if a_1 == b_1 + (-1):
                    H[index] += 0.5*H_dict[(1, 2)][0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1)
                # hamiltonian terms associated with H(2, 1)
                if a_1 == b_1 + (1):
                    H[index] += 0.5*H_dict[(2, 1)][0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1)
            
            if self.trunc.at_least_quartic:
                # hamiltonian terms associated with H(4, 0)
                if a_1 == b_1 + (4):
                    H[index] += 0.041666666666666664*H_dict[(4, 0)][0,0,0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_1 + 4)
                # hamiltonian terms associated with H(0, 4)
                if a_1 == b_1 + (-4):
                    H[index] += 0.041666666666666664*H_dict[(0, 4)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -3)
                # hamiltonian terms associated with H(2, 2)
                if a_1 == b_1 + (0):
                    H[index] += 0.25*H_dict[(2, 2)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * (b_1 + -1) * np.sqrt(b_1 + 0)
                # hamiltonian terms associated with H(1, 3)
                if a_1 == b_1 + (-2):
                    H[index] += 0.16666666666666666*H_dict[(1, 3)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -2)
                # hamiltonian terms associated with H(3, 1)
                if a_1 == b_1 + (2):
                    H[index] += 0.16666666666666666*H_dict[(3, 1)][0,0,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)

        self.single_mode_exact_eigvals = np.linalg.eigvalsh(H)
        vals,vecs = np.linalg.eigh(H)
         
        return(H)

    
    def project_quad_subspace(self,HO_basis_size):
        # on valid for 2 mode system 
        num_elem = HO_basis_size**4

        a = np.arange(num_elem).reshape(HO_basis_size,HO_basis_size,HO_basis_size,HO_basis_size)
        b= a.reshape(HO_basis_size**2,HO_basis_size**2)

        # determine index mapping 
        # store pairs in flat index
        flat_indexes = [] 
        for n_i in range(HO_basis_size):
            for n_j in range(HO_basis_size):
                two_tensor = (n_i,n_j)
                flat_indexes.append(two_tensor)

        # determine pairs in two tensor which match condition
        truth_vector = []
        for flat_index in flat_indexes:
            if sum(flat_index) > 2:
                truth_vector.append(0)
            else:
                truth_vector.append(1)    
        # print(np.count_nonzero(truth_vector))
        
        # build rectangular matrix for transformation
        hilbert_dim = len(truth_vector)
        subspace_dim = np.count_nonzero(truth_vector)

        # Create transformation matrix
        U = np.zeros((hilbert_dim,subspace_dim))

        # Loop through truth vector create list of the indexes where true
        true_indexes = []
        for idx, truth_claim in enumerate(truth_vector):
            if truth_claim == True:
                true_indexes.append(idx)
        basis = []
        for idx in true_indexes:
            basis.append(flat_indexes[idx])
        # breakpoint()
        # Create subspace transform 
        for idx in range(subspace_dim):
            U[true_indexes[idx],idx] = 1    
        
        subspace_hamiltonian = U.T@self.hamiltonian_2_mode@U
        self.subspace_hamiltonian = subspace_hamiltonian.copy()
        self.eigvals_2_mode, self.eigvecs_2_mode = np.linalg.eigh(subspace_hamiltonian)
        
        # self.eigvals_2_mode, self.eigvecs_2_mode = np.linalg.eigh(self.hamiltonian_2_mode)
        self.intensities = np.abs(self.eigvecs_2_mode.T[:,0])**2   
        # breakpoint()
        return
    
    def _project_subspace(self,HO_basis_size,Hamiltonian,ham_order):
        # on valid for 2 mode system 
        num_elem = HO_basis_size**4

        a = np.arange(num_elem).reshape(HO_basis_size,HO_basis_size,HO_basis_size,HO_basis_size)
        b= a.reshape(HO_basis_size**2,HO_basis_size**2)

        # determine index mapping 
        # store pairs in flat index
        flat_indexes = [] 
        for n_i in range(HO_basis_size):
            for n_j in range(HO_basis_size):
                two_tensor = (n_i,n_j)
                flat_indexes.append(two_tensor)

        # determine pairs in two tensor which match condition
        truth_vector = []
        for flat_index in flat_indexes:
            if sum(flat_index) > ham_order:
                truth_vector.append(0)
            else:
                truth_vector.append(1)    
        # print(np.count_nonzero(truth_vector))
        
        # build rectangular matrix for transformation
        hilbert_dim = len(truth_vector)
        subspace_dim = np.count_nonzero(truth_vector)

        # Create transformation matrix
        U = np.zeros((hilbert_dim,subspace_dim))

        # Loop through truth vector create list of the indexes where true
        true_indexes = []
        for idx, truth_claim in enumerate(truth_vector):
            if truth_claim == True:
                true_indexes.append(idx)
        basis = []
        for idx in true_indexes:
            basis.append(flat_indexes[idx])
        # breakpoint()
        # Create subspace transform 
        for idx in range(subspace_dim):
            U[true_indexes[idx],idx] = 1    
        
        subspace_hamiltonian = U.T@Hamiltonian@U
        
        return(subspace_hamiltonian)
    
    def _double_mode_FCI_exact_standard_hamiltonian(self,HO_basis_size):
        
        H = np.zeros([HO_basis_size, HO_basis_size, HO_basis_size, HO_basis_size], dtype=complex)
        # breakpoint()

        for a_1, a_2, b_1, b_2 in it.product(
            range(HO_basis_size), range(HO_basis_size), range(HO_basis_size), range(HO_basis_size)
        ):
            index = (a_1, a_2, b_1, b_2)
            # linear
            if self.trunc.at_least_linear:
                if a_1 == b_1 and a_2 == b_2:
                    H[index] += self.h[(0,0)] ##########    
                # hamiltonian terms associated with H(1, 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (0):
                    H[index] += self.h[(1, 0)][:,:,0]*np.sqrt(b_1 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (1):
                    H[index] += self.h[(1, 0)][:,:,1]*np.sqrt(b_2 + 1)
                # hamiltonian terms associated with H(0, 1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (0):
                    H[index] += self.h[(0, 1)][:,:,0]*np.sqrt(b_1 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-1):
                    H[index] += self.h[(0, 1)][:,:,1]*np.sqrt(b_2 + 0)
                # hamiltonian terms associated with H(1, 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 1*self.h[(1, 1)][:,:,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-1):
                    H[index] += 1*self.h[(1, 1)][:,:,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (1):
                    H[index] += 1*self.h[(1, 1)][:,:,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 1*self.h[(1, 1)][:,:,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
            
            # qaudratic
            if self.trunc.at_least_quadratic:       
                # hamiltonian terms associated with H(2, 0)
                if a_1 == b_1 + (2) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(2, 0)][:,:,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.5*self.h[(2, 0)][:,:,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.5*self.h[(2, 0)][:,:,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (2):
                    H[index] += 0.5*self.h[(2, 0)][:,:,1,1]*np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                # hamiltonian terms associated with H(0, 2)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(0, 2)][:,:,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.5*self.h[(0, 2)][:,:,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.5*self.h[(0, 2)][:,:,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-2):
                    H[index] += 0.5*self.h[(0, 2)][:,:,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                # breakpoint()
            # cubic 
            if self.trunc.at_least_cubic:
                # hamiltonian terms associated with H(3, 0)
                if a_1 == b_1 + (3) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3)
                if a_1 == b_1 + (2) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,0,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,0,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (2) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,1,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,1,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,1,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (0) and a_2 == b_2 + (3):
                    H[index] += 0.16666666666666666*self.h[(3, 0)][:,:,1,1,1]*np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3)
                # hamiltonian terms associated with H(0, 3)
                if a_1 == b_1 + (-3) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,0,1,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-3):
                    H[index] += 0.16666666666666666*self.h[(0, 3)][:,:,1,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2)
                # hamiltonian terms associated with H(1, 2)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(1, 2)][:,:,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-1):
                    H[index] += 0.5*self.h[(1, 2)][:,:,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-1):
                    H[index] += 0.5*self.h[(1, 2)][:,:,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-2):
                    H[index] += 0.5*self.h[(1, 2)][:,:,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (1):
                    H[index] += 0.5*self.h[(1, 2)][:,:,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(1, 2)][:,:,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(1, 2)][:,:,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-1):
                    H[index] += 0.5*self.h[(1, 2)][:,:,1,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -1)
                # hamiltonian terms associated with H(2, 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(2, 1)][:,:,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (-1):
                    H[index] += 0.5*self.h[(2, 1)][:,:,0,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (1):
                    H[index] += 0.5*self.h[(2, 1)][:,:,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(2, 1)][:,:,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (1):
                    H[index] += 0.5*self.h[(2, 1)][:,:,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (0):
                    H[index] += 0.5*self.h[(2, 1)][:,:,1,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (2):
                    H[index] += 0.5*self.h[(2, 1)][:,:,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (0) and a_2 == b_2 + (1):
                    H[index] += 0.5*self.h[(2, 1)][:,:,1,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1)
            
            # quartic
            if self.trunc.at_least_quartic: 
                # hamiltonian terms associated with H(4, 0)
                if a_1 == b_1 + (4) and a_2 == b_2 + (0):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_1 + 4)
                if a_1 == b_1 + (3) and a_2 == b_2 + (1):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,0,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (3) and a_2 == b_2 + (1):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,0,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (2):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (3) and a_2 == b_2 + (1):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,1,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (2):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,1,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (2) and a_2 == b_2 + (2):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,1,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (3):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,0,1,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3)
                if a_1 == b_1 + (3) and a_2 == b_2 + (1):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,0,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (2):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,0,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (2) and a_2 == b_2 + (2):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,0,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (3):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3)
                if a_1 == b_1 + (2) and a_2 == b_2 + (2):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,1,0,0]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (3):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,1,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3)
                if a_1 == b_1 + (1) and a_2 == b_2 + (3):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,1,1,0]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3)
                if a_1 == b_1 + (0) and a_2 == b_2 + (4):
                    H[index] += 0.041666666666666664*self.h[(4, 0)][:,:,1,1,1,1]*np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3) * np.sqrt(b_2 + 4)
                # hamiltonian terms associated with H(0, 4)
                if a_1 == b_1 + (-4) and a_2 == b_2 + (0):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -3)
                if a_1 == b_1 + (-3) and a_2 == b_2 + (-1):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-3) and a_2 == b_2 + (-1):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-2):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,0,1,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-3) and a_2 == b_2 + (-1):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-2):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-2):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-3):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,0,1,1,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2)
                if a_1 == b_1 + (-3) and a_2 == b_2 + (-1):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-2):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-2):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-3):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,0,1,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (-2):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-3):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-3):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-4):
                    H[index] += 0.041666666666666664*self.h[(0, 4)][:,:,1,1,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2) * np.sqrt(b_2 + -3)
                # hamiltonian terms associated with H(2, 2)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,0,0,0]*np.sqrt(b_1 + 0) * (b_1 + -1) * np.sqrt(b_1 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (2) and a_2 == b_2 + (-2):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,0,1,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (2):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (1):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (0):
                    H[index] += 0.25*self.h[(2, 2)][:,:,1,1,1,1]*np.sqrt(b_2 + 0) * (b_2 + -1) * np.sqrt(b_2 + 0)
                # hamiltonian terms associated with H(1, 3)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -2)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,0,1,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (-3):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,0,1,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2)
                if a_1 == b_1 + (-3) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,0,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,0,1,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,1,0,1]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (-2):
                    H[index] += 0.16666666666666666*self.h[(1, 3)][:,:,1,1,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + -1) * np.sqrt(b_2 + -2) * np.sqrt(b_2 + -2)
                # hamiltonian terms associated with H(3, 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)
                if a_1 == b_1 + (3) and a_2 == b_2 + (-1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,0,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,1,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,0,1,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,0,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (2) and a_2 == b_2 + (0):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,0,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0)
                if a_1 == b_1 + (0) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,0,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,0,1,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (0) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,1,0,0]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)
                if a_1 == b_1 + (1) and a_2 == b_2 + (1):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,1,0,1]*np.sqrt(b_1 + 1) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1)
                if a_1 == b_1 + (-1) and a_2 == b_2 + (3):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,1,1,0]*np.sqrt(b_1 + 0) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2) * np.sqrt(b_2 + 3)
                if a_1 == b_1 + (0) and a_2 == b_2 + (2):
                    H[index] += 0.16666666666666666*self.h[(3, 1)][:,:,1,1,1,1]*np.sqrt(b_2 + 0) * np.sqrt(b_2 + 0) * np.sqrt(b_2 + 1) * np.sqrt(b_2 + 2)

        H = H.reshape(HO_basis_size ** 2,HO_basis_size ** 2)
        self.hamiltonian_2_mode = H.copy()  
        # self.exact_eigvals = np.linalg.eigvalsh(H)
        # self.exact_eigval_idx = np.arange(len(self.exact_eigvals))
        # vals,vecs = np.linalg.eigh(H)
        # dicts = {}
        # dicts[(0,0)] = vecs[:,1][0]
        # dicts[(1,0)] = vecs[:,1][50]
        # dicts[(0,1)] = vecs[:,1][1]
        # dicts[(1,1)] = vecs[:,1][51]
        # dicts[(0,2)] = vecs[:,1][2]
        # dicts[(2,0)] = vecs[:,1][50*2]
        
        # den_mat, d_low, d_upp = self.build_reduced_density_matrix( vecs[:,0],HO_basis_size)
        # self.hamiltonian_2_mode = H
        # breakpoint()
        return(H)
    
    
    def wavefunction_evolution_dvr(self,Q,H,tfinal,stepsize,plotting_axis=[-5,10,-2.5,5]):
        energy,H_basis_mat = np.linalg.eigh(H)
        q_grid,q_basis_mat = np.linalg.eigh(Q)
        c_E_HO_0 = H_basis_mat[0,:] # HO vaccum state in Energy basis
        
        C_E_0 = H_basis_mat[0,:]*0 # ground state of hamiltonian in Energy basis
        C_E_0[0] = 1 
        
        # Construct potential
        V_q = q_grid*0
        V_q += self.model[VMK.E].flatten() 
        if self.trunc.at_least_linear:
            V_q += q_grid*(self.model[VMK.G1].flatten())
        if self.trunc.at_least_quadratic:
            V_q += (q_grid**2)*(self.model[VMK.G2].flatten() + 0.5*self.model[VMK.w])
        if self.trunc.at_least_cubic:
            V_q += (q_grid**3)*(self.model[VMK.G3].flatten())
        if self.trunc.at_least_quartic:
            V_q += (q_grid**4)*(self.model[VMK.G4].flatten())
                
        hbar =  0.658211        
        # stepsize = .0001
        prop_time = np.arange(0,tfinal,stepsize)
        # [np.vdot(psi, psi).real for psi in psi_p]
        psi_times = [] 
        psi_p = []
        for t in prop_time:
            psi_E_t = np.exp((-1j*t*energy)/hbar)*c_E_HO_0
            psi_HO_t = H_basis_mat@psi_E_t
            psi_Q_t = q_basis_mat.T@psi_HO_t 
            psi_density =np.abs(psi_Q_t)**2
            psi_times.append(psi_density)
            psi_p.append(psi_Q_t)
        
        
        H_eig_expect_qs =[]
        for i in range(len(energy)):
            H_eig_expect_q = (H_basis_mat[:,i].T)@Q@(H_basis_mat[:,i])
            H_eig_expect_qs.append(H_eig_expect_q)
        
        q_energy_eigenstate = pd.DataFrame({'Eigenstate_energy':energy,'Eigenstate mean position':H_eig_expect_qs})
        
               
        
        psi_times_arr = np.array(psi_times)  # (N_t, N_q)
        V_scaled = V_q #/ V_q.max() * psi_times_arr.max() * 0.8

        fig, ax = plt.subplots()

        # Left axis: probability density
        line, = ax.plot(q_grid, psi_times_arr[0], lw=2, color="C0", label=r"$|\psi|^2$")
        ax.set_xlabel("q")
        ax.set_ylabel(r"$|\psi(q,t)|^2$", color="C0")
        ax.tick_params(axis='y', labelcolor="C0")
        
        # Right axis: potential
        axV = ax.twinx()
        axV.plot(q_grid, V_scaled, color="C1", ls="--", label="V(q)")
        axV.set_ylabel("V(q)", color="C1")
        axV.tick_params(axis='y', labelcolor="C1")
        
        axV.set_xlim(plotting_axis[0],plotting_axis[1])
        axV.set_ylim(plotting_axis[2],plotting_axis[3])
        
        time_text = ax.text(0.95, 0.9, "", transform=ax.transAxes,
                            ha="right", va="top", color="k")

        def update(frame):
            line.set_ydata(psi_times_arr[frame])
            time_text.set_text(f"t = {prop_time[frame]:.3f}")
            return line, time_text
        
        import matplotlib.animation as anim
        exact_animation = anim.FuncAnimation(fig, update,frames=len(prop_time),interval=100,blit=False)
        writer = anim.PillowWriter(fps=20)   # choose fps
        
        plt.tight_layout()
        plt.show()
        # return animation object and data 
        
        return(q_energy_eigenstate,exact_animation)
    
    def project_subspace(self,HO_basis_size):
        # only valid for 2 mode system 
        num_elem = HO_basis_size**4

        a = np.arange(num_elem).reshape(HO_basis_size,HO_basis_size,HO_basis_size,HO_basis_size)
        b= a.reshape(HO_basis_size**2,HO_basis_size**2)

        # determine index mapping 
        # store pairs in flat index
        flat_indexes = [] 
        for n_i in range(HO_basis_size):
            for n_j in range(HO_basis_size):
                two_tensor = (n_i,n_j)
                flat_indexes.append(two_tensor)

                
        # determine pairs in two tensor which match condition
        truth_vector = []
        for flat_index in flat_indexes:
            if sum(flat_index) > 4:
                truth_vector.append(0)
            else:
                truth_vector.append(1)    
        
        # build rectangular matrix for transformation
        hilbert_dim = len(truth_vector)
        subspace_dim = np.count_nonzero(truth_vector)

        # Create transformation matrix
        U = np.zeros((hilbert_dim,subspace_dim))

        # Loop through truth vector create list of the indexes where true
        true_indexes = []
        for idx, truth_claim in enumerate(truth_vector):
            if truth_claim == True:
                true_indexes.append(idx)
        basis = []
        for idx in true_indexes:
            basis.append(flat_indexes[idx])
        
        # Create subspace transform 
        for idx in range(subspace_dim):
            U[true_indexes[idx],idx] = 1    
        
        subspace_hamiltonian = U.T@self.hamiltonian_2_mode@U
        self.subspace_hamiltonian = subspace_hamiltonian.copy()
        self.eigvals_2_mode, self.eigvecs_2_mode = np.linalg.eigh(subspace_hamiltonian)
        
        self.intensities = np.abs(self.eigvecs_2_mode.T[:,0])**2   
        
        return
    
    def _ed_acf(self,finaltime,stepsize,eigvecs,eigvals):
        
        hbar = .6582119569
        time = np.arange(0,finaltime+stepsize*2,stepsize*2) #*1E-15
        ed_acf = np.zeros_like(time, dtype=complex)
        intensities = eigvecs[0,:]**2
        
        for i, t in enumerate(time):
            ed_acf[i] = np.sum(intensities * np.exp(-1j*(eigvals/hbar) * t))
        
        
        return(time,ed_acf,intensities)
     
            
    def _construct_debug_hamiltonian_for_FCI(self):
        """ Should check if this function needs to be removed
        or reworked in the future?
        """

        shape = [self.A, self.HO_size**2, self.A, self.HO_size**2]
        H = np.zeros(shape, dtype=complex)

        for a, b in it.product(range(self.A), repeat=2):
            log.info(f"Doing SOS with ({a}, {b})")

            # setup the Hamiltonian
            model = model_two_mode(
                self.h[(0, 0)][a, b],
                self.h[(0, 1)][a, b, :],
                self.h[(1, 1)][a, b, :],
                self.h[(2, 0)][a, b, :]
            )

            # compute the SOS solution
            H[a, :, b][:] = model.sos_solution(HO_size=self.HO_size)

        # reshape into a matrix
        self.H = H.reshape(self.A * self.HO_size ** 2, self.A * self.HO_size ** 2)
        return

    def apply_time_step_conversion(self, step_size):
        """Change the units of each step to be in eV/fs"""

        # use consistent units of energy in eV, time in fs
        hbar_per_eV = 6.582119569e-16
        # 
        s_per_fs = 1e-15

        # apply the conversion
        new_step_size = step_size / hbar_per_eV * s_per_fs

        if self.wavenumber_units == True:
            hbar_per_wavenumber =  5.308837495683545e-12
            new_step_size = step_size/hbar_per_wavenumber *s_per_fs
            
            
        return new_step_size
    
    def remove_time_step_conversion(self, time_step_array):
        """Remove the units for each time step so that it can be compared to SOS"""

        # use consistent units of energy in eV, time in fs
        hbar_per_eV = 6.582119569e-16
        s_per_fs = 1e-15

        if self.wavenumber_units == True:
            hbar_per_wavenumber =  5.308837495683545e-12
            time_step_array *=  hbar_per_wavenumber/s_per_fs
        
        # remove the conversion
        else:
            time_step_array *= hbar_per_eV / s_per_fs
        
        
        if type(time_step_array) is not np.ndarray:
            # if we were in fact given a number then return it
            return time_step_array

        # otherwise we modified the array by-reference
        return

    def _postprocess_rk45_integration_results(self, sol, debug=False):
        """ extract the relevant information from the integrator object `sol` """
        # breakpoint()
        # number of integration steps accepted by the integrator
        
        log.info(f"RK45 preformed {len(self.C_tau_ABS)} integration calculations.")
        log.info(f"RK45 accepted  {len(sol.t)} of those as solutions to the ode's.")
        if debug:
            log.debug(f"Distance we reached when we stopped: {sol.t_events[0]}")

        # Copy the time value arrays
        self.t_cc = sol.t.copy()
        # log.info(len(self.t_cc))
        # log.info(len(self.t_cc_store))

        # shift the time back to units of fs
        self.remove_time_step_conversion(self.t_cc)

        # initialize the arrays to store the autocorrelation function
        self.C_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
        # self.C_tau_cc_ECD = np.zeros_like(self.t_cc, dtype=complex)

        # log.info(len(self.C_tau_cc))
        # log.info(len(self.C_tau_cc_store))

        # only extract the values which correspond to time steps in the solution
        # since we save C(t) for all integration steps, but only some are accepted
        if self.correlation_funcs:
            # self.G_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
            self.q_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
            self.q_2_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
            self.exp_H_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
            self.norm_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
            
            # G_dic_ABS = {c[0]: c[1] for c in self.G0_tau_abs} 
            q_dic_ABS = {c[0]: c[1] for c in self.exp_q}
            q_2_dic_ABS = {c[0]: c[1] for c in self.exp_q2}
            exp_H_dic_ABS = {c[0]: c[1] for c in self.exp_H}
            norm_dic_ABS = {c[0]: c[1] for c in self.exp_norm}
            # breakpoint()
             
            # if self.Z_truncation_order >= 1:
            #     self.Z1_storage
            # if self.Z_truncation_order >= 2:
            #     self.Z1_storage
            # if self.Z_truncation_order >= 3:
            #     self.Z1_storage
            # if self.Z_truncation_order >= 4:
            #     self.Z1_storage
        
        if self.N == 1:
            self.T1_tau_cc_ABS = np.zeros_like(self.t_cc, dtype=complex)
            T1_dic_ABS = {c[0]: c[1] for c in self.T1_tau}    
        
        self.T1_tau_cc = np.zeros((len(self.t_cc),self.N), dtype=complex)
        T1_dict = {c[0]: c[1] for c in self.T1_storage}
        C_dic_ABS = {c[0]: c[1] for c in self.C_tau_ABS}
        # breakpoint()
        for idx, t in enumerate(sol.t):
            self.C_tau_cc_ABS[idx] = C_dic_ABS[t]
            self.T1_tau_cc[idx] = T1_dict[t]
            
            if self.correlation_funcs:
                # self.G_tau_cc_ABS[idx] = G_dic_ABS[t] 
                if self.N == 1:
                    self.T1_tau_cc_ABS[idx] = T1_dic_ABS[t] 
                    self.q_tau_cc_ABS[idx] = q_dic_ABS[t] 
                    self.q_2_tau_cc_ABS[idx] = q_2_dic_ABS[t]
                self.exp_H_tau_cc_ABS[idx] = exp_H_dic_ABS[t]
                self.norm_tau_cc_ABS[idx]  = norm_dic_ABS[t]
                
                
                

        # C_dic_ECD = {c[0]: c[1] for c in self.C_tau_ECD}
        # for idx, t in enumerate(sol.t):
        #     self.C_tau_cc_ECD[idx] = C_dic_ECD[t]

        # store norm of the wave function
        # self.Norm_cc = np.zeros([len(self.t_cc), self.A], dtype=complex)

        # N_dic = {N[0]: N[1] for N in self.Norm}
        # for idx, t in enumerate(sol.t):
        #     self.Norm_cc[idx, :] = N_dic[t]

        if self.calculate_population_flag:
            # store the diabatic populations --------------------
            self.state_pop_cc_DB = np.zeros([len(self.t_cc), self.A, self.A], dtype=complex)

            P_dic = {P[0]: P[1] for P in self.State_pop_DB}

            for idx, t in enumerate(sol.t):
                self.state_pop_cc_DB[idx, :] += P_dic[t]

            # store diabatic state population
            dic_diab_pop = {}
            dic_diab_pop["time(fs)"] = self.t_cc

            for x, y in it.product(range(self.A), repeat=2):
                name = f"diab state pop {x} init state {y}"
                dic_diab_pop[name] = self.state_pop_cc_DB[:, x, y].real

            import pandas as pd
            df = pd.DataFrame(dic_diab_pop)
            df.to_csv(f"Diabatic_state_pop_{self.model_name}.csv", index=False)

            # store the adiabatic populations --------------------
            self.state_pop_cc_AB = np.zeros([len(self.t_cc), self.A, self.A], dtype=complex)

            P_dic = {P[0]: P[1] for P in self.State_pop_AB}
            for idx, t in enumerate(sol.t):
                self.state_pop_cc_AB[idx, :] += P_dic[t]
            # ---------------------------------------------------------------------------------

        if debug:
            # only extract the values which correspond to time steps in the solution
            # since we save them for all integration steps, but only some are accepted
            if self.highest_order >= 0:
                self.dS_debug = np.zeros((len(sol.t), self.A, self.A), dtype=complex)
                dS_dic = {x[0]: x[1] for x in self.dS_store}
                for idx, t in enumerate(sol.t):
                    self.dS_debug[idx] = dS_dic[t]
            # ---------------------------------------------------------------------------------
            if self.highest_order >= 1:
                self.dt_i_debug = np.zeros((len(sol.t), self.A, self.A, self.N), dtype=complex)
                dt_i_dic = {x[0]: x[1] for x in self.dt_i_store}
                for idx, t in enumerate(sol.t):
                    self.dt_i_debug[idx] = dt_i_dic[t]
            # ---------------------------------------------------------------------------------
            if self.highest_order >= 2:
                self.dt_ij_debug = np.zeros((len(sol.t), self.A, self.A, self.N, self.N), dtype=complex)
                dt_ij_dic = {x[0]: x[1] for x in self.dt_ij_store}
                for idx, t in enumerate(sol.t):
                    self.dt_ij_debug[idx] = dt_ij_dic[t]
            # ---------------------------------------------------------------------------------

        log.debug(f"Status message: {sol.message}")  # description of termination reason
        log.debug(f"status: {sol.status}")  # -1: step failed, 0: reached end of tspan, 1: termination event occurred
        log.debug(f"Succeeded?: {sol.success}")  # bool if reached end of interval or termination event occurred
        
        return

    def _unravel_y_tensor(self, y_tensor):
        """ Restore the original shape of the flattened y tensor """
        # added support for Z 4 and removed T_2 support,        
        A, N = self.A, self.N  # for brevity

        # all return tensors start as None
        Z = {0: None, 1: None, 2: None, 3: None, 4:None}
        T = {0: None, 1: None}

        # ------------------------------ restore z tensor ----------------------------

        # constant terms
        start_constant_slice_index = 0
        end_constant_slice_index = start_constant_slice_index + A * A
        Z[0] = np.reshape(
            y_tensor[start_constant_slice_index:end_constant_slice_index],
            newshape=(A, A)
        )

        if self.Z_truncation_order >= 1:
            # linear terms
            start_linear_slice_index = end_constant_slice_index
            end_linear_slice_index = start_linear_slice_index + A * A * N
            Z[1] = np.reshape(
                y_tensor[start_linear_slice_index: end_linear_slice_index],
                newshape=(A, A, N)
            )

        if self.Z_truncation_order >= 2:
            # quadratic terms
            start_quadratic_slice_index = end_linear_slice_index
            end_quadratic_slice_index = start_quadratic_slice_index + A * A * N * N
            Z[2] = np.reshape(
                y_tensor[start_quadratic_slice_index: end_quadratic_slice_index],
                newshape=(A, A, N, N)
            )

        if self.Z_truncation_order >= 3:
            # cubic terms
            start_cubic_slice_index = end_quadratic_slice_index
            end_cubic_slice_index = start_cubic_slice_index + A * A * N * N * N
            Z[3] = np.reshape(
                y_tensor[start_cubic_slice_index: end_cubic_slice_index],
                newshape=(A, A, N, N, N)
            )
        
        if self.Z_truncation_order >= 4:
            # quartic terms
            start_quartic_slice_index = end_cubic_slice_index
            end_quartic_slice_index = start_quartic_slice_index + A * A * N * N * N * N
            Z[4] = np.reshape(
                y_tensor[start_quartic_slice_index: end_quartic_slice_index],
                newshape=(A, A, N, N, N, N)
            )

        # ------------------------------ restore t tensor ----------------------------

        # need a bit of logic to figure out the final Z slice index

        start_constant_slice_index = end_constant_slice_index

        if self.Z_truncation_order >= 1:
            start_constant_slice_index = end_linear_slice_index

        if self.Z_truncation_order >= 2:
            start_constant_slice_index = end_quadratic_slice_index

        if self.Z_truncation_order >= 3:
            start_constant_slice_index = end_cubic_slice_index
        
        if self.Z_truncation_order >= 4:
            start_constant_slice_index = end_quartic_slice_index

        # constant terms
        end_constant_slice_index = start_constant_slice_index + A
        T[0] = np.reshape(
            y_tensor[start_constant_slice_index: end_constant_slice_index],
            newshape=(A, )
        )

        if self.T_truncation_order >= 1:
            # linear terms
            start_linear_slice_index = end_constant_slice_index
            end_linear_slice_index = start_linear_slice_index + A * N
            T[1] = np.reshape(
                y_tensor[start_linear_slice_index: end_linear_slice_index],
                newshape=(A, N)
            )


        return Z, T

    def _ravel_y_tensor(self, Z, T):
        """ Flatten the `t` and `z` tensors into a 1D array """
        # modified to include  Z 4 and no T_2 support
        z_tensor_list = [Z[0].ravel(), ]
        if self.Z_truncation_order >= 1:
            z_tensor_list.append(Z[1].ravel())
        if self.Z_truncation_order >= 2:
            z_tensor_list.append(Z[2].ravel())
        if self.Z_truncation_order >= 3:
            z_tensor_list.append(Z[3].ravel())
        if self.Z_truncation_order >= 4:
            z_tensor_list.append(Z[4].ravel())
        
        t_tensor_list = [T[0].ravel(), ]
        if self.T_truncation_order >= 1:
            t_tensor_list.append(T[1].ravel())

        # the t tensor should come before the z tensor
        y_tensor = np.concatenate((*z_tensor_list, *t_tensor_list))

        return y_tensor

    def _print_integration_progress(self, time, t_final, *args):
        """ Prints to stdout every 1e4 steps or if current fs value is a multiple of (0.1 * `t_final`). """

        # unpack any args we wish to print
        Z, T = args

        self.counter += 1
        self.last_counter += 1

        fs = self.remove_time_step_conversion(time)

        at_least_1_femtosecond_has_passed_since_last_print = bool((fs-self.last_print) > 1.0)
        # at_least_5_femtosecond_has_passed_since_last_print = bool((fs-self.last_print) > 5.0)

        # compute the fs mod 10% of the final runtime and check to see if its close to a small non-zero value.
        small_delta, fs_mod_10p = 0.1, round(fs, 1) % (t_final * 0.1)
        time_is_a_multiple_of_ten_percent_of_t_final = np.isclose(small_delta, fs_mod_10p)

        # compute the fs mod 5% of the final runtime and check to see if its close to a small non-zero value.
        # small_delta, fs_mod_5p = 0.1, round(fs, 1) % (t_final * 0.05)
        # time_is_a_multiple_of_five_percent_of_t_final = np.isclose(small_delta, fs_mod_5p)

        print_flag = bool(
            self.last_counter >= int(1e4)
            or
            (
                at_least_1_femtosecond_has_passed_since_last_print
                and time_is_a_multiple_of_ten_percent_of_t_final
            )
        )

        if print_flag:
            log.info(
                f"On integration step {self.counter:<8d} at {fs:>9.4f}fs\n"
                f"C(t)(ABS) = {self.C_tau_ABS[-1][1]:>9.4f}\n"
                # f"C(t)(ECD) = {self.C_tau_ECD[-1][1]:>9.4f}\n"
            )

            # -------------------------------------------------------------------------------------
            t_amplitude_values_string = f'max constant T[0]  amplitude: {abs(T[0]).max()}\n'

            if self.T_truncation_order >= 1:
                t_amplitude_values_string += f'max single   T[1]  amplitude: {abs(T[1]).max()}\n'
            if self.T_truncation_order >= 2:
                t_amplitude_values_string += f'max double   T[2]  amplitude: {abs(T[2]).max()}\n'
            if self.T_truncation_order >= 3:
                t_amplitude_values_string += f'max triple   T[3]  amplitude: {abs(T[3]).max()}\n'

            log.info(t_amplitude_values_string)
            # -------------------------------------------------------------------------------------
            z_amplitude_values_string = f'max constant z_0   amplitude: {abs(Z[0]).max()}\n'

            if self.Z_truncation_order >= 1:
                z_amplitude_values_string += f'max single   z_i   amplitude: {abs(Z[1]).max()}\n'
            if self.Z_truncation_order >= 2:
                z_amplitude_values_string += f'max double   z_ij  amplitude: {abs(Z[2]).max()}\n'
            if self.Z_truncation_order >= 3:
                z_amplitude_values_string += f'max triple   z_ijk amplitude: {abs(Z[3]).max()}\n'
            if self.Z_truncation_order >= 4:
                z_amplitude_values_string += f'max qaurtuple   z_ijkl amplitude: {abs(Z[4]).max()}\n'
            log.info(z_amplitude_values_string)

            # log.info(f'Norm of the wavefunction:\n{self.Norm[-1][1]}')

            if self.calculate_population_flag:
                log.info(f'Diabatic  State population:\n {self.State_pop_DB[-1][1]}')
                log.info(f'Adiabatic State population:\n {self.State_pop_AB[-1][1]}')

            self.last_print = fs
            self.last_counter = 0

        return
        
    def _similarity_trans_Oz(self, H_args, T_args):
        """ Apply a similarity transformation to the Hamiltonian.
        This is applied for each electronic surface.
        """
        
        A, N = self.A, self.N   # for readability

        # rewrite residue for T to avoid permutations of electronic labels
        def f_m0_n0(H_args,T_args):
            # initialize as zero
            R = np.zeros([A, A], dtype=complex)

            # constant
            R += H_args[(0, 0)]
            # upper 
            # lower 
            
            # linear
            if self.T_truncation_order >= 1:
                R += np.einsum('abi,i->ab', H_args[(0, 1)], T_args[1]) # 1,1 worked
                
                if self.trunc.at_least_quadratic:
                    R += 0.5 * np.einsum('abij,i,j->ab', H_args[(0, 2)], T_args[1], T_args[1])
                ###################################### H3 + H4 new 
                if self.trunc.at_least_cubic:
                    R += (1/6) * np.einsum('abijk,i,j,k-> ab', H_args[(0, 3)], T_args[1], T_args[1], T_args[1]) 
                if self.trunc.at_least_quartic:
                    R += (1/24) * np.einsum('abijkl,i,j,k,l-> ab', H_args[(0, 4)], T_args[1], T_args[1], T_args[1], T_args[1])
            
            return R
        
        def f_m0_n1(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N], dtype=complex)

            # constant
            # (upper,lower)
            #capital:lower index
            #lower case: upper index
            R += H_args[(0, 1)] # 1,1 worked
            

            if self.T_truncation_order >= 1:
                R += np.einsum('abij,j->abi', H_args[(0, 2)], T_args[1])

                ############################################ H3 + H4 new 
                if self.trunc.at_least_cubic:
                    R += (1/2)*np.einsum('abijk,j,k->abi', H_args[(0, 3)], T_args[1],T_args[1])                
                if self.trunc.at_least_quartic:
                    R += (1/6)*np.einsum('abijkl,j,k,l->abi', H_args[(0, 4)], T_args[1],T_args[1],T_args[1])

            return R
        
        def f_m1_n0(H_args,T_args):
            """return residue R_I"""

            # initialize
            R = np.zeros([A, A, N], dtype=complex)

            # non zero initial value of R
            R += H_args[(1, 0)]

            # linear
            if self.T_truncation_order >= 1: # and self.gen_trunc.h_at_least_quadratic:
                R += np.einsum('abij,j->abi', H_args[(1, 1)], T_args[1]) # 1,1 worked
                
                ######################################################### H3 + H4 new 
                if self.trunc.at_least_cubic:
                    R += 0.5 * np.einsum('abijk,j,k->abi', H_args[(1, 2)], T_args[1], T_args[1])
                if self.trunc.at_least_quartic:
                    R += (1/6) * np.einsum('abijkl,j,k,l -> abi', H_args[(1, 3)], T_args[1], T_args[1], T_args[1])
            
            return R

        def f_m1_n1(H_args,T_args):
            """return residue R_I"""

            # initialize
            R = np.zeros([A, A, N, N], dtype=complex) 

            # constant
            R += H_args[(1, 1)]
            
            ############################################################# H3 + H4 new
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_cubic:
                    R += np.einsum('abijk,k->abij', H_args[(1, 2)], T_args[1])
                if self.trunc.at_least_quartic:
                    R += 0.5*np.einsum('abijkl,k,l->abij', H_args[(1, 3)], T_args[1],T_args[1])
            
            return R        

        def f_m0_n2(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N, N], dtype=complex)
            
            # constant 
            R += H_args[(0,2)]

            # linear
            ############################################################# H3 + H4 new            
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_cubic:
                    R += np.einsum('abijk,k->abij', H_args[(0, 3)], T_args[1])
                if self.trunc.at_least_quartic:
                    R += 0.5*np.einsum('abijkl,k,l->abij', H_args[(0, 4)], T_args[1],T_args[1])
            
            return R   

        def f_m2_n0(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N, N], dtype=complex)
            
            R += H_args[(2,0)]

            ############################################################# H3 + H4 new
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_cubic:
                    R += np.einsum('abijk,k->abij', H_args[(2, 1)], T_args[1])

                if self.trunc.at_least_quartic:
                    R += 0.5*np.einsum('abijkl,k,l->abij', H_args[(2, 2)], T_args[1],T_args[1])
            
            return R
        
        ############################################################# H3 + H4 new
        def f_m0_n3(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            R += H_args[(0,3)]

            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic:
                    R += np.einsum('abijkl,l->abijk', H_args[(0, 4)], T_args[1])
  
            return R
        
        def f_m3_n0(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            R += H_args[(3,0)]

            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic:
                    R += np.einsum('abijkl,l->abijk', H_args[(3, 1)], T_args[1])
 
            return R
        
        def f_m1_n2(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            R += H_args[(1,2)]

            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic:
                    R += np.einsum('abijkl,l->abijk', H_args[(1, 3)], T_args[1])
 
            return R
        
        def f_m2_n1(H_args,T_args):
            """return residue R_I"""

            # initialize as zero
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            R += H_args[(2,1)]

            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic:
                    R += np.einsum('abijkl,l->abijk', H_args[(2, 2)], T_args[1])
 
            return R
        ########################################################################### H3 + H4 new
        
        # compute similarity transformed Hamiltonian over e^T
        h_bar = {}        
        
        h_bar[(0, 0)] = f_m0_n0(H_args, T_args)
        h_bar[(0, 1)] = f_m0_n1(H_args, T_args)
        h_bar[(1, 0)] = f_m1_n0(H_args, T_args)
        
        h_bar[(1, 1)] = f_m1_n1(H_args, T_args)
        h_bar[(0, 2)] = f_m0_n2(H_args, T_args)
        h_bar[(2, 0)] = f_m2_n0(H_args, T_args)
        
        # rank 3
        h_bar[(3, 0)] = f_m3_n0(H_args,T_args)
        h_bar[(0, 3)] = f_m0_n3(H_args,T_args)
        h_bar[(1, 2)] = f_m1_n2(H_args,T_args)
        h_bar[(2, 1)] = f_m2_n1(H_args,T_args)

        # rank 4 They get no functions based on document   
        h_bar[(3, 1)] = H_args[(3, 1)]
        h_bar[(1, 3)] = H_args[(1, 3)]
        h_bar[(0, 4)] = H_args[(0, 4)]
        h_bar[(4, 0)] = H_args[(4, 0)]
        h_bar[(2, 2)] = H_args[(2, 2)]        
        return h_bar

    def _cal_H_bar_tilde_Oz(self, H_args:dict, T_args:dict, opt_flag=False)-> dict:
        """ Summary
        Oz calculate the second similarity transform the the Hamiltonian

        Parameters
        ----------
        H_bar : dict
            _description_
        T_dict : dict
            _description_
        opt_flag : bool, optional
            _description_, by default False

        Returns
        -------
        dict
            _description_
        """        

        # maybe add hamiltonian term conditon

        A, N = self.A, self.N

        T_args = T_conj = {1: np.conj(T_args[1])}
        
        def f_m0_n0(H_args,T_args):
            
            R = np.zeros([A, A], dtype=complex)
            
            R += H_args[(0, 0)]

            if self.T_truncation_order >= 1:
                R += np.einsum('i,abi->ab',T_args[1],H_args[(1, 0)]) # 1,1 worked

                # not used debugging               
                if self.trunc.at_least_quadratic:
                    R += 0.5 * np.einsum('i,j,abij->ab', T_args[1], T_args[1],H_args[(2, 0)])
                    
                ############################################################# H3 + H4 new
                if self.trunc.at_least_cubic:
                    R += (1/6) * np.einsum('i,j,k,abijk-> ab',T_args[1], T_args[1],T_args[1],H_args[(3, 0)]) 
                if self.trunc.at_least_quartic:
                    R += (1/24) * np.einsum('i,j,k,l,abijkl-> ab', T_args[1], T_args[1],T_args[1], T_args[1],H_args[(4, 0)])
            return R
        
        def f_m0_n1(H_args,T_args):
            
            R = np.zeros([A, A, N], dtype=complex)
            
            R += H_args[(0, 1)]
 
            if self.T_truncation_order >= 1: 
                R += np.einsum('i,abil->abl',T_args[1],H_args[(1, 1)]) # 1,1 worked
                # breakpoint()
                
                ############################################################# H3 + H4 new
                if self.trunc.at_least_cubic:
                    R += 0.5 * np.einsum('i,j,abijl->abl', T_args[1], T_args[1],H_args[(2, 1)])
                if self.trunc.at_least_quartic:
                    R += (1/6) * np.einsum('i,j,k,abijkl-> abl',T_args[1], T_args[1],T_args[1],H_args[(3, 1)])
            
            return R

        def f_m1_n0(H_args,T_args):
            # constant
            R = np.zeros([A, A, N], dtype=complex)
            
            # constant
            R += H_args[(1, 0)] # 1,1 worked
            
            if self.T_truncation_order >= 1:            
                R += np.einsum('i,abil->abl',T_args[1],H_args[(2,0)])
                
                ############################################################# H3 + H4 new
                if self.trunc.at_least_cubic:
                    R += 0.5 * np.einsum('i,j,abijl->abl', T_args[1], T_args[1],H_args[(3, 0)])
                if self.trunc.at_least_quartic:
                    R += (1/6) * np.einsum('i,j,k,abijkl-> abl',T_args[1], T_args[1],T_args[1],H_args[(4, 0)])
            
            return R
        
        def f_m0_n2(H_args,T_args):
            R = np.zeros([A, A, N, N], dtype=complex)
            
            # constant
            R += H_args[(0,2)]
            
            ############################################################# H3 + H4 new
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_cubic:
                    R += np.einsum('i,abikl->abkl',T_args[1],H_args[(1, 2)])
                
                if self.trunc.at_least_quartic:
                    R += 0.5 * np.einsum('i,j,abijkl->abkl', T_args[1], T_args[1],H_args[(2, 2)])
            
            return R
        
        def f_m2_n0(H_args,T_args):
            R = np.zeros([A, A, N, N], dtype=complex)
            
            # constant
            R += H_args[(2,0)]
            
            ############################################################# H3 + H4 new
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_cubic:
                    R += np.einsum('i,abikl->abkl',T_args[1],H_args[(3, 0)])
                if self.trunc.at_least_quartic:
                    R += 0.5 * np.einsum('i,j,abijkl->abkl', T_args[1], T_args[1],H_args[(4, 0)])

            return R

        def f_m1_n1(H_args,T_args):
            R = np.zeros([A, A, N, N], dtype=complex)
            
            # constant
            R += H_args[(1,1)]
            
            ############################################################# H3 + H4 new
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_cubic :
                    R += np.einsum('i,abikl->abkl',T_args[1],H_args[(2, 1)])
                
                if self.trunc.at_least_quartic :
                    R += 0.5 * np.einsum('i,j,abijkl->abkl', T_args[1], T_args[1],H_args[(3, 1)])
            
            return R 

        ############################################################# H3 + H4 new
        def f_m3_n0(H_args,T_args):
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            # constant
            R += H_args[(3,0)]
                  
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic :
                    R += np.einsum('i,abijkl->abjkl',T_args[1],H_args[(4, 0)])
            
            return R
        
        def f_m0_n3(H_args,T_args):
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            # constant
            R += H_args[(0,3)]
                  
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic :
                    R += np.einsum('i,abijkl->abjkl',T_args[1],H_args[(4, 0)])
            
            return R
        
        def f_m1_n2(H_args,T_args):
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            # constant
            R += H_args[(1,2)]
                  
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic :
                    R += np.einsum('i,abijkl->abjkl',T_args[1],H_args[(2, 2)])
            
            return R
        
        def f_m2_n1(H_args,T_args):
            R = np.zeros([A, A, N, N, N], dtype=complex)
            
            # constant
            R += H_args[(2,1)]
                  
            if self.T_truncation_order >= 1:
                if self.trunc.at_least_quartic :
                    R += np.einsum('i,abijkl->abjkl',T_args[1],H_args[(3, 1)])
            
            return R
        ############################################################# H3 + H4 new
        
        h_tilde = {}        
        
        # Linear
        h_tilde[(0, 0)] = f_m0_n0(H_args, T_args)
        h_tilde[(0, 1)] = f_m0_n1(H_args, T_args)
        h_tilde[(1, 0)] = f_m1_n0(H_args, T_args)
        
        # Qaudratic 
        h_tilde[(1, 1)] = f_m1_n1(H_args, T_args)
        h_tilde[(0, 2)] = f_m0_n2(H_args, T_args)
        h_tilde[(2, 0)] = f_m2_n0(H_args, T_args)
        
        # rank 3
        h_tilde[(3, 0)] = f_m3_n0(H_args, T_args)
        h_tilde[(0, 3)] = f_m0_n3(H_args, T_args)
        h_tilde[(1, 2)] = f_m1_n2(H_args, T_args)
        h_tilde[(2, 1)] = f_m2_n1(H_args, T_args)
        
        # rank 4  
        h_tilde[(3, 1)] = H_args[(3, 1)]
        h_tilde[(1, 3)] = H_args[(1, 3)]
        h_tilde[(0, 4)] = H_args[(0, 4)]
        h_tilde[(4, 0)] = H_args[(4, 0)]
        h_tilde[(2, 2)] = H_args[(2, 2)] 
        
        return h_tilde

    def _compute_Z_matrix_Oz(self,Z_bar:dict,T_args:dict) -> dict:
        """Returns Similairty transform of Z with e^{T_dict^dagger}

        Parameters
        ----------
        Z : dict
            _description_
        T_dict : dict
            _description_

        Returns
        -------
        dict
            _description_
        """

        T_args = T_conj = {1: np.conj(T_args[1])}
        
        A, N = self.A, self.N
        Z_dict = {(0,0):np.zeros(shape=(A, ), dtype=complex),
            (1,0):np.zeros(shape=(A, N,),dtype=complex), 
            (2,0):np.zeros(shape=(A, N, N),dtype=complex), 
            (3,0):np.zeros(shape=(A, N, N, N),dtype=complex), 
            (4,0):np.zeros(shape=(A, N, N, N, N),dtype=complex)
        }
        
        if self.Z_truncation_order >= 1:
            Z_dict[(0,0)] += Z_bar[(0,0)]
            
            Z_dict[(0,0)] -= np.einsum('ai,i -> a',Z_bar[(1,0)],T_args[1])                 
            
            Z_dict[(1,0)] += Z_bar[(1,0)]
        
        if self.Z_truncation_order >= 2:
            Z_dict[(0,0)] += 0.5*np.einsum('aij,i,j -> a',Z_bar[(2,0)],T_args[1],T_args[1]) 
            
            Z_dict[(1,0)] -= np.einsum('aij,j -> ai',Z_bar[(2,0)],T_args[1])
            
            Z_dict[(2,0)] += Z_bar[(2,0)]
        
        if self.Z_truncation_order >= 3:
            Z_dict[(0,0)] -= (1/6)*np.einsum('aijk,i,j,k -> a',Z_bar[(3,0)],T_args[1],T_args[1],T_args[1]) 
            
            Z_dict[(1,0)] += 0.5*np.einsum('aijk,j,k -> ai',Z_bar[(3,0)],T_args[1],T_args[1])
            
            Z_dict[(2,0)] -= np.einsum('aijk, k -> aij',Z_bar[(3,0)],T_args[1])
            
            Z_dict[(3,0)] += Z_bar[(3,0)] 

        
        if self.Z_truncation_order >= 4:
            Z_dict[(0,0)] += (1/24)*np.einsum('aijkl,i,j,k,l -> a',Z_bar[(4,0)],T_args[1],T_args[1],T_args[1],T_args[1]) 
            
            Z_dict[(1,0)] -= (1/6)*np.einsum('aijkl,j,k,l -> ai',Z_bar[(4,0)],T_args[1],T_args[1],T_args[1])
            
            Z_dict[(2,0)] += 0.5*np.einsum('aijkl,k,l -> aij',Z_bar[(4,0)],T_args[1],T_args[1])
            
            Z_dict[(3,0)] -= np.einsum('aijkl,l -> aijk',Z_bar[(4,0)],T_args[1])
            
            Z_dict[(4,0)] += Z_bar[(4,0)]
    
        return Z_dict
    
    def _compute_acl_initial(self,z_args,t_args):
        # I think this function should calculate e^D and Z_init
        t_init = {1 : np.zeros((self.N),
                               complex)}
        t_init[1] += self.init_t
        z_0_inital = np.zeros(self.A,complex)
        
        if self.Z_truncation_order >= 0:
            z_0_inital += z_args[(0,0)] 

        if  self.Z_truncation_order >= 1:
            z_0_inital += np.einsum('ai,i -> a',z_args[1,0],t_init[1])
            
        if  self.Z_truncation_order >= 2:
            z_0_inital += 0.5*np.einsum('aij,i,j -> a', z_args[(2,0)], t_init[1],t_init[1])

        if  self.Z_truncation_order >= 3:
            z_0_inital += (1/6)*np.einsum('aijk,i,j,k -> a', z_args[(3,0)], t_init[1],t_init[1],t_init[1])
        
        if  self.Z_truncation_order >= 4:
            z_0_inital += (1/24)*np.einsum('aijkl,i,j,k,l -> a', z_args[(4,0)], t_init[1],t_init[1],t_init[1],t_init[1])
        
        # breakpoint()
        D_prime = np.dot(t_init[1],t_args[1])
        acl_init = z_0_inital*np.exp(D_prime)*np.exp(t_args[0])*np.exp(-np.dot(t_init[1],t_init[1])) 
        # print(z_0_inital)
        return(acl_init)
 
    
    # decoupling scheme
    def derivatives_T_decoupled_scheme(self,H_tilde,Z_bar,T_dict,derivatives):
        
        if self.z_check == True:
            t_coeff = 0 
        else:
            t_coeff = 1
            
        A,N = self.A,self.N
        if self.Z_truncation_order >= 1:
            # W is 0 its straightforward
            if N == 1:
                derivatives['dT_1'] = -1j*H_tilde[(1,0)].squeeze().reshape(1)*t_coeff
            else:
                derivatives['dT_1'] = -1j*H_tilde[(1,0)].squeeze()*t_coeff  # check that its not activated       
            T_1_conj_dict = { 1 : np.conj(T_dict[1])}
            # breakpoint()
            dT_0 = -1j*H_tilde[(0,0)]- np.einsum('i,i -> ',T_1_conj_dict[1],derivatives['dT_1'])
        
            derivatives['dT_0'] = dT_0
        
        return(derivatives)
    
    def derivatives_Z_decoupled_scheme(self,H_tilde,Z_bar,T_dict,derivatives):
        
        # update our hamiltonian
        T_1_conj_dict = { 1 : np.conj(T_dict[1])}   
        A, N = self.A, self.N
        
        # residual dictionary
        residual = {0:np.zeros(shape=(A, ), dtype=complex),
                    1:np.zeros(shape=(A, N,),dtype=complex), 
                    2:np.zeros(shape=(A, N, N),dtype=complex), 
                    3:np.zeros(shape=(A, N, N, N),dtype=complex), 
                    4:np.zeros(shape=(A, N, N, N, N),dtype=complex),}
        
        if self.Z_truncation_order >= 1:
            # update our hamiltonian   
            G = {}

            # hamiltonian needs to have check statemnts for order
            G[(0,0)] = -1j*H_tilde[(0,0)] - derivatives['dT_0'] - np.einsum('i,i -> ',T_1_conj_dict[1],derivatives['dT_1'])
            G[(0,1)] = -1j*H_tilde[(0,1)] + np.conj(derivatives['dT_1'])
            G[(1,0)] = -1j*H_tilde[(1,0)] - derivatives['dT_1']

            G[(0,2)] = -1j*H_tilde[(0,2)]
            G[(2,0)] = -1j*H_tilde[(2,0)]
            G[(1,1)] = -1j*H_tilde[(1,1)]
            
            G[(0,3)] = -1j*H_tilde[(0,3)]
            G[(3,0)] = -1j*H_tilde[(3,0)]
            G[(1,2)] = -1j*H_tilde[(1,2)]
            G[(2,1)] = -1j*H_tilde[(2,1)]
            G[(0,4)] = -1j*H_tilde[(0,4)]
            G[(4,0)] = -1j*H_tilde[(4,0)]
            G[(1,3)] = -1j*H_tilde[(1,3)]
            G[(3,1)] = -1j*H_tilde[(3,1)]
            G[(2,2)] = -1j*H_tilde[(2,2)]
            # breakpoint()

        if self.Z_truncation_order >= 1:
            z_four_eqns.add_m0_n0_HZ_terms(
                residual[0], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_1_conj_dict, G, Z_bar
            )
            
            derivatives['dZ_0'] = residual[0]
            
            z_four_eqns.add_m0_n1_HZ_terms(
                residual[1], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_1_conj_dict, G, Z_bar
            )
           
            derivatives['dZ_1'] = residual[1]
            
        if self.Z_truncation_order >= 2:
            z_four_eqns.add_m0_n2_HZ_terms(
                residual[2], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_1_conj_dict, G, Z_bar
            )
            derivatives['dZ_2'] = symmetrize_tensor(N,residual[2],2)
            
        if self.Z_truncation_order >= 3:
            z_four_eqns.add_m0_n3_HZ_terms(
                residual[3], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_1_conj_dict, G, Z_bar
            )
            derivatives['dZ_3'] = symmetrize_tensor(N,residual[3],3)
        
        if self.Z_truncation_order >= 4:
            z_four_eqns.add_m0_n4_HZ_terms(
                residual[4], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_1_conj_dict, G, Z_bar
            )
            derivatives['dZ_4'] = symmetrize_tensor(N,residual[4],4)
        
        # breakpoint()
        # make residuals a property so I can call it outside of object  
        # self.residual_dict = residual.copy()
        
        return(derivatives)
    
    def derivative_functions_decoupled_scheme(self,H_tilde,Z_bar,T_dict):
        A = self.A
        N = self.N        
        
        derivatives = {}
        
        for i in range(self.T_truncation_order+1):
            derivatives[f'dT_{i}'] = None
            
        for i in range(self.Z_truncation_order+1):
            derivatives[f'dZ_{i}'] = None
    
        derivatives = self.derivatives_T_decoupled_scheme(H_tilde,Z_bar,T_dict,derivatives)
        derivatives = self.derivatives_Z_decoupled_scheme(H_tilde,Z_bar,T_dict,derivatives)

        return derivatives
           
    def rk45_solve_ivp_integration_function_Oz(self,time,y_tensor,t_final):
        
        # INITIALIZE ARRAYS 
        A, N = self.A, self.N
        
        # unpack y tensor 
        # T-tensor is what is passed 
        # unravel y tensor is meant for z and t bound to each other 
        # for the inital step  Z dict is passed in here, however its initalized for values of Z-bar
        # all integration steps are with respect to updating Z bar
        # we only ever need to z_0 for the ACF   
        z_bar_unraveled, t_unraveled = self._unravel_y_tensor(y_tensor)
        
        # printing progression
        self._print_integration_progress(time, t_final, z_bar_unraveled, t_unraveled)
        
        # initialize new arrays
        self.norm = np.zeros(A, dtype=complex)
    
        # fill these arrays with zeros
        for key in self.dT.keys():
            self.dT[key].fill(complex(0.0))

        # for all intents and purposes we will think of this d_zbar in accordance to my formulae  
        for key in self.dZ.keys():
            self.dZ[key].fill(complex(0.0))
        
        
        # this goes every surface I only ever have 1 surface 
        for b in range(A):

            # pack T amplitude in to their dictionary 
            t_dict = {0: t_unraveled[0][b]}

            if self.T_truncation_order >= 1:
                t_dict[1] = t_unraveled[1][b, :]
            
            # if self.z_check == True:
            #     t_dict[1].fill(complex(0.0))
                # breakpoint()
                
            # pack Z amplitude
            z_bar_dict = {(0,0): z_bar_unraveled[0][b]}
            
            if self.Z_truncation_order >= 1:
                z_bar_dict[(1,0)] = z_bar_unraveled[1][b, :]
            
            if self.Z_truncation_order >= 2:
                z_bar_dict[(2,0)] = z_bar_unraveled[2][b, :]

            if self.Z_truncation_order >= 3:
                z_bar_dict[(3,0)] = z_bar_unraveled[3][b, :]
                
            if self.Z_truncation_order >= 4:
                z_bar_dict[(4,0)] = z_bar_unraveled[4][b, :]
            
            if energy_conservation_test:= False:
                t_dict[1] += self.init_t
            
            H_bar =  self._similarity_trans_Oz(self.h, t_dict) 
            H_tilde = self._cal_H_bar_tilde_Oz(H_bar,t_dict) 
            
            if energy_conservation_test == True:
                t_dict[1].fill(complex(0.0))
            
            derivatives = self.derivative_functions_decoupled_scheme(H_tilde,z_bar_dict,t_dict)
            
            self.dZ[0][b, :] += derivatives['dZ_0'] # The unravel of the dZ_0 only occurs in this scheme 
            self.dZ[1][b, :] += derivatives['dZ_1'] # The unravel of the dZ_1 only occurs in this scheme
        
            # update object derivatives
            self.dT[0][b] += derivatives['dT_0']
            self.dT[1][b,:] += derivatives['dT_1']
            
            if self.Z_truncation_order >= 2:
                self.dZ[2][b,:] += derivatives['dZ_2'][b,:]
            if self.Z_truncation_order >= 3:           
                self.dZ[3][b,:] += derivatives['dZ_3'][b,:]
            if self.Z_truncation_order >= 4:
                self.dZ[4][b,:] += derivatives['dZ_4'][b,:] 
            
            if energy_conservation_test == True:
                t_dict[1] += self.init_t
            # # calculate z_0 for the given step
            z_dict = self._compute_Z_matrix_Oz(z_bar_dict,t_dict)
            
            # compute ACF for ABS and ECD
            
            U = self._compute_acl_initial(z_dict,t_dict)
            (time,'U:',U,'\n','T_0:',t_dict[0])
            # if np.abs(np.exp(t_dict[0])) > 10:
            
 
        # flatten the z, t tensors into a 1D array
        delta_y_tensor = self._ravel_y_tensor(self.dZ, self.dT)
        
        # # store ACF data
        self.C_tau_ABS.append((time, U[0]))
        # Store T1 data
        self.T1_storage.append((time,t_dict[1].flatten()))
        
    
        if self.correlation_funcs:
            # storing T_1 
            if N == 1:
                self.T1_tau.append((time,t_dict[1][0]))
                
                # new methods 
                q,q_2 = self.eval_qs_1d(t_dict,z_bar_dict)
                
                self.exp_q.append((time,q))
                self.exp_q2.append((time,q_2))
                
            z_norm,z_exp_H = self.eval_norm_energy(t_dict,z_bar_dict,H_tilde)
            # print(z_exp_H)
            self.exp_norm.append((time,z_norm))
            self.exp_H.append((time,z_exp_H))
            
             
        
        return delta_y_tensor
    

    
    def rk45_integration_oz(self, t_init=0., t_final=10., density=1.0, nof_points=10000, debug_flag=False):
        """ Runge-Kutta integration scheme
        """

        # ------------------------------------------------------------------------
        # initialize integration parameters
        # ------------------------------------------------------------------------
        log.info(
            "We are going to preform a RK4(5) integration with: "
            f"({t_init=} {t_final=} {density=} {nof_points=} {debug_flag=})"
        )

        A, N = self.A, self.N  # to reduce line lengths, for conciseness

        # used for debugging purposes to print out the integration steps every n% of integration
        self.counter = 0
        self.last_counter = 0
        self.last_print = 0

        # this 'raw' step size is not scaled properly
        raw_step_size = (t_final - t_init) / nof_points

        # we need to convert the `raw_step_size`
        step_size = self.apply_time_step_conversion(raw_step_size) # step conversion needs to be used 
        log.info(
            f"{'Raw step size (unitless):':<26} ({raw_step_size})\n"
            f"{'Real step size (eV/fs):':<26} ({step_size})"
        )

        # initialize accumulation tensor
        U_0 = np.eye(A, dtype=complex)

        # initialize autocorrelation list
        # We need to store the t values for each C(t) so that we can match them up at the end
        
        self.C_tau_ABS = [(0.0, np.einsum('ac,cd,ad->', self.E_tdm, U_0, self.E_tdm, dtype=complex))]
        
        if self.correlation_funcs == True:
            # self.G0_tau_abs = [(0,self.h[(0,0)])]
            self.exp_q = [(0,0)]
            self.exp_q2 = [(0,0)]
            self.exp_H = [(0,0)]
            self.exp_norm = [(0,0)]
        
        # initialize z_bar amplitude we never expliclty need Z we only every use z_tilde
        initial_Z_tilde = {
            0: np.eye(A, dtype=complex),
            1: np.zeros([A, A, N], dtype=complex),
            2: np.zeros([A, A, N, N], dtype=complex),
            3: np.zeros([A, A, N, N, N], dtype=complex),
            4: np.zeros([A, A, N, N, N, N], dtype=complex),
        }

        # initialize t amplitude
        initial_T = {
            0: np.zeros(A, dtype=complex),
            1: np.zeros([A, N], dtype=complex),
        }
        
        # initialise the T_1 for displacement
        initial_T[1] += self.init_t 
        # breakpoint()
        if N == 1: 
            self.T1_tau = [(0.0, initial_T[1])]
             
        # prepare the initial y_tensor
        initial_y_tensor = self._ravel_y_tensor(initial_Z_tilde, initial_T)
        
        # initilize T_1 Z csvs
        self.T1_storage = []
        
        
        # ------------------------------------------------------------------------
        # the integration function called by the `solve_ivp` integrator at each step of integration
        # ------------------------------------------------------------------------
        
        # specify the precision of the integrator so that the output for the test models is numerically identical
        if self.comparing_to_test_models:
            relative_tolerance = 1e-10
            absolute_tolerance = 1e-12
        else:
        
            # So the new "fix" does work but we need to up the tolerance parameters of the RK integrator to get agreement with MCTDH.
            # I would do at least 1e-9 and 1e-12, the issue is that the integrator simply takes too large steps based on the ode's represented by S_0
            relative_tolerance = 1e-12
            absolute_tolerance = 1e-14
        # ------------------------------------------------------------------------
        # call the integrator
        # ------------------------------------------------------------------------
        integration_function = self.rk45_solve_ivp_integration_function_Oz
        
        sol = new_solve_ivp(
            fun=integration_function,  # the function we are integrating
            # method="RK45", 
            # method="RK23", 
            method='DOP853',
            first_step=step_size,  # fix the initial step size
            t_span=(
                t_init,  # initial time
                self.apply_time_step_conversion(t_final),  # boundary time, integration end point
            ),
            y0=initial_y_tensor,  # initial state - shape (n, )
            args=(t_final, ),  # extra args to pass to `rk45_solve_ivp_integration_function`
            max_step= 0.1, #self.step_size,  # maximum allowed step size
            rtol=relative_tolerance,  # relative tolerance
            atol=absolute_tolerance,  # absolute tolerance
            store_y_values=False,  # do not store the y values over the integration
            t_eval=None,  # store all the time values that we integrated over
            dense_output=False,  # extra debug information
            # we do not need to vectorize
            # this means to process multiple time steps inside the function `rk45_solve_ivp_integration_function`
            # it would be useful for a method which does some kind of block stepping
            vectorized=False,
        )
        # ------------------------------------------------------------------------
        # now we extract the relevant information from the integrator object `sol`
        # ------------------------------------------------------------------------
        
        # breakpoint()        
        self._postprocess_rk45_integration_results(sol, debug=debug_flag)
        
        return
    ##########################################################

    @staticmethod
    def _save_data(file_path, x_array, y_array):
        """ Save complex function y(x) into plain text file formatted as per MCTDH auto files

        This preforms file saving for both `save_acf_data` and `save_sos_data`
        `x_array` and `y_array` should be numpy arrays of the same length.
        `x_array` should have dtype float
        `y_array` should have dtype complex
        """

        # some simple data checking
        assert isinstance(x_array, np.ndarray), \
            f"x_array is not an numpy array it is a {type(x_array)}"
        assert isinstance(y_array, np.ndarray), \
            f"y_array is not an numpy array it is a {type(x_array)}"

        assert x_array.dtype == float, \
            f"x_array has type {x_array.dtype} instead of float"
        assert y_array.dtype == complex, \
            f"y_array has type {y_array.dtype} instead of complex"

        assert len(x_array) == len(y_array.real) == len(y_array.imag), \
            (
                f"Arrays must be the same length\n"
                f"x_array      has length: {len(x_array)}\n"
                f"y_array.real has length: {len(y_array.real)}\n"
                f"y_array.imag has length: {len(y_array.imag)}"
        )

        header = (
            f"#{'time[fs]': >12}"
            f" {'Re(autocorrel)': >23}"
            f" {'Im(autocorrel)': >19}"
            f" {'Abs(autocorrel)': >20}"
        )

        file_data = header + '\n'

        y_absolute = np.abs(y_array)

        # we assume all arrays are the same length
        for idx in range(len(x_array)):
            file_data += (
                f"{x_array[idx]: > 15.8f}"
                f" {y_array.real[idx]: > 23.14f}"
                f" {y_array.imag[idx]: > 19.14f}"
                f" {y_absolute[idx]: > 19.14f}"
                "\n"
            )
        # breakpoint()
        # write data to file
        with open(file_path, 'w') as file:
            file.write(file_data)
        return

    @staticmethod
    def _save_real_data(file_path, x_array, y_array,x_str,y_str):
        """ Save complex function y(x) into plain text file formatted as per MCTDH auto files

        This preforms file saving for both `save_acf_data` and `save_sos_data`
        `x_array` and `y_array` should be numpy arrays of the same length.
        `x_array` should have dtype float
        `y_array` should have dtype complex
        """

        # some simple data checking
        assert isinstance(x_array, np.ndarray), \
            f"x_array is not an numpy array it is a {type(x_array)}"
        assert isinstance(y_array, np.ndarray), \
            f"y_array is not an numpy array it is a {type(x_array)}"

        assert x_array.dtype == float or x_array.dtype == int, \
            f"x_array has type {x_array.dtype} instead of float or int"
        assert y_array.dtype == float, \
            f"y_array has type {y_array.dtype} instead of float"

        assert len(x_array) == len(y_array), \
            (
                f"Arrays must be the same length\n"
                f"x_array      has length: {len(x_array)}\n"
                f"y_array      has length: {len(y_array)}\n"

        )

        header = (
            f"# {f'{x_str}': >12}"
            f" {f'{y_str}': >20}"
        )

        file_data = header + '\n'

        # we assume all arrays are the same length
        if x_array.dtype == int:
            for idx in range(len(x_array)):
                file_data += (
                    f"{x_array[idx]: >12}"
                    f" {y_array[idx]: > 32.14f}"
                    "\n"
                )
        else:        
            for idx in range(len(x_array)):
                file_data += (
                    f"{x_array[idx]: > 15.18f}"
                    f" {y_array[idx]: > 23.18f}"
                    "\n"
                )

        # write data to file
        # breakpoint()
        with open(file_path, 'w') as file:
            file.write(file_data)
        return

    @classmethod
    def class_save_acf_data(cls, file_name, time, acf, output_path="./"):
        """ Store provided auto correlation function (ACF) into plain text file """
        file_path = join(output_path, f"ACF_CC_{file_name}.txt")
        cls._save_data(file_path, time, acf)
        return file_path

    @classmethod
    def class_save_sos_data(cls, file_name, time, sos, output_path="./"):
        """ Store provided Sum-Over-States data (SOS) into plain text file """
        file_path = join(output_path, f"ACF_SOS_{file_name}.txt")
        cls._save_data(file_path, time, sos)
        return file_path

    def save_acf_data(self, file_name, time=None, acf_ABS=None, acf_ECD=None, output_path="./"):
        """ Store auto correlation function (ACF) into plain text file

        The file is saved to `output_path` + "ACF_CC_{file_name}.txt"
        Caller can provide arguments `time` and/or `acf` as alternatives to `self.t_cc` and `self.C_tau_cc`
        if they want to preform some data processing before saving to a file.
        """
        if time is None:
            time = self.t_cc
        # breakpoint()
        if acf_ABS is None:
            acf_ABS = self.C_tau_cc_ABS
        # if acf_ECD is None:
        #     acf_ECD = self.C_tau_cc_ECD

        file_path_ABS = join(output_path, f"ACF_ABS_CC_{file_name}.txt")
        self._save_data(file_path_ABS, time, acf_ABS)

        # file_path_ECD = join(output_path, f"ACF_ECD_CC_{file_name}.txt")
        # self._save_data(file_path_ECD, time, acf_ECD)
        return file_path_ABS #file_path_ECD

    def save_sos_data(self, file_name, time=None, sos=None, output_path="./"):
        """ Store SOS (for debugging/testing/profiling) into plain text file

        Caller can provide arguments `time` and/or `sos` as alternatives to `self.t_sos` and `self.C_tau_sos`
        if they want to preform some data processing before saving to a file
        """
        if time is None:
            time = self.t_sos
        if sos is None:
            sos = self.C_tau_sos

        file_path = join(output_path, f"ACF_SOS_{file_name}.txt")
        self._save_data(file_path, time, sos)
        return file_path

    def save_raw_integration_data(self, file_name, output_path="./"):
        """ store this objects ACF and SOS values to text files without any preprocessing """

        # but now we use the same file_name for both ACf and sos?
        acf_path = self.save_acf_data(file_name, output_path=output_path)
        sos_path = self.save_sos_data(file_name, output_path=output_path)
        return acf_path, sos_path

    @staticmethod
    def _load_auto_data(file_path):
        """ Returns a tuple (x, y),
            x is array of time steps
            y is array of complex values at each x
        Only works on `auto` files generated by MCTDH.
        """

        # read data from file
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            raise e

        # starting from the end of the file go line by line to the front of the file
        for line in lines[-1:0:-1]:
            length = len(line)
            # all lines of data are at least 77 characters long
            if length < 10:
                del lines[-1]
            elif length >= 77:
                break
            else:
                raise Exception(str(
                    "Line length is > 10 but < 77?\n"
                    "This file has something strange going on at the end. Please check.\n"
                    f"The line is {length} characters long.\n"
                    f"This is the offending line:\n{[line,]}\n"
                ))

        # the first line is a comment
        num_data_points = len(lines) - 1
        y_val = np.zeros(num_data_points, dtype=complex)
        time = np.zeros(num_data_points, dtype=float)

        p = parse.compile("{t:>15.8f} {y_real:>22.14f} {y_imag:>18.14f} {y_abs:>18.14f}")

        # extract data from lines of file
        for idx, line in enumerate(lines[1:]):

            r = p.parse(line)
            time[idx] = r['t']
            y_val.real[idx] = r['y_real']
            y_val.imag[idx] = r['y_imag']

        return time, y_val

    @staticmethod
    def load_auto_data(file_path):
        """ Wrapper function for `_load_auto_data`
        Returns a tuple (x, y),  x is array of time steps, y is array of ACF value at each x.
        Only works on `auto` files generated by MCTDH.
        """
        time, acf = vibronic_hamiltonian._load_auto_data(file_path)
        return time, acf

    @staticmethod
    def _load_data(file_path):
        """ Returns a tuple (x, y),
            x is array of time steps
            y is array of complex values at each x
        """

        # read data from file
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            raise e

        # starting from the end of the file go line by line to the front of the file
        for line in lines[-1:0:-1]:
            length = len(line)
            # all lines of data are at least 80 characters long
            if length < 10:
                del lines[-1]
            elif length >= 80:
                break
            else:
                raise Exception(str(
                    "Line length is > 10 but < 80?\n"
                    "This file has something strange going on at the end. Please check.\n"
                    f"The line is {length} characters long.\n"
                    f"This is the offending line:\n{[line,]}\n"
                ))

        # the first line is a comment
        num_data_points = len(lines) - 1
        y_val = np.zeros(num_data_points, dtype=complex)
        time = np.zeros(num_data_points, dtype=float)

        p = parse.compile("{t:>15.8f} {y_real:>23.14f} {y_imag:>19.14f} {y_abs:>19.14f}")

        # check to make sure parse isn't broken, as per issue #88
        # breakpoint()
        r_test = p.parse(lines[1])
        assert r_test is not None, (
            "the python package parse was unable to process the first valid line in the dataset\n"
            "if your data set is properly formatted this likely indicates your using the wrong version"
            "of parse. version 1.18 should be stable, but not 1.19"
        )

        # extract data from lines of file
        for idx, line in enumerate(lines[1:]):

            r = p.parse(line)
            time[idx] = r['t']
            y_val.real[idx] = r['y_real']
            y_val.imag[idx] = r['y_imag']

        return time, y_val

    @staticmethod
    def load_acf_data(file_path):
        """ Wrapper function for `_load_data`
        Returns a tuple (x, y),  x is array of time steps, y is array of ACF value at each x
        """
        time, acf = vibronic_hamiltonian._load_data(file_path)
        return time, acf

    @staticmethod
    def load_sos_data(file_path):
        """ Wrapper function for `_load_data`
        Returns a tuple (x, y),  x is array of time steps, y is array of SOS value at each x
        """
        time, sos = vibronic_hamiltonian._load_data(file_path)
        return time, sos

    def load_inplace_acf_data(self, file_path):
        """ Wrapper function for `_load_data`
        Overwrite `self.C_tau_cc` and `self.t_cc` with the values of the file at `file_path`.
        """
        time, acf = self._load_data(file_path)
        self.C_tau_cc = acf.copy()
        self.t_cc = time.copy()
        return

    def load_inplace_sos_data(self, file_path):
        """ Wrapper function for `_load_data`
        Overwrite `self.C_tau_sos` and `self.t_sos` with the values of the file at `file_path`.
        """
        time, sos = self._load_data(file_path)
        self.C_tau_sos = sos.copy()
        self.t_sos = time.copy()
        return

    def plot_acf(
        self, file_name, sos_flag=False, output_path="./",
        plot_log=False, long_propagation_zoom=False,
        acf_y_array=None, acf_x_array=None, sos_y_array=None, sos_x_array=None,
    ):
        r""" Plot the Auto Correlation Function (ACF)

        `plot_log` is Boolean, if True an additional plot with log scale y-axis is generated.
        `long_propagation_zoom` if Boolean, if True changes x-axis to show only last n-1 femtoseconds of ACF in plot.

        The arguments `acf_y_array`, `acf_x_array`, `sos_y_array`, `sos_x_array` are provided so that the caller can provide alternative data for plotting.
        For large systems it would be advantageous to run one calculation of SOS data for a large number of basis functions and save the data.
        Then the ACF integration code could be called multiple times, and compared against the SOS data
        simply by loading the data from the file and providing it in arguments `sos_y_array`, `sos_x_array`.

        The plot is saved to a file at the path `output_path` + "ACF_{file_name}.png"
        an example: \home\username\plots\ACF_h2o_linear.png
        IF the `plot_log` boolean flag is true an additional plot with logarithmic scales is saved to `output_path` + "ACF_{file_name}_logscale.png"
        an example: \home\username\plots\ACF_h2o_linear_logscale.png
        """

        # if no alternative data is provided use the internal data for plotting
        if acf_y_array is None:
            acf_y_array = self.C_tau_cc_ABS
        if acf_x_array is None:
            acf_x_array = self.t_cc
        if sos_y_array is None and sos_flag:
            sos_y_array = self.C_tau_sos
        if sos_x_array is None and sos_flag:
            sos_x_array = self.t_sos

        # set style for all text in plot
        mpl.rc('font', **{'family': 'sans-serif', 'weight': 'normal', 'size': 20})

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"ACF of {file_name}", fontweight='bold')

        # if plotting SOS we need two different subplots
        if sos_flag:
            ax_real = fig.add_subplot(211)
            ax_imag = fig.add_subplot(212)
        else:
            ax_real = fig.add_subplot(111)
            ax_imag = ax_real  # just a pointer to `ax_real`

        if not sos_flag:
            ax_real.plot(
                acf_x_array, acf_y_array.real, label='Real component of CC',
                linewidth=3.0, color='black', alpha=0.7,
            )
            ax_imag.plot(
                acf_x_array, acf_y_array.imag, label='Imaginary component of CC',
                linewidth=3.0, color='black', alpha=0.7, ls='-.',
            )
        else:

            # for debugging we might also plot the SOS results
            ax_real.plot(
                sos_x_array, sos_y_array.real, label='Real component of SOS',
                linewidth=3.0, color='red', alpha=0.6,
            )
            ax_imag.plot(
                sos_x_array, sos_y_array.imag, label='Imaginary component of SOS',
                linewidth=3.0, color='red', alpha=0.6,
            )



            # plot the CC after so its plotted 'on top of' the red SOS for better visibility
            ax_real.plot(
                acf_x_array, acf_y_array.real, label='Real component of CC',
                linewidth=3.0, color='black', alpha=1.0, ls='-.',
            )
            ax_imag.plot(
                acf_x_array, acf_y_array.imag, label='Imaginary component of CC',
                linewidth=3.0, color='black', alpha=1.0, ls='-.',
            )

        # set x labels
        ax_real.set_xlabel(r'''$\tau \,\slash\, \hbar$''')
        ax_imag.set_xlabel(r'''$\tau \,\slash\, \hbar$''')
        # set y labels
        ax_real.set_ylabel(r'''C($\tau \,\slash\, \hbar$)''')
        ax_imag.set_ylabel(r'''C($\tau \,\slash\, \hbar$)''')
        # create legend
        ax_real.legend()
        ax_imag.legend()

        # if using long propagation(> 100fs) we can zoom in on the end
        if long_propagation_zoom:
            x1, x2 = ax_real.get_xlim()
            x1 += abs(x1)
            x2 -= abs(x1)
            ax_real.set_xlim([x2 - 1, x2])
            ax_imag.set_xlim([x2 - 1, x2])

        # save plot
        plot_path = join(output_path, f"ACF_{file_name}.png")
        fig.savefig(plot_path)

        # save log plot
        if plot_log:
            ax_real.set_yscale('log')
            ax_imag.set_yscale('log')
            plot_path = join(output_path, f"ACF_{file_name}_logscale.png")
            fig.savefig(plot_path)

        plt.close(fig)
        return

    def plot_acf_and_save_data(self, file_name, sos_flag=False, output_path="./", **kwargs):
        """ Wrapper function for `plot_acf`, `save_acf_data`, and `save_sos_data`."""
        self.plot_acf(file_name, sos_flag, output_path, **kwargs)
        self.save_acf_data(file_name, output_path=output_path, **kwargs)
        if sos_flag:
            self.save_sos_data(file_name, output_path=output_path, **kwargs)
        return

    @classmethod
    def class_smooth_acf_using_interpolation(cls, time, acf, t_init=0., t_final=10., nof_points=10000, kind='cubic'):
        """ use interpolation to get evenly spaced points """
        f_real = interp1d(time, acf.real, kind)
        f_imag = interp1d(time, acf.imag, kind)

        # first we generate a reasonable dt step
        # if we use endpoint=True then its very difficult to get good deltas
        _, dt = np.linspace(t_init, t_final, num=nof_points, endpoint=False, retstep=True)
        time_step = round(dt, ndigits=2)
        print(f"Time step is {time_step:12.8f}")

        # then we make sure we generate uniformly spaced points
        # we use (t_final+time_step) because np.arange doesn't include the last point
        # so if we want to include t_final (we do) then we need to make the 'last point'
        # one step after the point that we want to be the real 'last point'
        new_x = np.arange(t_init, t_final+time_step, step=time_step, dtype=float)

        new_acf = np.zeros_like(new_x, dtype=complex)
        new_acf.real = f_real(new_x)
        new_acf.imag = f_imag(new_x)
        return new_x, new_acf

    def smooth_acf_using_interpolation(self, nof_points=10000, kind='cubic'):
        """ use interpolation to get evenly spaced points """
        x = self.t_cc
        y_real = self.C_tau_cc.real
        y_imag = self.C_tau_cc.imag

        f_real = interp1d(x, y_real, kind)
        f_imag = interp1d(x, y_imag, kind)

        t_init, t_final = x[0], x[-1]

        # first we generate a reasonable dt step
        # if we use endpoint=True then its very difficult to get good deltas
        _, dt = np.linspace(t_init, t_final, num=nof_points, endpoint=False, retstep=True)
        time_step = round(dt, ndigits=2)
        print(f"Time step is {time_step:12.8f}")

        # then we make sure we generate uniformly spaced points
        # we use (t_final+time_step) because np.arange doesn't include the last point
        # so if we want to include t_final (we do) then we need to make the 'last point'
        # one step after the point that we want to be the real 'last point'
        new_x = np.arange(t_init, t_final+time_step, step=time_step, dtype=float)

        self.t_cc = new_x.copy()
        self.C_tau_cc = np.zeros_like(new_x, dtype=complex)
        self.C_tau_cc.real = f_real(new_x)
        self.C_tau_cc.imag = f_imag(new_x)
        return

    def _init_qs_1d(self,):
        
        N, A = self.N, self.A  # for brevity

        # define coefficient tensors
        self.q = {
            (0, 0): np.zeros((A, A), dtype=complex),
            (0, 1): np.zeros((A, A, N), dtype=complex),
            (1, 0): np.zeros((A, A, N), dtype=complex),
            (1, 1): np.zeros((A, A, N, N), dtype=complex),
            (0, 2): np.zeros((A, A, N, N), dtype=complex),
            (2, 0): np.zeros((A, A, N, N), dtype=complex),
            
            # NEW 3 hamiltonian 
            (0, 3): np.zeros((A, A, N, N, N), dtype=complex),
            (3, 0): np.zeros((A, A, N, N, N), dtype=complex),
            (1, 2): np.zeros((A, A, N, N, N), dtype=complex),
            (2, 1): np.zeros((A, A, N, N, N), dtype=complex),
            
            # New 4 Hamiltonian
            (1, 3): np.zeros((A, A, N, N, N, N), dtype=complex),
            (3, 1): np.zeros((A, A, N, N, N, N), dtype=complex),
            (0, 4): np.zeros((A, A, N, N, N, N), dtype=complex),
            (4, 0): np.zeros((A, A, N, N, N, N), dtype=complex),
            (2, 2): np.zeros((A, A, N, N, N, N), dtype=complex),
        }
        
        self.q2 = {
            (0, 0): np.zeros((A, A), dtype=complex),
            (0, 1): np.zeros((A, A, N), dtype=complex),
            (1, 0): np.zeros((A, A, N), dtype=complex),
            (1, 1): np.zeros((A, A, N, N), dtype=complex),
            (0, 2): np.zeros((A, A, N, N), dtype=complex),
            (2, 0): np.zeros((A, A, N, N), dtype=complex),
            
            # NEW 3 hamiltonian 
            (0, 3): np.zeros((A, A, N, N, N), dtype=complex),
            (3, 0): np.zeros((A, A, N, N, N), dtype=complex),
            (1, 2): np.zeros((A, A, N, N, N), dtype=complex),
            (2, 1): np.zeros((A, A, N, N, N), dtype=complex),
            
            # New 4 Hamiltonian
            (1, 3): np.zeros((A, A, N, N, N, N), dtype=complex),
            (3, 1): np.zeros((A, A, N, N, N, N), dtype=complex),
            (0, 4): np.zeros((A, A, N, N, N, N), dtype=complex),
            (4, 0): np.zeros((A, A, N, N, N, N), dtype=complex),
            (2, 2): np.zeros((A, A, N, N, N, N), dtype=complex),
        }
        
        self.q[(1,0)] += 1/np.sqrt(2)
        self.q[(0,1)] += 1/np.sqrt(2)
        
        self.q2[(0,0)] += 1/2
        self.q2[(1,1)] += 1
        self.q2[(2,0)] += 1
        self.q2[(0,2)] += 1
        
        # the operator is given in normal order so maybe we include normal ordering pre coeeficents 
        if self.normal_order == True:
            self.q2[(2,0)] *= 1
            self.q2[(0,2)] *= 1
            # breakpoint()
        return
    
    def eval_qs_1d(self,T_dict,Z_bar_dict):#,H_args):
        
        # represent_vector_zbar in CI basis
        ci_vector = [] 
        ci_basis_size = 1
        
        z_elem = Z_bar_dict[(0,0)].flatten()
        ci_vector.append(z_elem)
        
        if self.Z_truncation_order >= 1:
            z_elem = Z_bar_dict[(1,0)].flatten()
            ci_vector.append(z_elem)
            ci_basis_size += 1
        if self.Z_truncation_order >= 2:
            z_elem = (1/np.sqrt(2))*Z_bar_dict[(2,0)].flatten()
            ci_vector.append(z_elem)
            ci_basis_size += 1
        if self.Z_truncation_order >= 3:
            z_elem = (1/np.sqrt(6))*Z_bar_dict[(3,0)].flatten()
            ci_vector.append(z_elem)
            ci_basis_size += 1
        if self.Z_truncation_order >= 4:
            z_elem = (1/np.sqrt(24))*Z_bar_dict[(4,0)].flatten()
            ci_vector.append(z_elem)
            ci_basis_size += 1
        
        ci_vector = np.array(ci_vector).flatten()
        ci_vector_conj = np.conj(ci_vector).flatten()
        
        # double similarity transform  
        # follows convention from hamiltonian
        q_bar =  self._similarity_trans_Oz(self.q, T_dict) 
        q_tilde = self._cal_H_bar_tilde_Oz(q_bar,T_dict)
        
        q2_bar =  self._similarity_trans_Oz(self.q2, T_dict) 
        q2_tilde = self._cal_H_bar_tilde_Oz(q2_bar,T_dict)
        
        # construct matricies in CI/HO basis 
        q_tilde_mat = self._1d_operator_construction_HO_basis(q_tilde,ci_basis_size) 
        q2_tilde_mat = self._1d_operator_construction_HO_basis(q2_tilde,ci_basis_size)   

        # initial displacement coefficent 
        t_init = {1 : np.zeros((self.N),
                               complex)}
        t_init[1] += self.init_t
        s_0_init =  np.exp(-np.vdot(t_init[1],t_init[1]))
        
        t_0_comp = np.exp(T_dict[0]+T_dict[0].conj())
        t_1_comp = np.exp(np.vdot(T_dict[1],T_dict[1]))
        t_factors = t_0_comp*t_1_comp*s_0_init
       
        # norm
        ci_denom = np.dot(ci_vector_conj,ci_vector)
        ci_denom *= t_factors
                 
        ev_q = (ci_vector_conj@q_tilde_mat@ci_vector)
        ev_q *= t_factors
        
        ev_q2 = (ci_vector_conj@q2_tilde_mat@ci_vector)
        ev_q2 *= t_factors 
        
        if self.include_norm == True:
            ev_q /= ci_denom
            ev_q2 /= ci_denom
        # print(T_dict[1])  
        # breakpoint()
        return(ev_q, ev_q2)#,ci_denom)#,exp_H) 
    
    def eval_norm_energy(self,T_dict,Z_bar_dict,H_args):
        A,N = self.A, self.N
        
        # initial normalization factor
        t_init = {1 : np.zeros((self.N),
                               complex)}
        t_init[1] += self.init_t
        s_0_init =  np.exp(-np.vdot(t_init[1],t_init[1]))
        # scalar coefficents
        t_0_comp = np.exp(T_dict[0]+T_dict[0].conj())
        t_1_comp = np.exp(np.vdot(T_dict[1],T_dict[1]))
        t_factors = t_0_comp*t_1_comp*s_0_init
        # print(Z_bar_dict[(0,0)])
        # breakpoint()
        norm_z =  np.einsum('a,a->a',Z_bar_dict[(0,0)],Z_bar_dict[(0,0)].conj())
        if self.Z_truncation_order >= 1:
            norm_z += np.einsum('ai,ai->',Z_bar_dict[(1,0)],Z_bar_dict[(1,0)].conj())
        if self.Z_truncation_order >= 2:
            norm_z += (1/2)*np.einsum('aij,aij->a',Z_bar_dict[(2,0)],Z_bar_dict[(2,0)].conj())
        if self.Z_truncation_order >= 3:
            norm_z += (1/6)*np.einsum('aijk,aijk->a',Z_bar_dict[(3,0)],Z_bar_dict[(3,0)].conj())
        if self.Z_truncation_order >= 4: 
            norm_z += (1/24)*np.einsum('aijkl,aijkl->a',Z_bar_dict[(4,0)],Z_bar_dict[(4,0)].conj())
        norm_z *= t_factors
        # breakpoint()
        # Energy otherway 
        HZtilde_non_sym = {0:np.zeros(shape=(A, ), dtype=complex),
                    1:np.zeros(shape=(A, N,),dtype=complex), 
                    2:np.zeros(shape=(A, N, N),dtype=complex), 
                    3:np.zeros(shape=(A, N, N, N),dtype=complex), 
                    4:np.zeros(shape=(A, N, N, N, N),dtype=complex),}
        HZtilde = {}
        exp_H = 0 
        if self.Z_truncation_order >= 1:
            z_four_eqns.add_m0_n0_HZ_terms(
                HZtilde_non_sym[0], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_dict, H_args, Z_bar_dict
            )
            
            HZtilde[0] = HZtilde_non_sym[0]
            exp_H += np.einsum('a,a->a',Z_bar_dict[(0,0)].conj(),HZtilde[0])  
            
            z_four_eqns.add_m0_n1_HZ_terms(
                HZtilde_non_sym[1], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_dict, H_args, Z_bar_dict
            )
           
            HZtilde[1] = HZtilde_non_sym[1]
            exp_H += np.einsum('ai,ai->a',Z_bar_dict[(1,0)].conj(),HZtilde[1])
            
        if self.Z_truncation_order >= 2:
            z_four_eqns.add_m0_n2_HZ_terms(
                HZtilde_non_sym[2], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_dict, H_args, Z_bar_dict
            )
            HZtilde[2] = symmetrize_tensor(N,HZtilde_non_sym[2],2)
            exp_H += (1/2)*np.einsum('aij,aij->a',Z_bar_dict[(2,0)].conj(),HZtilde[2])
            
        if self.Z_truncation_order >= 3:
            z_four_eqns.add_m0_n3_HZ_terms(
                HZtilde_non_sym[3], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_dict, H_args, Z_bar_dict
            )
            HZtilde[3] = symmetrize_tensor(N,HZtilde_non_sym[3],3)
            exp_H += (1/6)*np.einsum('aijk,aijk->a',Z_bar_dict[(3,0)].conj(),HZtilde[3])
        
        if self.Z_truncation_order >= 4:
            z_four_eqns.add_m0_n4_HZ_terms(
                HZtilde_non_sym[4], # the residual functions just updates the dictionary entries in place
                self.ansatz, self.gen_trunc,
                T_dict, H_args, Z_bar_dict
            )
            HZtilde[4] = symmetrize_tensor(N,HZtilde_non_sym[4],4)
            exp_H += (1/24)*np.einsum('aijkl,aijkl->a',Z_bar_dict[(4,0)].conj(),HZtilde[4])
        
        exp_H *= t_factors
        
        if self.include_norm == True:
            exp_H /= norm_z
        # breakpoint()
        return(norm_z,exp_H)
    
    def _1d_operator_construction_HO_basis(self,oper,HO_basis_size):
        # this is the exact code used to construct the basis of the hamiltonian
        # i have only removed the extra dimensionality associates with self.h and have made replaced self
        # self.h with a dict with the correct array array size for 1d 
        oper_mat = np.zeros([HO_basis_size, HO_basis_size], dtype=complex)
        for a_1, b_1 in it.product(range(HO_basis_size),range(HO_basis_size)):
            index = (a_1,b_1)

            if self.trunc.at_least_linear:
                # hamiltonian terms associated with H(1, 0)
                if a_1 == b_1 + (1):
                    oper_mat[index] += oper[(1, 0)]*np.sqrt(b_1 + 1)
                # hamiltonian terms associated with H(0, 1)
                if a_1 == b_1 + (-1):
                    oper_mat[index] += oper[(0, 1)]*np.sqrt(b_1 + 0)            
                # hamiltonian terms associated with H(1, 1), H(0, 0) 
                if a_1 == b_1 + (0):

                    oper_mat[index] += oper[(0,0)] 
                    oper_mat[index] += 1*oper[(1, 1)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0)
            # breakpoint()
            if self.trunc.at_least_quadratic:
                
                # hamiltonian terms associated with H(2, 0)
                if a_1 == b_1 + (2):
                    oper_mat[index] += 0.5*oper[(2, 0)]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)
                # hamiltonian terms associated with H(0, 2)
                if a_1 == b_1 + (-2):
                    oper_mat[index] += 0.5*oper[(0, 2)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1)
            
            if self.trunc.at_least_cubic:
                # hamiltonian terms associated with H(3, 0)
                if a_1 == b_1 + (3):
                    oper_mat[index] += 0.16666666666666666*oper[(3, 0)]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3)
                # hamiltonian terms associated with H(0, 3)
                if a_1 == b_1 + (-3):
                    oper_mat[index] += 0.16666666666666666*oper[(0, 3)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2)
                # hamiltonian terms associated with H(1, 2)
                if a_1 == b_1 + (-1):
                    oper_mat[index] += 0.5*oper[(1, 2)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -1)
                # hamiltonian terms associated with H(2, 1)
                if a_1 == b_1 + (1):
                    oper_mat[index] += 0.5*oper[(2, 1)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1)
            
            if self.trunc.at_least_quartic:
                # hamiltonian terms associated with H(4, 0)
                if a_1 == b_1 + (4):
                    oper_mat[index] += 0.041666666666666664*oper[(4, 0)]*np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2) * np.sqrt(b_1 + 3) * np.sqrt(b_1 + 4)
                # hamiltonian terms associated with H(0, 4)
                if a_1 == b_1 + (-4):
                    oper_mat[index] += 0.041666666666666664*oper[(0, 4)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -3)
                # hamiltonian terms associated with H(2, 2)
                if a_1 == b_1 + (0):
                    oper_mat[index] += 0.25*oper[(2, 2)]*np.sqrt(b_1 + 0) * (b_1 + -1) * np.sqrt(b_1 + 0)
                # hamiltonian terms associated with H(1, 3)
                if a_1 == b_1 + (-2):
                    oper_mat[index] += 0.16666666666666666*oper[(1, 3)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + -1) * np.sqrt(b_1 + -2) * np.sqrt(b_1 + -2)
                # hamiltonian terms associated with H(3, 1)
                if a_1 == b_1 + (2):
                    oper_mat[index] += 0.16666666666666666*oper[(3, 1)]*np.sqrt(b_1 + 0) * np.sqrt(b_1 + 0) * np.sqrt(b_1 + 1) * np.sqrt(b_1 + 2)

         
        return(oper_mat)