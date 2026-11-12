def zeroth_dT_alter(self,derivatives:dict,R:dict,T_dict:dict)-> dict:
    """_Function solves for the derivative of T_0 amplitudes 

    Parameters
    ----------
    derivatives : dict
        dictionary of derivative values 
    R : dict
        dictionary of residuals (H^tilde Z_bar)_conn 
    T_dict : dict
        dictionary of current time step T operators 

    Returns
    -------
    dict
        derivative values 
    """       
    
    T_1_conj_dict = { 1 : np.conj(T_dict[1])}
    #dT_0 = -1j*R[0]-np.einsum('i,i -> ',T_1_conj_dict[1],derivatives['dT_1']) 
    dT_0 = -1j*R[0]-np.einsum('i,i -> ',T_1_conj_dict[1],derivatives['dT_1'])
    
    derivatives['dT_0'] = dT_0[0]
    
    return derivatives

def linear_dT_alter(self,Z_bar:dict,derivatives:dict,R:dict) -> dict:
    """Function solves for the derivative of T_1 amplitudes in time dependent mixed CI/Coupled cluster theory, 
    by solving a set of linear equations posed by equation number in notes. it updates the derivative dictionary for 
    the dt_1 key

    Parameters
    ----------
    Z_bar : dict
        ditcionary of Similarity transformed Z matrix 
    derivatives : dict
        dictionary of the arrays for each derivative 
    R : dict
        Resdiual GW dictionary for for all projections

    Returns
    -------
    dict
        Updated derivative dictionary 
    """     
    A,N = self.A,self.N
    if self.Z_truncation_order == 1:
        # W is 0 its straightforward
        derivatives['dT_1'] = -1j*R[1][0,:] # check that its not activated
    
    test = False
    
    if self.Z_truncation_order >=2 and test==False:
                
        # TODO ensure block_T_1 is a proper 1d array with imag and real terms collated
            
        # real component of Z^2 matrix   
        real_Z_2 = Z_bar[(2,0)].real.copy()[0] # remove extra dimension from surfaces  
        
        # complex Z^2 matrix 
        imag_Z_2 = Z_bar[(2,0)].imag.copy()[0]
        
        # build block matrix Z_bar
        # block_W = np.block(([[imag_Z_2, -real_Z_2+np.eye(real_Z_2.shape[0])],
        #                     [-1j*(real_Z_2-np.eye(real_Z_2.shape[0])), -1j*imag_Z_2]])) 
        block_mat_W = np.block(([[imag_Z_2,-(np.eye(real_Z_2.shape[0])+real_Z_2)],
                                    [(np.eye(real_Z_2.shape[0])-real_Z_2),-imag_Z_2]]))
        
        # Surrogate R_1 
        R_1 = R[1][0]  # remove extra dimension from surfaces 
        
        # build block matrix R 
        real_R = R_1.real.copy()
        imag_R = R_1.imag.copy()
        block_R = np.block([real_R,imag_R])
                    
        # Solve for T_1 amps
        block_dT_1 = np.linalg.solve(block_mat_W,block_R) 
        real_block_dT_1 = block_dT_1[:len(real_R)]
        imag_block_dT_1 = 1j*block_dT_1[len(real_R):]
        
        # Change block mat T_1 to regular 1d array 
        dT_1 = real_block_dT_1+imag_block_dT_1 
        derivatives['dT_1'] = dT_1
    return derivatives

def derivative_functions_alter(self,H_tilde,Z_bar,T_dict):
    A = self.A
    N = self.N        
    
    # residual dictionary
    residual = {0:np.zeros(shape=(A, ), dtype=complex),
                1:np.zeros(shape=(A, N,),dtype=complex), 
                2:np.zeros(shape=(A, N, N),dtype=complex), 
                3:np.zeros(shape=(A, N, N, N),dtype=complex), 
                4:np.zeros(shape=(A, N, N, N, N),dtype=complex),}
    
    derivatives = {}
    
    for i in range(self.T_truncation_order+1):
        derivatives[f'dT_{i}'] = None
        
    for i in range(2, self.Z_truncation_order+1):
        derivatives[f'dZ_{i}'] = None
    
    _special_T_conj = {
                1: np.conj(T_dict[1]),
            }
            
    # Update derivative dictionarys
    # find dT_1
    z_three_eqns.add_m0_n1_HZ_terms(
                        residual[1], # the residual functions just updates the dictionary entries in place 
                        self.ansatz, self.gen_trunc,
                        _special_T_conj, H_tilde, Z_bar
                    )
    
    derivatives = self.linear_dT(Z_bar,derivatives,residual)
    
    # find dT_0
    z_three_eqns.add_m0_n0_HZ_terms(
                        residual[0], # the residual functions just updates the dictionary entries in place
                        self.ansatz, self.gen_trunc,
                        _special_T_conj, H_tilde, Z_bar
                    )
    derivatives = self.zeroth_dT(derivatives,residual,T_dict)
    
    # find dZ_2
    if self.Z_truncation_order >= 2:
        # update our hamiltonian   
        G = {}
        G[(0,0)] = -1j*H_tilde[(0,0)] - derivatives['dT_0'] - np.einsum('i,i -> ',_special_T_conj[1],derivatives['dT_1'])
        G[(0,1)] = -1j*H_tilde[(0,1)] - np.conj(derivatives['dT_1'])
        G[(1,0)] = -1j*H_tilde[(1,0)] - derivatives['dT_1']
        G[(0,2)] = -1j*H_tilde[(0,2)]
        G[(2,0)] = -1j*H_tilde[(2,0)]
        G[(1,1)] = -1j*H_tilde[(1,1)]
        
        z_three_eqns.add_m0_n2_HZ_terms(
                        residual[2], self.ansatz, self.gen_trunc,
                        _special_T_conj, G, Z_bar
                    )
        derivatives['dZ_2'] = residual[2]
        
    # find dZ_3
    if self.Z_truncation_order >=3:
            
        z_three_eqns.add_m0_n3_HZ_terms(
                        residual[3], self.ansatz, self.gen_trunc,
                        _special_T_conj, G, Z_bar
                    )
        derivatives['dZ_3'] = residual[3]  
    
    # # apply conjugation?
    # for key in derivatives.keys():
    #     derivatives[key] *= -1j

    return derivatives
