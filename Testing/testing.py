import sys
import os
from os.path import abspath, join, dirname, basename
import argparse
import numpy as np
# from project.vibronic import vIO, VMK, model_op

# import the path to the package
project_dir = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, project_dir)
print(os.getcwd())
import project
import test_funcs as tf 
# breakpoint()

# tasks we can do in this file 
parser = argparse.ArgumentParser(description='Multi-process script')
subparsers = parser.add_subparsers(dest='command', help='Available commands')

# run spectra for CC 
CC_prop_parser = subparsers.add_parser('CCP',help='propagates CC')
CC_prop_parser.add_argument('-f',dest='model_name',help='cc filename')
CC_prop_parser.add_argument('-o',type=int,dest='order',default=1,help='hamiltonian order')
CC_prop_parser.add_argument('-z',type=int,dest='nZ',default=1,help='CI operator Z order')
CC_prop_parser.add_argument('-t',type=float,dest='tf',default=10,help='time of propagation')
CC_prop_parser.add_argument('-e',type=str,dest='ext',default=None,help='extra info for project')
CC_prop_parser.add_argument('-d',type=str,dest='dist',default=0,help='displacment')
CC_prop_parser.add_argument('-sd',type=float,nargs='*',dest='spec_dist',default=[],help='displacment mode specific')
CC_prop_parser.add_argument('-zchk',dest='zchk',default=False,action='store_true')
CC_prop_parser.add_argument('-q',dest='q_eval',default=False,action='store_true')
CC_prop_parser.add_argument('-n',dest='norm',default=False,action='store_true')
CC_prop_parser.add_argument('-dir',type=str,dest='outdir',default='../Results',help='extra info for project')
     
    
CC_EEED_parser = subparsers.add_parser('CCEEED', help='CC EED for eigenvalue comparison')
CC_EEED_parser.add_argument('-f',dest='model_name',help='model name')
CC_EEED_parser.add_argument('-o',type=int,dest='order',default=1,help='H order')
CC_EEED_parser.add_argument('-t',type=float,dest='tf',default=10)
CC_EEED_parser.add_argument('-HO',type= int,dest='basis_size',default=10)
CC_EEED_parser.add_argument('-q',dest='q_eval',default=False,action='store_true')
CC_EEED_parser.add_argument('-s',dest='sub',default=False,action='store_true')
CC_EEED_parser.add_argument('-e',type=str,dest='ext',default=None,help='extra info for project')
CC_EEED_parser.add_argument('-dir',type=str,dest='outdir',default='../Results',help='extra info for project')

# Check double similarity transform
CC_dsim_parser = subparsers.add_parser('dsim', help='performs check for double similarity transform')
CC_dsim_parser.add_argument('-f',dest='model_name',help='model name')
CC_dsim_parser.add_argument('-o',type=int,dest='order',default=1,help='H order')
CC_dsim_parser.add_argument('-t1',type=float,default=.25)
CC_dsim_parser.add_argument('-e',dest='ext',default=None)
CC_dsim_parser.add_argument('-dir',type=str,dest='outdir',default='../Results',help='extra info for project')

# Create Hamiltonian-Residual test
Residual_test_parser = subparsers.add_parser('RT',help='performs construction of hamiltonian')
Residual_test_parser.add_argument('-f',dest='model_name',help='cc filename')
Residual_test_parser.add_argument('-o',type=int,dest='order',default=1,help='hamiltonian order')
Residual_test_parser.add_argument('-z',type=int,dest='nZ',default=1,help='CI operator Z order')
Residual_test_parser.add_argument('-dir',type=str,dest='outdir',default='../Results',help='extra info for project')


#------------------------------------------------------
# Main work 
#------------------------------------------------------


args = parser.parse_args()


order_dict = {
    0: "constant",
    1: "linear",
    2: "quadratic",
    3: "cubic",
    4: "quartic",
}

# finding the hamiltonian file 
hamiltonian_file_path ='../CC_files/json_files'

# initialize model
hamiltonian_file_path = join(hamiltonian_file_path,f'{args.model_name}.json')
model = tf.get_model_from_json_file(hamiltonian_file_path, args.order)

# output path for generated material
output_path = args.outdir

# run cmds
if (__name__ == '__main__'):
        
    if args.command == 'CCEEED':

        # location of results
        if args.ext == None:
            output_dir = join(output_path,f'model_{args.model_name}_rank_{args.order}_basis_{args.basis_size}_time_{int(args.tf)}')
        else: 
            output_dir = join(output_path,f'model_{args.model_name}_rank_{args.order}_basis_{args.basis_size}_time_{int(args.tf)}_ext_{args.ext}')    
        os.makedirs(output_dir,exist_ok=True)
       
        H_mat = tf.generate_ed_results(model,args.model_name,args.order,
                                       args.basis_size,args.tf,args.sub,q_eval=args.q_eval,
                                       ext=args.ext,output_path=output_dir)
    
    
    # propagation CC
    if args.command == 'CCP':
        
        # location of results
        if args.ext == None:
            output_dir = join(output_path,f'model_{args.model_name}_rank_{args.order}_Z_{args.nZ}_time_{int(args.tf)}_dist_{args.dist}')
        else: 
            output_dir = join(output_path,f'model_{args.model_name}_rank_{args.order}_Z_{args.nZ}_time_{int(args.tf)}_dist_{args.dist}_ext_{args.ext}')    
        os.makedirs(output_dir,exist_ok=True)
        
        normal_path = tf.generate_cc_acf_data_oz(model, args.model_name, args.order, nZ =args.nZ, t_final=args.tf,ext=args.ext,dist=args.dist,specd=args.spec_dist,
                                                 output_path=output_dir,q=args.q_eval,z_chk=args.zchk,norm=args.norm)
        print('complete')
        
   
    
    # Double similarly test 
    if args.command == 'dsim':

        # location of results
        os.makedirs(arg.outdir,exist_ok=True)
        output_filename = f'model_{args.model_name}_{args.order}_t1_{args.t1}_ext_{args.ext}.txt'
        output_file_path = join(output_path,output_filename) 
        
        double_sim_model =tf.get_doublesim_model_from_json_file(path,args.order,args.t1)
        single_sim_model =tf.get_singlesim_model_from_json_file(path,args.order,args.t1)
        tf.double_similarity_test(model,double_sim_model,single_sim_model,args.model_name,args.order,
                                  t_1 = np.array(args.t1).reshape(1),output_path=output_file_path)
        print('complete')
    
    # Matrix vector multiplication check    
    if args.command == 'RT':
        
        os.makedirs(arg.outdir,exist_ok=True)
        output_filename = f'model_{args.model_name}_{args.order}_Z_{args.nZ}.txt'
        output_file_path = join(output_path,output_filename)
        
        tf.residual_check(model,args.model_name,args.order,args.nZ,output_file_path)
        print('complete')
        
        
