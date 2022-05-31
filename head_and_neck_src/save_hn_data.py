import os
import glob
import matlab.engine
from scipy.ndimage import zoom
import numpy as np
import nibabel as nib
import sys
import time
import ray


from matRad_wo_hn_env import MatRadMetaOpt
from run_meta_optimization_framework import get_meta_optimized_weights



def load_final_plan(sample_case_num = None,
                                data_parent_dir = '',
                                results_dir = '',
                                weights = None,
                                high_res = True,
                                vmat = False,
                                vmat_traj_dir = ''):

    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir,'*')) if os.path.isdir(i)])
    num_cases = len(case_list)
    if sample_case_num is None:
        np.random.seed(0)
        sample_case_num = int(np.random.uniform(0,num_cases))

    ctDir = os.path.join(case_list[sample_case_num],'CT')
    rtStDir = os.path.join(case_list[sample_case_num],'RTst')
    rtDoseDir = os.path.join(case_list[sample_case_num],'RTDOSE')
    rtPlanDir = os.path.join(case_list[sample_case_num],'RTPLAN')

    print(ctDir)

    traj_path = os.path.join(results_dir,'trajectory_dict_for_case_num_'+str(sample_case_num)+'.npy')
    if os.path.exists(traj_path):
        trajectory_dict = np.load(traj_path,allow_pickle = True).item()
        couch_angle_list = trajectory_dict['sampled_path_couch_angles']
        gantry_angle_list = trajectory_dict['sampled_path_gantry_angles']
    else:
        couch_angle_list = [0,0,0,0,0,0,0,0,0]
        gantry_angle_list = [0,40,80,120,160,200,240,280,320]

    matrad_env = MatRadMetaOpt()
    if weights is None:
        results_dict = np.load(os.path.join(results_dir,'results_for_case_num_'+str(sample_case_num)+'.npy'),allow_pickle = True).item()
        meta_optimized_weights = np.array(results_dict['final_projected_simplex'][0])
    else:
        meta_optimized_weights = np.array(weights)
    if vmat:
        vmat_traj_path = os.path.join(vmat_traj_dir,
                                      'vmat_trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')
        if os.path.exists(vmat_traj_path):
            vmat_trajectory_dict = np.load(vmat_traj_path, allow_pickle=True).item()
            vmat_control_points_couch = vmat_trajectory_dict['vmat_control_points_couch']
            vmat_control_points_gantry = vmat_trajectory_dict['vmat_control_points_gantry']
            vmat_fmo_couch_angles = vmat_trajectory_dict['vmat_fmo_couch_angles']
            vmat_fmo_gantry_angles = vmat_trajectory_dict['vmat_fmo_gantry_angles']

            matrad_env = MatRadMetaOpt()
            matrad_env.reset_pops(ctDir,
                                  rtStDir,
                                  high_res=high_res,
                                  acceptable_iter=600,
                                  acceptable_constr_viol_tol=0.0001,
                                  ipopt_max_iters=600,
                                  vmat=True,
                                  additional_vmat_couch_angles=vmat_control_points_couch,
                                  additional_vmat_gantry_angles=vmat_control_points_gantry,
                                  additional_fmo_angles=vmat_fmo_gantry_angles)

        else:
            matrad_env.reset_pops(ctDir,
                                  rtStDir,
                                  high_res=high_res,
                                  acceptable_iter=600,
                                  acceptable_constr_viol_tol=0.0001,
                                  ipopt_max_iters=600,
                                  vmat=True)

        matrad_env.wo_projection_no_score(meta_optimized_weights)
        matrad_env.run_seq_and_dao()
    else:
        matrad_env.reset_pops(ctDir,
                              rtStDir,
                              couch_angle_list = couch_angle_list,
                              gantry_angle_list = gantry_angle_list,
                              high_res=high_res,
                              acceptable_iter= 200,
                              acceptable_constr_viol_tol = 0.001,
                              ipopt_max_iters = 200)
        matrad_env.wo_projection_no_score(meta_optimized_weights)


    matrad_env.get_ci_and_hi()
    matrad_env.get_r50_and_r90()
    matrad_env.render(True)
    time.sleep(60)

    return matrad_env

def run_on_dataset(data_folder = '../autoplan_database/dicom_data/head_and_neck_dicoms',
                   output_folder = '../autoplan_database/matrad_data/head_and_neck',
                   weights_folder = '../autoplan_database/meta_optimization_results_hn/'):
    case_list = sorted([i for i in glob.glob(os.path.join(data_folder, '*')) if os.path.isdir(i)])
    num_cases = len(case_list)
    print('Num cases: ', num_cases)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for case_num in range(num_cases):
        env = load_final_plan(sample_case_num=case_num,
                                          data_parent_dir=data_folder,
                                          results_dir=weights_folder,
                                          high_res=True,
                                          vmat=True)
        output_file = str(case_num)+'.mat'
        output_file = os.path.join(output_folder,output_file)
        env.eng.eval('save("'+output_file+'")',nargout=0)


def run_metaplanner_on_folder(data_folder = '../autoplan_database/dicom_data/head_and_neck_dicoms',
                              meta_optimization_results_dir = '../autoplan_database/meta_optimization_results_hn/'):
    '''
    Run meta-optimization framework
    '''

    get_meta_optimized_weights(data_parent_dir=data_folder,
                               output_directory=meta_optimization_results_dir)


if __name__ == '__main__':
    os.environ['matRad'] = '../matRad-dev_VMAT/'
    run_metaplanner_on_folder()
    run_on_dataset()
