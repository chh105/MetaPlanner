from matRad_wo_hn_env import *
from meta_opt_hn_env import *
from bao_hn_env import *
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import copy
from scipy import interpolate
import time
import pandas as pd
from scipy.stats import wilcoxon

import ray



def get_meta_optimized_weights(data_parent_dir,output_directory):
    n_beams = 9
    memory = 12000000000 * 9
    object_store_memory = 2500000000 * 8
    ray.init(memory=memory, object_store_memory=object_store_memory, ignore_reinit_error=True, log_to_driver=False)
    pops = POPS_algo_wo()

    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir, '*')) if os.path.isdir(i)])
    num_cases = len(case_list)

    directory = output_directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    iter_time_list = []
    setup_time_list = []

    for i in range(num_cases):
        sample_case_num = i
        file_path = os.path.join(directory, 'equispaced_trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')

        couch_angle_list = list(np.zeros(n_beams))
        gantry_angle_list = list(np.linspace(0,360,n_beams+1)[:n_beams])
        trajectory_dict = {
            'sampled_path_couch_angles': couch_angle_list,
            'sampled_path_gantry_angles': gantry_angle_list
        }
        np.save(file_path, trajectory_dict)


        print('Couch angle list:', couch_angle_list)
        print('Gantry angle list:', gantry_angle_list)
        setup_start_time = time.time()
        pops.reset(sample_case_num,
                   data_parent_dir=data_parent_dir,
                   couch_angle_list=couch_angle_list,
                   gantry_angle_list=gantry_angle_list)

        setup_time_elapsed = time.time() - setup_start_time
        print('setup time elapsed:', setup_time_elapsed)
        setup_time_list.append(setup_time_elapsed)

        done = False

        iter_start_time = time.time()
        while not done:
            obs, reward, done, info_dict = pops.step(2)

        np.save(os.path.join(directory, 'results_for_case_num_' + str(sample_case_num) + '.npy'), info_dict)

        iter_time_elapsed = time.time() - iter_start_time
        print('iter time elapsed:', iter_time_elapsed)
        iter_time_list.append(iter_time_elapsed)

    setup_time_data = np.array(setup_time_list)
    np.save(os.path.join(directory, 'setup_time_data.npy'), setup_time_data)
    print('setup time_data:', setup_time_data)
    print('avg setup time per patient:', np.sum(setup_time_data) / num_cases)

    iter_time_data = np.array(iter_time_list)
    np.save(os.path.join(directory, 'iter_time_data.npy'), iter_time_data)
    print('iter time_data:', iter_time_data)
    print('avg iter time per patient:', np.sum(iter_time_data) / num_cases)



def run_double_vmat_arc_coplanar(data_parent_dir,
            output_directory):

    memory = 12000000000*9
    object_store_memory = 2500000000*8
    ray.init(memory=memory,object_store_memory=object_store_memory, ignore_reinit_error=True, log_to_driver=False)

    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir,'*')) if os.path.isdir(i)])
    num_cases = len(case_list)

    directory = output_directory

    if not os.path.exists(directory):
        os.makedirs(directory)


    trajectory_search_time_list = []
    for i in range(num_cases):
        sample_case_num = i

        file_path = os.path.join(directory, 'trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')
        trajectory_start_time = time.time()
        couch_angle_list = list(np.zeros(12))
        gantry_angle_list = list(np.linspace(-135,225,13)[:-1])
        trajectory_dict = {
            'sampled_path_couch_angles':couch_angle_list,
            'sampled_path_gantry_angles':gantry_angle_list,
        }

        trajectory_time_elapsed = time.time() - trajectory_start_time
        print('trajectory search time elapsed:',trajectory_time_elapsed)
        trajectory_search_time_list.append(trajectory_time_elapsed)

        np.save(file_path,trajectory_dict)

        print('Couch angle list:',couch_angle_list)
        print('Gantry angle list:',gantry_angle_list)


    
    trajectory_time_data = np.array(trajectory_search_time_list)
    np.save(os.path.join(directory,'trajectory_time_data.npy'),trajectory_time_data)
    print('trajectory time_data:',trajectory_time_data)
    print('avg trajectory time per patient:',np.sum(trajectory_time_data)/num_cases)


def run_bao(data_parent_dir,
            meta_optimized_weights_dir,
            output_directory,
            use_noncoplanar=False):
    memory = 12000000000 * 9
    object_store_memory = 2500000000 * 8
    ray.init(memory=memory, object_store_memory=object_store_memory, ignore_reinit_error=True, log_to_driver=False)

    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir, '*')) if os.path.isdir(i)])
    num_cases = len(case_list)

    directory = output_directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    trajectory_search_time_list = []
    for i in range(num_cases):
        sample_case_num = i
        meta_opt_info_dict = np.load(
            os.path.join(meta_optimized_weights_dir, 'results_for_case_num_' + str(sample_case_num) + '.npy'),
            allow_pickle='TRUE').item()
        meta_optimized_weights = meta_opt_info_dict['final_projected_simplex'][0]
        sample_case_num = i
        file_path = os.path.join(directory, 'trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')
        trajectory_start_time = time.time()
        if use_noncoplanar:
            trajectory_searcher = BeamAngleOptimization(meta_optimized_weights=meta_optimized_weights,
                                                        data_dir=data_parent_dir,
                                                        sample_case_num=sample_case_num,
                                                        visualize=True,
                                                        results_dir=directory,
                                                        num_threads=6)
        else:
            trajectory_searcher = CoplanarBAO(meta_optimized_weights=meta_optimized_weights,
                                              data_dir=data_parent_dir,
                                              sample_case_num=sample_case_num,
                                              visualize=True,
                                              results_dir=directory,
                                              num_threads=6)

        # trajectory_dict = trajectory_searcher.run_simplex_search()
        trajectory_searcher.generate_cost_map()
        trajectory_dict = trajectory_searcher.get_trajectory()
        couch_angle_list = trajectory_dict['sampled_path_couch_angles']
        gantry_angle_list = trajectory_dict['sampled_path_gantry_angles']

        trajectory_time_elapsed = time.time() - trajectory_start_time
        print('trajectory search time elapsed:', trajectory_time_elapsed)
        trajectory_search_time_list.append(trajectory_time_elapsed)

        np.save(file_path, trajectory_dict)

        print('Couch angle list:', couch_angle_list)
        print('Gantry angle list:', gantry_angle_list)

    trajectory_time_data = np.array(trajectory_search_time_list)
    np.save(os.path.join(directory, 'trajectory_time_data.npy'), trajectory_time_data)
    print('trajectory time_data:', trajectory_time_data)
    print('avg trajectory time per patient:', np.sum(trajectory_time_data) / num_cases)

def run_nonuniform_sampling_coplanar(data_parent_dir,
                                    bao_dir,
                                    output_directory,
                                    vmat_starting_angle = [0,-180], # [couch,gantry]
                                    vmat_ending_angle = [0,180], # [couch,gantry]
                                    partial_arc_degrees = 32,
                                    vmat_control_point_spacing = 4):
    memory = 12000000000 * 9
    object_store_memory = 2500000000 * 8
    ray.init(memory=memory, object_store_memory=object_store_memory, ignore_reinit_error=True, log_to_driver=False)
    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir, '*')) if os.path.isdir(i)])
    num_cases = len(case_list)

    directory = output_directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    trajectory_search_time_list = []
    for i in range(num_cases):
        '''Order the imrt beams'''
        sample_case_num = i
        file_path = os.path.join(bao_dir, 'trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')
        trajectory_dict = np.load(file_path, allow_pickle=True).item()
        couch_angle_list = trajectory_dict['sampled_path_couch_angles']
        gantry_angle_list = trajectory_dict['sampled_path_gantry_angles']
        assert len(couch_angle_list) == len(gantry_angle_list)

        trajectory_start_time = time.time()
        ref_angle = np.array(vmat_ending_angle)
        ordered_couch_angle_list = []
        ordered_gantry_angle_list = []
        angle_ids = []

        for j in range(len(couch_angle_list)):
            min_dist = np.infty
            closest_angle = None
            closest_angle_id = None
            for k in range(len(couch_angle_list)):
                if k in angle_ids:
                    continue
                cg = np.array([couch_angle_list[k],gantry_angle_list[k] + 360])
                dist_from_start = np.linalg.norm(cg - np.array(ref_angle))
                if dist_from_start < min_dist:
                    min_dist = dist_from_start
                    closest_angle = cg
                    closest_angle_id = k

            ordered_couch_angle_list.append(closest_angle[0])
            ordered_gantry_angle_list.append(closest_angle[1])
            angle_ids.append(closest_angle_id)
            ref_angle = np.array(closest_angle)

        '''Add vmat control points around the angles'''
        partial_arc_degrees = np.ceil(partial_arc_degrees/vmat_control_point_spacing) * vmat_control_point_spacing
        num_control_points = int(partial_arc_degrees/vmat_control_point_spacing)
        if num_control_points % 2 ==0:
            num_control_points += 1
        print('Each VMAT arc has {} degrees, {} control points, {} degree spacing'.format(partial_arc_degrees,
                                                                                         num_control_points,
                                                                                         vmat_control_point_spacing))
        assert num_control_points % 2 != 0
        vmat_control_points_couch = []
        vmat_control_points_gantry = []
        vmat_fmo_couch_angles = ordered_couch_angle_list
        vmat_fmo_gantry_angles = ordered_gantry_angle_list
        prev_ending_angle = copy.deepcopy(vmat_ending_angle)
        for j in range(len(ordered_couch_angle_list)):
            partial_arc_starting_gantry_angle = ordered_gantry_angle_list[j] \
                                                - (num_control_points-1)//2*vmat_control_point_spacing
            offset = np.maximum(prev_ending_angle[-1] - partial_arc_starting_gantry_angle,0)
            if offset > 0:
                offset += vmat_control_point_spacing
            partial_arc_starting_gantry_angle += offset
            vmat_fmo_gantry_angles[j] += offset
            for k in range(num_control_points):
                vmat_control_points_couch.append(ordered_couch_angle_list[j]) # should be 0 for coplanar
                vmat_control_points_gantry.append(partial_arc_starting_gantry_angle \
                                                  + k*vmat_control_point_spacing)
            prev_ending_angle[-1] = copy.deepcopy(vmat_control_points_gantry[-1])

        vmat_traj_dict = {
            'vmat_control_points_couch':vmat_control_points_couch,
            'vmat_control_points_gantry':vmat_control_points_gantry,
            'vmat_fmo_couch_angles':vmat_fmo_couch_angles,
            'vmat_fmo_gantry_angles':vmat_fmo_gantry_angles,
        }

        out_path = os.path.join(directory, 'vmat_trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')
        np.save(out_path, vmat_traj_dict)

        trajectory_time_elapsed = time.time() - trajectory_start_time
        print('Nonuniform vmat time elapsed:', trajectory_time_elapsed)
        trajectory_search_time_list.append(trajectory_time_elapsed)


        print('BAO couch angle list:', ordered_couch_angle_list)
        print('BAO gantry angle list:', ordered_gantry_angle_list)
        print('VMAT couch control points:', vmat_control_points_couch)
        print('VMAT gantry control points:', vmat_control_points_gantry)

    trajectory_time_data = np.array(trajectory_search_time_list)
    np.save(os.path.join(directory, 'nonuniform_vmat_time_data.npy'), trajectory_time_data)
    print('Nonuniform vmat time data:', trajectory_time_data)
    print('avg time per patient:', np.sum(trajectory_time_data) / num_cases)


# TODO
def load_data_and_make_plots(meta_optimization_results_dir,
                             trajectory_directory,
                             data_parent_dir,
                             output_directory,
                             high_res = True):

    
    if not os.path.exists(output_directory):
            os.makedirs(output_directory)


    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir,'*')) if os.path.isdir(i)])
    num_cases = len(case_list)
    figure_count = 0

    phys_dict = {'dvh': [],
                 'ci_70': [],
                 'hi_52': [],
                 'hi_56': [],
                 'hi_70': [],
                 'r50': [],
                 'r90': [],
                 'structure_means': []}
    pops_dict = {'dvh': [],
                 'ci_70': [],
                 'hi_52': [],
                 'hi_56': [],
                 'hi_70': [],
                 'r50': [],
                 'r90': [],
                 'structure_means': [],
                 'point': []}

    results_dict = {}

    if os.path.exists(os.path.join(output_directory, 'comparison_dict.npy')):
        loaded_results_dict = np.load(os.path.join(output_directory, 'comparison_dict.npy'), allow_pickle=True).item()
        phys_dict = loaded_results_dict['physician']
        pops_dict = loaded_results_dict['pops']

    start = len(pops_dict['dvh'])

    for i in range(start, num_cases):
        sample_case_num = i
        ctDir = os.path.join(case_list[sample_case_num],'CT')
        rtStDir = os.path.join(case_list[sample_case_num],'RTst')
        rtDoseDir = os.path.join(case_list[sample_case_num],'RTDOSE')
        '''
        Physician
        '''
        matrad_env = MatRadMetaOpt()
        phys_ci_70, phys_hi_70, phys_hi_52, phys_hi_56, phys_r50_val, phys_r90_val = matrad_env.get_ptv_stats_from_treatment_plan(
            ctDir, rtStDir, rtDoseDir)
        phys_dvh = matrad_env.get_dvh_of_structs()
        phys_dvh = matrad_env.rescale_dvh(phys_dvh)
        phys_struct_means = np.array(matrad_env.get_structure_means())

        phys_dict['dvh'].append(phys_dvh)
        phys_dict['ci_70'].append(phys_ci_70)
        phys_dict['hi_52'].append(phys_hi_52)
        phys_dict['hi_56'].append(phys_hi_56)
        phys_dict['hi_70'].append(phys_hi_70)
        phys_dict['r50'].append(phys_r50_val)
        phys_dict['r90'].append(phys_r90_val)
        phys_dict['structure_means'].append(phys_struct_means)

        '''
        Meta opt
        '''
        traj_path = os.path.join(trajectory_directory, 'trajectory_dict_for_case_num_' + str(sample_case_num) + '.npy')
        if os.path.exists(traj_path):
            trajectory_dict = np.load(traj_path, allow_pickle=True).item()
            couch_angle_list = trajectory_dict['sampled_path_couch_angles']
            gantry_angle_list = trajectory_dict['sampled_path_gantry_angles']
        else:
            print('No trajectory file, loading equispaced coplanar beams instead!')
            n_beams = 9
            couch_angle_list = list(np.zeros(n_beams))
            gantry_angle_list = list(np.linspace(0, 360, n_beams + 1)[:n_beams])
            # couch_angle_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            # gantry_angle_list = [0, 40, 80, 120, 160, 200, 240, 280, 320]


        print('Couch angle list:', couch_angle_list)
        print('Gantry angle list:', gantry_angle_list)
        meta_opt_info_dict = np.load(
            os.path.join(meta_optimization_results_dir, 'results_for_case_num_' + str(sample_case_num) + '.npy'),
            allow_pickle='TRUE').item()
        meta_optimized_weights = meta_opt_info_dict['final_projected_simplex'][0]

        matrad_env = MatRadMetaOpt()
        matrad_env.reset_pops(ctDir,
                              rtStDir,
                              couch_angle_list=couch_angle_list,
                              gantry_angle_list=gantry_angle_list,
                              high_res=high_res,
                              acceptable_iter=150,
                              acceptable_constr_viol_tol=0.001,
                              ipopt_max_iters=150)

        matrad_env.wo_projection_no_score(meta_optimized_weights)
        ci_70,hi_52,hi_56,hi_70 = matrad_env.get_ci_and_hi()
        r50_val,r90_val = matrad_env.get_r50_and_r90()
        dvh = matrad_env.get_dvh_of_structs()
        dvh = matrad_env.rescale_dvh(dvh)
        struct_means = np.array(matrad_env.get_structure_means())
        pops_dict['dvh'].append(dvh)
        pops_dict['ci_70'].append(ci_70)
        pops_dict['hi_52'].append(hi_52)
        pops_dict['hi_56'].append(hi_56)
        pops_dict['hi_70'].append(hi_70)
        pops_dict['r50'].append(r50_val)
        pops_dict['r90'].append(r90_val)
        pops_dict['structure_means'].append(struct_means)
        pops_dict['point'].append(meta_optimized_weights)


        plt.figure()
        color=iter(plt.cm.rainbow(np.linspace(0,1,pops_dict['dvh'][-1].shape[0])))
        dvh_figure = plt.figure(figure_count)
        figure_count += 1
        for j in range(pops_dict['dvh'][-1].shape[0]):

            c=next(color)
            plt.plot(phys_dvh[j,:,0],phys_dvh[j,:,1],'--',c = c)
            plt.plot(pops_dict['dvh'][-1][j,:,0],pops_dict['dvh'][-1][j,:,1],'-',c = c)

        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')
        plt.xlim(-0.025,2.05)
        plt.ylim(-2.5,102.5)
        plt.title('DVH #'+str(sample_case_num))
        plt.draw()
        plt.savefig(os.path.join(output_directory,'dvh_num_'+str(sample_case_num)+'.png'))



        results_dict['physician'] = copy.deepcopy(phys_dict)
        results_dict['pops'] = copy.deepcopy(pops_dict)
        # results_dict['initial'] = init_dict
        np.save(os.path.join(output_directory, 'comparison_dict.npy'), results_dict)





def load_vmat_data_and_make_plots(meta_optimization_results_dir,
                             data_parent_dir,
                             output_directory,
                             trajectory_directory='',
                             high_res = True):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir, '*')) if os.path.isdir(i)])
    num_cases = len(case_list)
    figure_count = 0

    phys_dict = {'dvh': [],
                 'ci_70': [],
                 'hi_52': [],
                 'hi_56': [],
                 'hi_70': [],
                 'r50': [],
                 'r90': [],
                 'structure_means': []}
    pops_dict = {'dvh': [],
                 'ci_70': [],
                 'hi_52': [],
                 'hi_56': [],
                 'hi_70': [],
                 'r50': [],
                 'r90': [],
                 'structure_means': [],
                 'point': []}

    results_dict = {}

    if os.path.exists(os.path.join(output_directory, 'comparison_dict.npy')):
        loaded_results_dict = np.load(os.path.join(output_directory, 'comparison_dict.npy'),allow_pickle=True).item()
        phys_dict = loaded_results_dict['physician']
        pops_dict = loaded_results_dict['pops']

    start = len(pops_dict['dvh'])

    for i in range(start,num_cases):
        sample_case_num = i
        ctDir = os.path.join(case_list[sample_case_num], 'CT')
        rtStDir = os.path.join(case_list[sample_case_num], 'RTst')
        rtDoseDir = os.path.join(case_list[sample_case_num], 'RTDOSE')
        '''
        Physician
        '''
        matrad_env = MatRadMetaOpt()
        phys_ci_70, phys_hi_70, phys_hi_52, phys_hi_56, phys_r50_val, phys_r90_val = matrad_env.get_ptv_stats_from_treatment_plan(
            ctDir, rtStDir, rtDoseDir)
        phys_dvh = matrad_env.get_dvh_of_structs()
        phys_dvh = matrad_env.rescale_dvh(phys_dvh)
        phys_struct_means = np.array(matrad_env.get_structure_means())

        phys_dict['dvh'].append(phys_dvh)
        phys_dict['ci_70'].append(phys_ci_70)
        phys_dict['hi_52'].append(phys_hi_52)
        phys_dict['hi_56'].append(phys_hi_56)
        phys_dict['hi_70'].append(phys_hi_70)
        phys_dict['r50'].append(phys_r50_val)
        phys_dict['r90'].append(phys_r90_val)
        phys_dict['structure_means'].append(phys_struct_means)

        '''
        Meta opt
        '''

        meta_opt_info_dict = np.load(
            os.path.join(meta_optimization_results_dir, 'results_for_case_num_' + str(sample_case_num) + '.npy'),
            allow_pickle='TRUE').item()
        meta_optimized_weights = meta_opt_info_dict['final_projected_simplex'][0]


        vmat_traj_path = os.path.join(trajectory_directory,
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
                                  acceptable_iter=800,
                                  acceptable_constr_viol_tol=0.0001,
                                  ipopt_max_iters=800,
                                  vmat=True,
                                  additional_vmat_couch_angles = vmat_control_points_couch,
                                  additional_vmat_gantry_angles = vmat_control_points_gantry,
                                  additional_fmo_angles = vmat_fmo_gantry_angles)

        else:
            matrad_env = MatRadMetaOpt()
            matrad_env.reset_pops(ctDir,
                                  rtStDir,
                                  high_res=high_res,
                                  acceptable_iter=800,
                                  acceptable_constr_viol_tol=0.0001,
                                  ipopt_max_iters=800,
                                  vmat=True)


        matrad_env.wo_projection_no_score(meta_optimized_weights)
        matrad_env.run_seq_and_dao()
        ci_70,hi_52,hi_56,hi_70 = matrad_env.get_ci_and_hi()
        r50_val, r90_val = matrad_env.get_r50_and_r90()
        dvh = matrad_env.get_dvh_of_structs()
        dvh = matrad_env.rescale_dvh(dvh)
        struct_means = np.array(matrad_env.get_structure_means())
        pops_dict['dvh'].append(dvh)
        pops_dict['ci_70'].append(ci_70)
        pops_dict['hi_52'].append(hi_52)
        pops_dict['hi_56'].append(hi_56)
        pops_dict['hi_70'].append(hi_70)
        pops_dict['r50'].append(r50_val)
        pops_dict['r90'].append(r90_val)
        pops_dict['structure_means'].append(struct_means)
        pops_dict['point'].append(meta_optimized_weights)

        plt.figure()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, pops_dict['dvh'][-1].shape[0])))
        dvh_figure = plt.figure(figure_count)
        figure_count += 1
        for j in range(pops_dict['dvh'][-1].shape[0]):
            c = next(color)
            plt.plot(phys_dvh[j, :, 0], phys_dvh[j, :, 1], '--', c=c)
            plt.plot(pops_dict['dvh'][-1][j, :, 0], pops_dict['dvh'][-1][j, :, 1], '-', c=c)

        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')
        plt.xlim(-0.025, 2.05)
        plt.ylim(-2.5, 102.5)
        plt.title('DVH #' + str(sample_case_num))
        plt.draw()
        plt.savefig(os.path.join(output_directory, 'dvh_num_' + str(sample_case_num) + '.png'))

        results_dict['physician'] = copy.deepcopy(phys_dict)
        results_dict['pops'] = copy.deepcopy(pops_dict)
        # results_dict['initial'] = init_dict
        np.save(os.path.join(output_directory, 'comparison_dict.npy'), results_dict)

def find_nearest_dose_given_volume_point(volume_array, vol, dose_array):
        volume_array = np.asarray(volume_array)
        idx = (np.abs(volume_array - vol)).argmin()
        return dose_array[idx]


if __name__ == "__main__":

    '''
    Run meta-optimization framework
    '''
    data_dir = './head_and_neck_data/usable/'
    meta_optimization_results_dir = './meta_optimization_results/'
    bao_results_dir = './coplanar_bao_results/'


    '''get meta-optimized weights'''
    get_meta_optimized_weights(data_parent_dir = data_dir,
                               output_directory = meta_optimization_results_dir)



    # '''run 2 arc vmat'''
    # load_vmat_data_and_make_plots(meta_optimization_results_dir = meta_optimization_results_dir,
    #                               trajectory_directory = '',
    #                               data_parent_dir = data_dir,
    #                               output_directory = 'dvhs_and_stats/two_arc_coplanar_vmat',
    #                               high_res=True)
    # '''run imrt'''
    #
    # load_data_and_make_plots(meta_optimization_results_dir = meta_optimization_results_dir,
    #                          trajectory_directory = '',
    #                          data_parent_dir = data_dir,
    #                          output_directory = 'dvhs_and_stats/equispaced_coplanar')
    #
    #
    # ''''''


