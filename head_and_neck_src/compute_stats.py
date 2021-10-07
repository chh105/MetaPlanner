import numpy as np
import pandas as pd
import os
from matRad_wo_hn_env import *
import glob
from scipy.stats import wilcoxon
import statsmodels.api as sm
import matplotlib.pyplot as plt

def visualize_dose_distribution(sample_case_num = None,
                                data_parent_dir = './clean_prostate_data/isodose/usable/',
                                results_dir = '',
                                weights = None,
                                high_res = False,
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
                                  acceptable_iter=800,
                                  acceptable_constr_viol_tol=0.0001,
                                  ipopt_max_iters=800,
                                  vmat=True,
                                  additional_vmat_couch_angles=vmat_control_points_couch,
                                  additional_vmat_gantry_angles=vmat_control_points_gantry,
                                  additional_fmo_angles=vmat_fmo_gantry_angles)

        else:
            matrad_env.reset_pops(ctDir,
                                  rtStDir,
                                  high_res=high_res,
                                  acceptable_iter=800,
                                  acceptable_constr_viol_tol=0.0001,
                                  ipopt_max_iters=800,
                                  vmat=True)

        matrad_env.wo_projection_no_score(meta_optimized_weights)
        matrad_env.run_seq_and_dao()
    else:
        matrad_env.reset_pops(ctDir,
                              rtStDir,
                              couch_angle_list = couch_angle_list,
                              gantry_angle_list = gantry_angle_list,
                              high_res=high_res,
                              acceptable_iter=200,
                              acceptable_constr_viol_tol=0.0001,
                              ipopt_max_iters=200,)
        matrad_env.wo_projection_no_score(meta_optimized_weights)


    matrad_env.get_ci_and_hi()
    matrad_env.get_r50_and_r90()
    matrad_env.render(True)
    x = input('Finished?')

    return matrad_env

def visualize_manual_dose_distribution(sample_case_num = None,
                                    data_parent_dir = './clean_prostate_data/isodose/usable/'):

    case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir,'*')) if os.path.isdir(i)])
    num_cases = len(case_list)
    if sample_case_num is None:
        np.random.seed(0)
        sample_case_num = int(np.random.uniform(0,num_cases))

    ctDir = os.path.join(case_list[sample_case_num],'CT')
    rtStDir = os.path.join(case_list[sample_case_num],'RTst')
    rtDoseDir = os.path.join(case_list[sample_case_num],'RTDOSE')
    rtPlanDir = os.path.join(case_list[sample_case_num],'RTPLAN')
    
    matrad_env = MatRadMetaOpt()
    phys_ci_70, phys_hi_70, phys_hi_52, phys_hi_56, phys_r50_val, phys_r90_val = matrad_env.get_ptv_stats_from_treatment_plan(
        ctDir, rtStDir, rtDoseDir)
    matrad_env.get_ci_and_hi()
    matrad_env.get_r50_and_r90()
    matrad_env.render(True)
    x = input('Finished?')

    return matrad_env

def find_nearest_dose_given_volume_point(volume_array, vol, dose_array):
        volume_array = np.asarray(volume_array)
        idx = (np.abs(volume_array - vol)).argmin()
        return dose_array[idx]

def load_data_and_make_tables(method_a_path = './results_all_algos_pops_paper_edition_11_8_2020/pops_apops_deep_pops_and_phys_comparison.npy',
                             method_b_path = './hi_10_results/pops_coplanar_imrt_plots_and_tables/coplanar_pops_and_phys_comparison.npy',
                             method_a_name = 'pops',
                             method_b_name = 'pops'):

    ''''''
    import pandas as pd
    control_point_list = [2,20,40,60,80,98]
    num_oars = 12
    num_fractions = 35
    normalized_dose = 70
    ptv_index = 2
    data_dict = {}

    method_a_results_dict = np.load(method_a_path,allow_pickle = True).item()
    method_b_results_dict = np.load(method_b_path,allow_pickle = True).item()
    method_a_dvhs = method_a_results_dict[method_a_name]['dvh']
    method_b_dvhs = method_b_results_dict[method_b_name]['dvh']
    method_a_struct_means = num_fractions*np.array(method_a_results_dict[method_a_name]['structure_means'])
    method_b_struct_means = num_fractions*np.array(method_b_results_dict[method_b_name]['structure_means'])

    method_a_ci_70 = method_a_results_dict[method_a_name]['ci_70']
    method_a_hi_52 = np.array(method_a_results_dict[method_a_name]['hi_52'])/100
    method_a_hi_56 = np.array(method_a_results_dict[method_a_name]['hi_56'])/100
    method_a_hi_70 = np.array(method_a_results_dict[method_a_name]['hi_70'])/100
    method_a_r90 = method_a_results_dict[method_a_name]['r90']
    method_a_r50 = np.array(method_a_results_dict[method_a_name]['r50'])

    method_b_ci_70 = method_b_results_dict[method_b_name]['ci_70']
    method_b_hi_52 = np.array(method_b_results_dict[method_b_name]['hi_52'])/100
    method_b_hi_56 = np.array(method_b_results_dict[method_b_name]['hi_56'])/100
    method_b_hi_70 = np.array(method_b_results_dict[method_b_name]['hi_70'])/100
    method_b_r90 = method_b_results_dict[method_b_name]['r90']
    method_b_r50 = np.array(method_b_results_dict[method_b_name]['r50'])

    data_dict['method_a'] = np.zeros((len(method_a_dvhs),num_oars,len(control_point_list)))
    data_dict['method_b'] = np.zeros((len(method_b_dvhs),num_oars,len(control_point_list)))

    for case_num in range(len(method_a_dvhs)):
        current_dvh = method_a_dvhs[case_num]
        unnorm_dose = num_fractions*find_nearest_dose_given_volume_point(current_dvh[ptv_index,:,1],
                                                              95,
                                                              current_dvh[ptv_index,:,0])
        rescale_factor = normalized_dose/unnorm_dose

        for oar in range(current_dvh.shape[0]):
            for i,control_point in enumerate(control_point_list):
                data_dict['method_a'][case_num,oar,i] = find_nearest_dose_given_volume_point(current_dvh[oar,:,1],
                                                                                              control_point,
                                                                                              current_dvh[oar,:,0])
        data_dict['method_a'][case_num,:,:] *= rescale_factor*num_fractions
        method_a_struct_means[case_num] *= rescale_factor

    for case_num in range(len(method_b_dvhs)):
        current_dvh = method_b_dvhs[case_num]
        unnorm_dose = num_fractions*find_nearest_dose_given_volume_point(current_dvh[ptv_index,:,1],
                                                              95,
                                                              current_dvh[ptv_index,:,0])
        rescale_factor = normalized_dose/unnorm_dose

        for oar in range(current_dvh.shape[0]):
            for i,control_point in enumerate(control_point_list):
                data_dict['method_b'][case_num,oar,i] = find_nearest_dose_given_volume_point(current_dvh[oar,:,1],
                                                                                              control_point,
                                                                                              current_dvh[oar,:,0])
        data_dict['method_b'][case_num,:,:] *= rescale_factor*num_fractions
        method_b_struct_means[case_num] *= rescale_factor

    data_dict['method_a'] = data_dict['method_a']
    data_dict['method_b'] = data_dict['method_b']   




    # df_phys_mean = pd.DataFrame(np.mean(data_dict['method_a'],axis = 0))
    # df_phys_std = pd.DataFrame(np.std(data_dict['method_a'],axis = 0))
    # df_method_b_mean = pd.DataFrame(np.mean(data_dict['method_b'],axis = 0))
    # df_method_b_std = pd.DataFrame(np.std(data_dict['method_b'],axis = 0))
    # out_path = os.path.join(output_directory,'method_a_data_')
    # df_phys_mean.to_csv(out_path+'mu.csv',index=False)
    # df_phys_std.to_csv(out_path+'sigma.csv',index=False)
    # out_path = os.path.join(output_directory,'method_b_data_')
    # df_method_b_mean.to_csv(out_path+'mu.csv',index=False)
    # df_method_b_std.to_csv(out_path+'sigma.csv',index=False)

    # df_phys_struct_means_mean = pd.DataFrame(np.mean(method_a_struct_means,axis = 0))
    # df_phys_struct_means_std = pd.DataFrame(np.std(method_a_struct_means,axis = 0))
    # df_method_b_struct_means_mean = pd.DataFrame(np.mean(method_b_struct_means,axis = 0))
    # df_method_b_struct_means_std = pd.DataFrame(np.std(method_b_struct_means,axis = 0))
    # out_path = os.path.join(output_directory,'method_a_struct_means_')
    # df_phys_struct_means_mean.to_csv(out_path+'mu.csv',index=False)
    # df_phys_struct_means_std.to_csv(out_path+'sigma.csv',index=False)
    # out_path = os.path.join(output_directory,'method_b_struct_means_')
    # df_method_b_struct_means_mean.to_csv(out_path+'mu.csv',index=False)
    # df_method_b_struct_means_std.to_csv(out_path+'sigma.csv',index=False)
    

    # mean_relative_diffs = (method_b_struct_means - method_a_struct_means)\
    #                 /((method_b_struct_means + method_a_struct_means)/2) 

    # mean_relative_diffs = np.squeeze(mean_relative_diffs) # shape: [patients,oar,cp]

    # wilcoxon_test = np.zeros((num_oars,2))
    # for oar in range(num_oars):
        
    #     w, p = wilcoxon(mean_relative_diffs[:,oar])
    #     wilcoxon_test[oar,0] = w
    #     wilcoxon_test[oar,1] = p

    # print('Wilcoxon test (p) for method_b and method_a:\n',wilcoxon_test[:,1])
    # print('Null hypothesis rejection method_b and method_a:\n',wilcoxon_test[:,0])

    # df_wilcoxon = pd.DataFrame(wilcoxon_test)
    # out_path = os.path.join(output_directory,'wilcoxon_test.csv')
    # df_wilcoxon.to_csv(out_path,index=False)

    num_comparisons = num_oars*len(control_point_list)-2
    wilcoxon_test = np.zeros((num_oars,len(control_point_list),2))
    for oar in range(num_oars):
        for cp in range(len(control_point_list)):
            d = np.array(data_dict['method_b'][:,oar,cp])-np.array(data_dict['method_a'][:,oar,cp])
            if np.max(np.abs(d))==0:
                w = np.nan
                p = np.nan
            else:
                w, p = wilcoxon(d)
            wilcoxon_test[oar,cp,0] = w
            wilcoxon_test[oar,cp,1] = p

    method_a_struct_means = np.squeeze(method_a_struct_means)
    method_b_struct_means = np.squeeze(method_b_struct_means)
    p_struct_means = np.zeros(method_a_struct_means.shape[1])
    w_struct_means = np.zeros(method_a_struct_means.shape[1])
    for struct in range(method_a_struct_means.shape[1]):
        d_struct_means = method_b_struct_means[:,struct]-method_a_struct_means[:,struct]
        if np.max(np.abs(d_struct_means))==0:
            w_struct_means[struct] = np.nan
            p_struct_means[struct] = np.nan
        else:
            w_struct_means[struct], p_struct_means[struct] = wilcoxon(d_struct_means)

    np.set_printoptions(precision = 5,suppress = True)
    print('"Structure means" P-val:',p_struct_means)

    d_ci_70 = np.round(np.array(method_b_ci_70),decimals=3)-np.round(np.array(method_a_ci_70),decimals=3)
    d_hi_52 = np.round(np.array(method_b_hi_52),decimals=3)-np.round(np.array(method_a_hi_52),decimals=3)
    d_hi_56 = np.round(np.array(method_b_hi_56),decimals=3)-np.round(np.array(method_a_hi_56),decimals=3)
    d_hi_70 = np.round(np.array(method_b_hi_70),decimals=3)-np.round(np.array(method_a_hi_70),decimals=3)
    d_r90 = np.round(np.array(method_b_r90),decimals=3)-np.round(np.array(method_a_r90),decimals=3)
    d_r50 = np.round(np.array(method_b_r50),decimals=3)-np.round(np.array(method_a_r50),decimals=3)
    if np.max(np.abs(d_ci_70))==0:
        w_ci_70 = np.nan
        p_ci_70 = np.nan
    else:
        w_ci_70, p_ci_70 = wilcoxon(d_ci_70)

    if np.max(np.abs(d_hi_52))==0:
        w_hi_52 = np.nan
        p_hi_52 = np.nan
    else:
        w_hi_52, p_hi_52 = wilcoxon(d_hi_52)

    if np.max(np.abs(d_hi_56))==0:
        w_hi_56 = np.nan
        p_hi_56 = np.nan
    else:
        w_hi_56, p_hi_56 = wilcoxon(d_hi_56)

    if np.max(np.abs(d_hi_70))==0:
        w_hi_70 = np.nan
        p_hi_70 = np.nan
    else:
        w_hi_70, p_hi_70 = wilcoxon(d_hi_70)

    if np.max(np.abs(d_r90))==0:
        w_r90 = np.nan
        p_r90 = np.nan
    else:
        w_r90, p_r90 = wilcoxon(d_r90)

    if np.max(np.abs(d_r50))==0:
        w_r50 = np.nan
        p_r50 = np.nan
    else:
        w_r50, p_r50 = wilcoxon(d_r50)

    print('Oar P-vals:',wilcoxon_test[:,:,1])
    print('CI (70) P-val:',p_ci_70)
    print('HI (52,56,70) P-val:',p_hi_52,p_hi_56,p_hi_70)
    print('R90 P-val:',p_r90)
    print('R50 P-val:',p_r50)
    # p_vals = list(np.ravel(wilcoxon_test[:,:,1]))[:-2] + [p_ci] + [p_hi]
    # reject, corrected_p_vals, alpha_s, alpha_b = sm.stats.multipletests(p_vals,method = 'fdr_bh')
    #
    # oar_p_vals = np.reshape(np.concatenate([corrected_p_vals[:-2],[np.nan]*2]),
    #                        (num_oars,len(control_point_list)))
    # oar_reject = np.reshape(np.concatenate([reject[:-2],[np.nan]*2]),
    #                        (num_oars,len(control_point_list)))
    # ci_p_val = corrected_p_vals[-2]
    # ci_reject = reject[-2]
    # hi_p_val = corrected_p_vals[-1]
    # hi_reject = reject[-1]


    # np.set_printoptions(precision = 2,suppress = True)
    # print('method_a "structure means" means:',np.mean(method_a_struct_means,axis = 0))
    # print('method_a "structure means" stdvs:',np.std(method_a_struct_means,axis = 0))
    # print('method_b "structure means" means:',np.mean(method_b_struct_means,axis = 0))
    # print('method_b "structure means" stdvs:',np.std(method_b_struct_means,axis = 0))

    # print('method_a means:',np.mean(data_dict['method_a'],axis = 0))
    # print('method_a stdvs:',np.std(data_dict['method_a'],axis = 0))
    # print('method_b means:',np.mean(data_dict['method_b'],axis = 0))
    # print('method_b stdvs:',np.std(data_dict['method_b'],axis = 0))


    # print('method_a mean ci:',np.mean(method_a_ci,axis = 0))
    # print('method_a std ci:',np.std(method_a_ci,axis = 0))
    # print('method_a mean hi:',np.mean(method_a_hi,axis = 0))
    # print('method_a std hi:',np.std(method_a_hi,axis = 0))
    # print('method_b mean ci:',np.mean(method_b_ci,axis = 0))
    # print('method_b std ci:',np.std(method_b_ci,axis = 0))
    # print('method_b mean hi:',np.mean(method_b_hi,axis = 0))
    # print('method_b std hi:',np.std(method_b_hi,axis = 0))
    # print('Wilcoxon test (p) for method_b and method_a:\n',oar_p_vals)
    # print('Null hypothesis rejection method_b and method_a:\n',oar_reject)
    # print('CI Wilcoxon test (p) for method_b and method_a:',ci_p_val)
    # print('Null hypothesis rejection CI for method_b and method_a:',ci_reject)
    # print('HI Wilcoxon test (p) for method_b and method_a:',hi_p_val)
    # print('Null hypothesis rejection HI for method_b and method_a:',hi_reject)

    # print('Alpha Sidak:',alpha_s)
    # print('Alpha Bonferroni:',alpha_b)


def compare_scores(method_a_dir = './hi_5_multiobj_results/coplanar_results_9_beams/',
                 method_b_dir = './hi_5_multiobj_results/noncoplanar_config_1_results_9_beams',
                 method_a_name = 'pops',
                 method_b_name = 'pops'):

    method_a_case_list = sorted([i for i in glob.glob(os.path.join(method_a_dir,'results*.npy'))])
    num_cases_a = len(method_a_case_list)
    method_b_case_list = sorted([i for i in glob.glob(os.path.join(method_b_dir,'results*.npy'))])
    num_cases_b = len(method_b_case_list)
    assert num_cases_a==num_cases_b

    a_scores = []
    b_scores = []

    for sample_case_num in range(num_cases_a):
        a_dict = np.load(os.path.join(method_a_dir,'results_for_case_num_'+str(sample_case_num)+'.npy'),allow_pickle=True).item()
        b_dict = np.load(os.path.join(method_b_dir,'results_for_case_num_'+str(sample_case_num)+'.npy'),allow_pickle=True).item()
        a_scores.append(a_dict['final_gem_scores'][0])
        b_scores.append(b_dict['final_gem_scores'][0])


    # wilcoxon test
    d_scores = np.array(b_scores)-np.array(a_scores)
    if np.max(np.abs(d_scores))==0:
        w = np.nan
        p = np.nan
    else:
        w, p = wilcoxon(d_scores)


    print('Score differences (b-a):',d_scores)
    print('Method a scores: '+str(np.mean(a_scores))+' +/- '+str(np.std(a_scores)))
    print('Method b scores: '+str(np.mean(b_scores))+' +/- '+str(np.std(b_scores)))
    print('Wilcoxon p values:',p)
    print('Wilcoxon w values:',w)

def compare_dvhs_and_stats(method_a_path = './hi_5_multiobj_results/coplanar_plots_and_tables_9_beams/comparison_dict.npy',
                         method_b_path = './hi_5_multiobj_results/noncoplanar_config_1_plots_and_tables_9_beams/comparison_dict.npy',
                         method_a_name = 'pops',
                         method_b_name = 'pops',
                         output_directory = './hi_5_multiobj_results/comparisons/nc_9_config_1_vs_c_9'):

    ''''''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    import pandas as pd
    control_point_list = [2,20,40,60,80,98]
    num_oars = 12
    num_fractions = 35
    normalized_dose = 70
    target_homogeneity = 1.05
    ptv_index = 2
    data_dict = {}

    method_a_results_dict = np.load(method_a_path,allow_pickle = True).item()
    method_b_results_dict = np.load(method_b_path,allow_pickle = True).item()
    method_a_structure_means = num_fractions*np.squeeze(method_a_results_dict[method_a_name]['structure_means'])
    method_b_structure_means = num_fractions*np.squeeze(method_b_results_dict[method_b_name]['structure_means'])
    method_a_dvhs = method_a_results_dict[method_a_name]['dvh']
    method_b_dvhs = method_b_results_dict[method_b_name]['dvh']
    scaled_dvhs_a = method_a_dvhs
    scaled_dvhs_b = method_b_dvhs

    method_a_ci_70 = method_a_results_dict[method_a_name]['ci_70']
    method_a_hi_52 = np.array(method_a_results_dict[method_a_name]['hi_52']) / 100
    method_a_hi_56 = np.array(method_a_results_dict[method_a_name]['hi_56']) / 100
    method_a_hi_70 = np.array(method_a_results_dict[method_a_name]['hi_70']) / 100
    method_a_r90 = method_a_results_dict[method_a_name]['r90']
    method_a_r50 = np.array(method_a_results_dict[method_a_name]['r50'])

    method_b_ci_70 = method_b_results_dict[method_b_name]['ci_70']
    method_b_hi_52 = np.array(method_b_results_dict[method_b_name]['hi_52']) / 100
    method_b_hi_56 = np.array(method_b_results_dict[method_b_name]['hi_56']) / 100
    method_b_hi_70 = np.array(method_b_results_dict[method_b_name]['hi_70']) / 100
    method_b_r90 = method_b_results_dict[method_b_name]['r90']
    method_b_r50 = np.array(method_b_results_dict[method_b_name]['r50'])

    data_dict['method_a'] = np.zeros((len(method_a_dvhs),num_oars,len(control_point_list)))
    data_dict['method_b'] = np.zeros((len(method_b_dvhs),num_oars,len(control_point_list)))

    for case_num in range(len(method_a_dvhs)):
        current_dvh = method_a_dvhs[case_num]
        unnorm_dose = num_fractions*find_nearest_dose_given_volume_point(current_dvh[ptv_index,:,1],
                                                              95,
                                                              current_dvh[ptv_index,:,0])
        rescale_factor = normalized_dose/unnorm_dose

        for oar in range(current_dvh.shape[0]):
            for i,control_point in enumerate(control_point_list):
                data_dict['method_a'][case_num,oar,i] = find_nearest_dose_given_volume_point(current_dvh[oar,:,1],
                                                                                              control_point,
                                                                                              current_dvh[oar,:,0])
        data_dict['method_a'][case_num,:,:] *= rescale_factor*num_fractions
        scaled_dvhs_a[case_num][:,:,0] *= rescale_factor*num_fractions
        method_a_structure_means[case_num] *= rescale_factor

    for case_num in range(len(method_b_dvhs)):
        current_dvh = method_b_dvhs[case_num]
        unnorm_dose = num_fractions*find_nearest_dose_given_volume_point(current_dvh[ptv_index,:,1],
                                                              95,
                                                              current_dvh[ptv_index,:,0])
        rescale_factor = normalized_dose/unnorm_dose

        for oar in range(current_dvh.shape[0]):
            for i,control_point in enumerate(control_point_list):
                data_dict['method_b'][case_num,oar,i] = find_nearest_dose_given_volume_point(current_dvh[oar,:,1],
                                                                                              control_point,
                                                                                              current_dvh[oar,:,0])
        data_dict['method_b'][case_num,:,:] *= rescale_factor*num_fractions
        scaled_dvhs_b[case_num][:,:,0] *= rescale_factor*num_fractions
        method_b_structure_means[case_num] *= rescale_factor


    figure_count = 0
    for case_num in range(len(scaled_dvhs_a)):
        num_structs = scaled_dvhs_a[case_num].shape[0]

        plt.close('all')
        plt.figure()
        color=iter(plt.cm.rainbow(np.linspace(0,1,num_structs)))
        dvh_figure = plt.figure(figure_count)
        figure_count += 1
        for s in range(num_structs):
            c=next(color)
            plt.plot(scaled_dvhs_a[case_num][s,:,0],scaled_dvhs_a[case_num][s,:,1],'--',c = c)
            plt.plot(scaled_dvhs_b[case_num][s,:,0],scaled_dvhs_b[case_num][s,:,1],'-',c = c)


        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')
        plt.xlim(-0.025*num_fractions,np.round(target_homogeneity*normalized_dose))
        plt.ylim(-2.5,102.5)
        plt.title('DVH #'+str(case_num))
        plt.draw()
        plt.savefig(os.path.join(output_directory,'dvh_num_'+str(case_num)+'.png'))

    figure_count = 0
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_structs)))

    import seaborn as sns
    plot_data_a = {}
    plot_data_b = {}

    a_mat = np.zeros((len(scaled_dvhs_a),)+scaled_dvhs_a[0].shape)
    b_mat = np.zeros((len(scaled_dvhs_b),)+scaled_dvhs_b[0].shape)
    for s in range(num_structs):
        for case_num in range(len(scaled_dvhs_a)):
            a_mat[case_num,s,:,0] = scaled_dvhs_a[case_num][s,:,0]
            a_mat[case_num,s,:,1] = scaled_dvhs_a[case_num][s,:,1]
    for s in range(num_structs):
        for case_num in range(len(scaled_dvhs_b)):
            b_mat[case_num,s,:,0] = scaled_dvhs_b[case_num][s,:,0]
            b_mat[case_num,s,:,1] = scaled_dvhs_b[case_num][s,:,1]

    plot_data_a['mean'] = np.mean(a_mat,axis=0).astype('float64')
    plot_data_a['std'] = np.std(a_mat,axis=0).astype('float64')
    plot_data_b['mean'] = np.mean(b_mat,axis=0).astype('float64')
    plot_data_b['std'] = np.std(b_mat,axis=0).astype('float64')

    plt.close('all')
    plt.figure()
    dvh_figure = plt.figure(figure_count)
    figure_count += 1
    color=iter(plt.cm.rainbow(np.linspace(0,1,num_structs)))
    for s in range(num_structs):
        c=next(color)
        plt.plot(plot_data_a['mean'][s,:,0],plot_data_a['mean'][s,:,1],'-',c = c)
        plt.fill_between(plot_data_a['mean'][s, :, 0],
                          np.maximum(plot_data_a['mean'][s, :, 1] - plot_data_a['std'][s, :, 1],0),
                          np.minimum(plot_data_a['mean'][s, :, 1] + plot_data_a['std'][s, :, 1],100),
                          color=c, alpha=0.2)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.xlim(-0.025 * num_fractions, np.round(target_homogeneity * normalized_dose))
    plt.ylim(-2.5, 102.5)
    plt.title('DVH')
    plt.savefig(os.path.join(output_directory, 'dvh_a.png'))


    plt.close('all')
    plt.figure()
    dvh_figure = plt.figure(figure_count)
    figure_count += 1
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_structs)))
    for s in range(num_structs):
        c = next(color)
        plt.plot(plot_data_b['mean'][s, :, 0], plot_data_b['mean'][s, :, 1], '-', c=c)
        plt.fill_between(plot_data_b['mean'][s, :, 0],
                          np.maximum(plot_data_b['mean'][s, :, 1] - plot_data_b['std'][s, :, 1],0),
                          np.minimum(plot_data_b['mean'][s, :, 1] + plot_data_b['std'][s, :, 1],100),
                          color=c,alpha=0.2)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.xlim(-0.025 * num_fractions, np.round(target_homogeneity * normalized_dose))
    plt.ylim(-2.5, 102.5)
    plt.title('DVH')
    plt.savefig(os.path.join(output_directory, 'dvh_b.png'))



    num_decimals = 1
    np.set_printoptions(precision = num_decimals,suppress = True)

    print('method a mean ci (70):',np.mean(method_a_ci_70,axis = 0))
    print('method a std ci (70):',np.std(method_a_ci_70,axis = 0))
    print('method a mean hi (52,56,70):',np.mean(method_a_hi_52,axis = 0),
                                         np.mean(method_a_hi_56,axis = 0),
                                         np.mean(method_a_hi_70,axis = 0))
    print('method a std hi (52,56,70):',np.std(method_a_hi_52,axis = 0),
                                         np.std(method_a_hi_56,axis = 0),
                                         np.std(method_a_hi_70,axis = 0))
    print('method_a mean r90:',np.mean(method_a_r90,axis = 0))
    print('method_a std r90:',np.std(method_a_r90,axis = 0))
    print('method_a mean r50:',np.mean(method_a_r50,axis = 0))
    print('method_a std r50:',np.std(method_a_r50,axis = 0))

    print('method b mean ci (70):', np.mean(method_b_ci_70, axis=0))
    print('method b std ci (70):', np.std(method_b_ci_70, axis=0))
    print('method b mean hi (52,56,70):', np.mean(method_b_hi_52, axis=0),
          np.mean(method_b_hi_56, axis=0),
          np.mean(method_b_hi_70, axis=0))
    print('method b std hi (52,56,70):', np.std(method_b_hi_52, axis=0),
          np.std(method_b_hi_56, axis=0),
          np.std(method_b_hi_70, axis=0))
    print('method_b mean r90:',np.mean(method_b_r90,axis = 0))
    print('method_b std r90:',np.std(method_b_r90,axis = 0))
    print('method_b mean r50:',np.mean(method_b_r50,axis = 0))
    print('method_b std r50:',np.std(method_b_r50,axis = 0))

    method_a_structure_mean_std = [[np.round(np.mean(method_a_structure_means,0),num_decimals)[i],\
                                    np.round(np.std(method_a_structure_means,0),num_decimals)[i]]\
                                    for i in range(len(np.round(np.mean(method_a_structure_means,0),num_decimals)))]
    print('method a structure mean doses: ')
    print('\t'.join('{} ({})'.format(x[0],x[1]) for x in method_a_structure_mean_std))

    method_b_structure_mean_std = [[np.round(np.mean(method_b_structure_means,0),num_decimals)[i],\
                                    np.round(np.std(method_b_structure_means,0),num_decimals)[i]]\
                                    for i in range(len(np.round(np.mean(method_b_structure_means,0),num_decimals)))]
    print('method b structure mean doses: ')
    print('\t'.join('{} ({})'.format(x[0],x[1]) for x in method_b_structure_mean_std))

    method_a_dvh_means = np.mean(data_dict['method_a'],axis = 0)
    method_a_dvh_stds = np.std(data_dict['method_a'],axis = 0)
    method_b_dvh_means = np.mean(data_dict['method_b'],axis = 0)
    method_b_dvh_stds = np.std(data_dict['method_b'],axis = 0)
    print('method a dvh control points:')
    for i in range(method_a_dvh_means.shape[0]):
        method_a_mean_std = [[np.round(method_a_dvh_means[i,j],num_decimals),np.round(method_a_dvh_stds[i,j],num_decimals)] for j in range(method_a_dvh_means.shape[1])]
        print('\t'.join('{} ({})'.format(x[0],x[1]) for x in method_a_mean_std))
    print('method b dvh control points:')
    for i in range(method_b_dvh_means.shape[0]):
        method_b_mean_std = [[np.round(method_b_dvh_means[i,j],num_decimals),np.round(method_b_dvh_stds[i,j],num_decimals)] for j in range(method_b_dvh_means.shape[1])]
        print('\t'.join('{} ({})'.format(x[0],x[1]) for x in method_b_mean_std))

if __name__ == "__main__":
    # visualize_dose_distribution(sample_case_num = 0,
    #                             data_parent_dir = './head_and_neck_data/usable/',
    #                             results_dir = './meta_optimization_results/',
    #                             vmat = False,
    #                             high_res=True)

    # visualize_dose_distribution(sample_case_num = 0,
    #                             data_parent_dir = './head_and_neck_data/usable/',
    #                             results_dir = './meta_optimization_results/',
    #                             vmat = True,
    #                             high_res=True)

    # visualize_dose_distribution(sample_case_num = 0,
    #                             data_parent_dir = './head_and_neck_data/usable/',
    #                             results_dir = './meta_optimization_results/',
    #                             vmat_traj_dir='./nonuniform_sampling_results/',
    #                             vmat = True)

    # visualize_dose_distribution(sample_case_num = 0,
    #                             data_parent_dir = './head_and_neck_data/usable/',
    #                             results_dir = './meta_optimization_results/',
    #                             vmat_traj_dir='./two_arc_vmat_results/',
    #                             vmat = True)

    visualize_manual_dose_distribution(sample_case_num = 0,
                                data_parent_dir = './head_and_neck_data/usable/')



    ''''''


    # compare_dvhs_and_stats(
    #     method_a_path='./dvhs_and_stats/coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/coplanar_vmat/comparison_dict.npy',
    #     method_a_name='physician',
    #     method_b_name='pops',
    #     output_directory='./dvhs_and_stats/coplanar_vmat/mo_vmat_vs_manual')
    # load_data_and_make_tables(
    #     method_a_path='./dvhs_and_stats/coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/coplanar_vmat/comparison_dict.npy',
    #     method_a_name='physician',
    #     method_b_name='pops')

    # compare_dvhs_and_stats(
    #     method_a_path='./dvhs_and_stats/equispaced_coplanar/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='pops',
    #     method_b_name='pops',
    #     output_directory='./dvhs_and_stats/two_arc_coplanar_vmat/mo_vmat_vs_mo_imrt')
    # load_data_and_make_tables(
    #     method_a_path='./dvhs_and_stats/equispaced_coplanar/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='pops',
    #     method_b_name='pops')

    # compare_dvhs_and_stats(
    #     method_a_path='./dvhs_and_stats/equispaced_coplanar/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/equispaced_coplanar/comparison_dict.npy',
    #     method_a_name='physician',
    #     method_b_name='pops',
    #     output_directory='./dvhs_and_stats/equispaced_coplanar/mo_imrt_vs_manual')
    # load_data_and_make_tables(
    #     method_a_path='./dvhs_and_stats/equispaced_coplanar/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/equispaced_coplanar/comparison_dict.npy',
    #     method_a_name='physician',
    #     method_b_name='pops')


    ''''''
    # compare_dvhs_and_stats(
    #     method_a_path='./dvhs_and_stats/coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/nu_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='pops',
    #     method_b_name='pops',
    #     output_directory='./dvhs_and_stats/nu_coplanar_vmat/mo_nu_vmat_vs_mo_vmat')
    # load_data_and_make_tables(
    #     method_a_path='./dvhs_and_stats/coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/nu_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='pops',
    #     method_b_name='pops')

    # compare_dvhs_and_stats(
    #     method_a_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='physician',
    #     method_b_name='pops',
    #     output_directory='./dvhs_and_stats/two_arc_coplanar_vmat/mo_two_arc_vmat_vs_manual')
    # load_data_and_make_tables(
    #     method_a_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='physician',
    #     method_b_name='pops')

    # compare_dvhs_and_stats(
    #     method_a_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/nu_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='pops',
    #     method_b_name='pops',
    #     output_directory='./dvhs_and_stats/nu_coplanar_vmat/mo_nu_vmat_vs_mo_two_arc_vmat')
    # load_data_and_make_tables(
    #     method_a_path='./dvhs_and_stats/two_arc_coplanar_vmat/comparison_dict.npy',
    #     method_b_path='./dvhs_and_stats/nu_coplanar_vmat/comparison_dict.npy',
    #     method_a_name='pops',
    #     method_b_name='pops')