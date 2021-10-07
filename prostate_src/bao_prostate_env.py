from matRad_wo_prostate_env import *
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import copy
from scipy import interpolate
import time
import pandas as pd
from scipy.stats import wilcoxon
import random

import ray



        
class BeamAngleOptimization:    
    def __init__(self,
                 meta_optimized_weights,
                 data_dir = '',
                 sample_case_num = 0, 
                 collision_angles_path = './collision_angle_data/collision_angles_data_prostate.csv',
                 visualize = True,
                 results_dir = './trajectory_search_results/',
                 num_imrt_beams = 5,
                 starting_couch_angles = [],
                 starting_gantry_angles = [],
                 num_threads = 4):
        self.data_dir = data_dir
        self.collision_angles_path = collision_angles_path
        self.max_change_in_gantry = 12 # 12
        self.max_change_in_couch = 2
        self.visualize = visualize
        self.starting_couch_angle = 90
        self.ending_couch_angle = -90
        self.starting_gantry_angle = 180
        self.ending_gantry_angle = -180
        self.num_imrt_beams = num_imrt_beams
        self.results_dir = results_dir
        self.starting_couch_angles = starting_couch_angles
        self.starting_gantry_angles = starting_gantry_angles
        self.num_threads = num_threads


        assert len(self.starting_couch_angles) == len(self.starting_gantry_angles)
        # assert self.num_imrt_beams == len(self.starting_couch_angles)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        case_list = sorted([i for i in glob.glob(os.path.join(data_dir,'*')) if os.path.isdir(i)])
        num_cases = len(case_list)
        if sample_case_num is None:
            sample_case_num = int(np.random.uniform(0,num_cases))
            # sample_case_num = 21

        self.patient_dir = case_list[sample_case_num]
        self.sample_case_num = sample_case_num
        print('Patient Directory: ',self.patient_dir)
        self.ct_dir = os.path.join(case_list[sample_case_num],'CT')
        self.rtst_dir = os.path.join(case_list[sample_case_num],'RTst')
        self.meta_optimized_weights = meta_optimized_weights

        self.compute_feasible_angles()

    def compute_feasible_angles(self):
        self.load_collision_angles()
        feasible_couch_angle_list, feasible_gantry_angle_list = self.compute_collision_free_points()
        print('Number of feasible angles: ', len(feasible_couch_angle_list))
        assert len(feasible_couch_angle_list) == len(feasible_gantry_angle_list)

        self.trajectory = {}
        self.feasible_couch_angle_list = copy.deepcopy(feasible_couch_angle_list)
        self.feasible_gantry_angle_list = copy.deepcopy(feasible_gantry_angle_list)
        self.angle_indices = np.arange(len(feasible_couch_angle_list))



    def generate_cost_map(self):
        num_threads = self.num_threads
        assert num_threads > 1

        self.generate_cost_map_parallel(num_threads = num_threads,
                                        starting_couch_angles = self.starting_couch_angles,
                                        starting_gantry_angles = self.starting_gantry_angles)


    
    def generate_cost_map_parallel(self, num_threads, starting_couch_angles = [], starting_gantry_angles = []):
        self.worker_dict = {'workers': [],}

        temp_results = []
        for i in range(num_threads):
            remote_class = ray.remote(MatRadMetaOpt)
            time.sleep(5)
            worker = remote_class.remote()
            result = worker.reset_bao.remote(self.ct_dir,
                                             self.rtst_dir,
                                             self.meta_optimized_weights,
                                             high_res=False,
                                             acceptable_iter=50,
                                             acceptable_constr_viol_tol=0.01,
                                             ipopt_max_iters=50
                                             )
            self.worker_dict['workers'].append(worker)
            temp_results.append(result)

        for i in range(len(temp_results)):
            _ = ray.get(temp_results[i])




        # precompute dij
        print('Precomputing dose influence matrix on each worker...')

        temp_results = []
        for w, worker in enumerate(self.worker_dict['workers']):
            result = worker.precompute_dij.remote(self.feasible_couch_angle_list,
                                                  self.feasible_gantry_angle_list)
            temp_results.append(result)
        
        for i in range(len(temp_results)):
            precompute_completed = ray.get(temp_results[i])


        if self.visualize:

            sub_dir = os.path.join(self.results_dir,
                                   'heat_map_plots_for_case_'+str(self.sample_case_num))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)



        # starting_angle_ids = self.convert_angles_to_ids(couch_angle_list=starting_couch_angles,
        #                                                 gantry_angle_list=starting_gantry_angles)
        starting_angle_ids = []
        optimal_indices = copy.deepcopy(starting_angle_ids)
        optimal_couch_angles = copy.deepcopy(starting_couch_angles)
        optimal_gantry_angles = copy.deepcopy(starting_gantry_angles)
        optimal_obj_func_vals = []
        final_couch_angles = []
        final_gantry_angles = []
        final_angle_ids = []
        for i in range(self.num_imrt_beams):
            print('Iter #', i)
            # optimal_couch_angles = optimal_couch_angles[:0] + optimal_couch_angles[1:]
            # optimal_gantry_angles = optimal_gantry_angles[:0] + optimal_gantry_angles[1:]
            # optimal_indices = optimal_indices[:0] + optimal_indices[1:]
            heat_map = self.compute_heat_map_parallel(self.feasible_couch_angle_list, self.feasible_gantry_angle_list,
                                             optimal_couch_angles = optimal_couch_angles,
                                             optimal_gantry_angles = optimal_gantry_angles,
                                             optimal_indices = optimal_indices,
                                             num_threads = num_threads)

            optimal_indices = heat_map['updated_optimal_indices']
            optimal_couch_angles = heat_map['updated_optimal_couch_angles']
            optimal_gantry_angles = heat_map['updated_optimal_gantry_angles']
            optimal_obj_func_val = heat_map['optimal_obj_func_val']
            optimal_obj_func_vals.append(optimal_obj_func_val)
            print('Optimal angle indices: ', optimal_indices)
            print('Optimal couch angles: ', optimal_couch_angles)
            print('Optimal gantry angles: ', optimal_gantry_angles)
            print('Optimal obj function value: ', optimal_obj_func_val)
            final_angle_ids.append(optimal_indices[-1])
            final_couch_angles.append(optimal_couch_angles[-1])
            final_gantry_angles.append(optimal_gantry_angles[-1])

            heat_map_couch_angles = [self.feasible_couch_angle_list[x] for x in heat_map['selected_indices']]
            heat_map_gantry_angles = [self.feasible_gantry_angle_list[x] for x in heat_map['selected_indices']]
            self.trajectory['heat_map_iter_'+str(i)] = np.column_stack([heat_map_couch_angles,
                                                                        heat_map_gantry_angles,
                                                                        heat_map['obj_func_vals']])

            if self.visualize:
                heat_map_output_path =  os.path.join(sub_dir,
                                                     'heat_map_for_case_'+str(self.sample_case_num)\
                                                     +'beam_config_'+str(i)\
                                                     +'.png')

                self.plot_and_save_heat_maps(self.feasible_couch_angle_list,
                                             self.feasible_gantry_angle_list,
                                             heat_map['selected_indices'],
                                             heat_map['obj_func_vals'],
                                             heat_map_output_path)


        control_point_couch_angles = copy.deepcopy(final_couch_angles)
        control_point_gantry_angles = copy.deepcopy(final_gantry_angles)
        control_point_angle_ids = copy.deepcopy(final_angle_ids)
        print('Final couch angles:',control_point_couch_angles)
        print('Final gantry angles:',control_point_gantry_angles)

        if self.visualize:
            self.fig, self.axes2 = plt.subplots()
            self.axes2.scatter(control_point_couch_angles,control_point_gantry_angles,c = 'b')
            self.axes2.set_xlim(-90,90)
            self.axes2.set_ylim(-180,180)
            # plt.show()
            out_path = os.path.join(self.results_dir,'imrt_control_points_plot_for_case_'+str(self.sample_case_num)+'.png')
            plt.savefig(out_path)
            plt.close()

        self.trajectory['sampled_path_couch_angles'] = control_point_couch_angles
        self.trajectory['sampled_path_gantry_angles'] = control_point_gantry_angles
        self.trajectory['optimal_obj_func_vals'] = optimal_obj_func_vals

        # print(self.trajectory)

        for worker in self.worker_dict['workers']:
            ray.kill(worker)

    def plot_and_save_heat_maps(self, feasible_couch_angle_list, 
                                feasible_gantry_angle_list,
                                selected_indices,
                                heat_map_vals,
                                output_path):
        '''
        Parameters
        ----------
        feasible_couch_angles: list
            couch angles in the heat map that do not cause collisions between couch/gantry
        feasible_gantry_angles: list
            gantry angles in the heat map that do not cause collisions between couch/gantry
        selected_indices: list
            list of indexes used to decode beam angles
        heat_map_vals: list
            heat map objective function values for plotting
        output_path: string
            output path for saving figure
        '''
        from scipy.interpolate import griddata
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self.heat_map_fig, self.heat_map_axs = plt.subplots()
        divider = make_axes_locatable(self.heat_map_axs)
        cax = divider.append_axes('right',size='5%',pad=0.05)

        plot_couch_angles = [feasible_couch_angle_list[x] for x in selected_indices]
        plot_gantry_angles = [feasible_gantry_angle_list[x] for x in selected_indices]

        points = np.column_stack((plot_couch_angles,plot_gantry_angles))
        xx, yy = np.meshgrid(np.linspace(-90,90,1000), np.linspace(-180,180,1000))
        grid_0 = griddata(points,heat_map_vals,(xx,yy),method='nearest')
        mask_arr = np.zeros((1000,1000))

        for ix in range(xx.shape[0]):
            for iy in range(yy.shape[0]):
                is_feasible = self.check_if_angle_is_feasible(xx[ix,iy],yy[ix,iy])
                mask_arr[ix,iy] = is_feasible

        self.heat_map_im = self.heat_map_axs.imshow(np.ma.array(grid_0,mask = 1-mask_arr).filled(),
                                 extent = (-90,90,-180,180),
                                 origin='lower',
                                 vmax=np.mean(grid_0))
        self.heat_map_fig.colorbar(self.heat_map_im,cax=cax,orientation='vertical')
        plt.savefig(output_path)
        plt.close()


    def check_if_angle_is_feasible(self,couch,gantry):
        if couch >= 0 and gantry >= 0:
            quadrant = 1
            is_point_feasible = gantry <= self.f1(couch)
        elif couch < 0 and gantry >= 0:
            quadrant = 2
            is_point_feasible = gantry <= self.f2(couch)
        elif couch < 0 and gantry < 0:
            quadrant = 3
            is_point_feasible = gantry >= self.f3(couch)
        else:
            quadrant = 4
            is_point_feasible = gantry >= self.f4(couch)
        return is_point_feasible


    def compute_heat_map_parallel(self, feasible_couch_angles, feasible_gantry_angles, 
                         optimal_couch_angles = [], optimal_gantry_angles = [], optimal_indices = [],
                         num_threads = 2, max_count = 10):
        '''
        Parameters
        ----------
        feasible_couch_angles: list
            couch angles in the heat map that do not cause collisions between couch/gantry
        feasible_gantry_angles: list
            gantry angles in the heat map that do not cause collisions between couch/gantry
        optimal_couch_angles: list
            running list of selected control points (couch coordinate)
        optimal_gantry_angles: list
            running list of selected control points (gantry coordinate)
        optimal_indices: list
            running list of selected control point indices
        num_threads: int
            refers to the number of parallel threads being run

        Returns
        -------
        heat_map: dict
            contains information for the control points and objective function values
        '''
        heat_map = {'feasible_couch_angles':feasible_couch_angles,
                    'feasible_gantry_angles':feasible_gantry_angles,
                    'optimal_indices':optimal_indices,
                    'optimal_couch_angles':optimal_couch_angles,
                    'optimal_gantry_angles':optimal_gantry_angles,
                    'obj_func_vals': [],
                    'selected_indices': [],
                    'debug_dict': {}}

        # split up work by thread
        index_spacing = len(feasible_couch_angles)//num_threads
        remainder = len(feasible_couch_angles) % num_threads
        indices = np.arange(len(feasible_couch_angles))
        indices_split_by_thread = [indices[x:x + index_spacing] for x in range(0,len(indices),index_spacing)]
        if len(indices_split_by_thread) > num_threads:
            remaining = np.concatenate(indices_split_by_thread[num_threads:])
            indices_split_by_thread = indices_split_by_thread[:num_threads]
            for i,rem in enumerate(list(remaining)):
                indices_split_by_thread[i] = np.concatenate([indices_split_by_thread[i], np.array([rem])])


        # results = []
        # for w, worker in enumerate(self.worker_dict['workers']):
        #     angle_indices = list(indices_split_by_thread[w])

        #     for i in angle_indices:
        #         current_indices = optimal_indices + [i]
        #         current_couch_angles = optimal_couch_angles + [feasible_couch_angles[i]]
        #         current_gantry_angles = optimal_gantry_angles + [feasible_gantry_angles[i]]

        #         result = worker.get_obj_function_value.remote(couch_angle = current_couch_angles, 
        #                                                       gantry_angle = current_gantry_angles,
        #                                                       current_indices = current_indices)
        #         results.append(result)

        # for i in range(len(results)):
        #     obj_func_val = ray.get(results[i])
        #     heat_map['obj_func_vals'].append(obj_func_val)
        
        split_lengths = []
        for w, worker in enumerate(self.worker_dict['workers']):
            split_lengths.append(len(indices_split_by_thread[w]))

        for i in range(np.max(split_lengths)):
            worker_results = []
            selected_indices = []

            for w, worker in enumerate(self.worker_dict['workers']):
                if i < split_lengths[w]:
                    selected_index = indices_split_by_thread[w][i]
                    current_indices = optimal_indices + [selected_index]
                    current_couch_angles = optimal_couch_angles + \
                                            [feasible_couch_angles[selected_index]]
                    current_gantry_angles = optimal_gantry_angles + \
                                            [feasible_gantry_angles[selected_index]]
                    result = worker.get_obj_function_value.remote(couch_angle = current_couch_angles, 
                                                                  gantry_angle = current_gantry_angles,
                                                                  current_indices = current_indices)
                    worker_results.append(result)
                    selected_indices.append(selected_index)
            
            for j in range(len(worker_results)):
                obj_func_val, debug_dict = ray.get(worker_results[j])

                selected_index = selected_indices[j]
                selected_couch_angle = feasible_couch_angles[selected_index]
                selected_gantry_angle = feasible_gantry_angles[selected_index]
                obj_func_val = self.add_regularization_term(optimal_couch_angles + [selected_couch_angle],
                                                            optimal_gantry_angles + [selected_gantry_angle],
                                                            optimal_indices + [selected_index],
                                                            obj_func_val)

                heat_map['obj_func_vals'].append(obj_func_val)
                heat_map['selected_indices'].append(selected_index)

        assert len(heat_map['obj_func_vals']) == len(feasible_couch_angles)

        next_optimal_idx = heat_map['selected_indices'][np.argmin(heat_map['obj_func_vals'])]
        heat_map['optimal_obj_func_val'] = np.min(heat_map['obj_func_vals'])
        next_optimal_couch_angle = feasible_couch_angles[next_optimal_idx]
        next_optimal_gantry_angle = feasible_gantry_angles[next_optimal_idx]

        heat_map['updated_optimal_indices'] = optimal_indices + [next_optimal_idx]
        heat_map['updated_optimal_couch_angles'] = optimal_couch_angles + [next_optimal_couch_angle]
        heat_map['updated_optimal_gantry_angles'] = optimal_gantry_angles + [next_optimal_gantry_angle]
        heat_map['debug_dict'] = debug_dict

        return heat_map

    def add_regularization_term(self,couch_angles,gantry_angles,angle_ids,obj_val,lambda_val = 5):

        assert len(couch_angles) == len(gantry_angles)

        check_angle = not len(set(angle_ids)) == len(couch_angles)


        return obj_val + lambda_val * check_angle * obj_val

    def convert_angles_to_ids(self,couch_angle_list,gantry_angle_list):
        angle_ids = []

        for j in range(self.num_imrt_beams):
            matching_ids = []
            for i, x in enumerate(self.feasible_couch_angle_list):
                if self.feasible_couch_angle_list[i] == couch_angle_list[j] and \
                        self.feasible_gantry_angle_list[i] == gantry_angle_list[j]:
                    matching_ids.append(i)
            angle_ids.append(matching_ids[0])

        # print('Couch angles:',couch_angle_list,'Gantry angles:',gantry_angle_list)
        # print('Angle ids:',angle_ids)
        return angle_ids

    def evenly_sample_trajectory(self,num_samples):
        idx = np.round(np.linspace(0, len(self.trajectory['full_path']) - 1, num_samples)).astype(int)
        sampled_path = [self.trajectory['full_path'][i] for i in idx]
        self.trajectory['sampled_path'] = sampled_path
        return sampled_path


    def get_trajectory(self):
        return self.trajectory


    def compute_collision_free_points(self):
        couch_angles = np.linspace(-90,90,19) # spaced every 10 deg
        gantry_angles = np.linspace(-170,180,36) # spaced every 10 deg
        xx, yy = np.meshgrid(couch_angles, gantry_angles)
        xx = np.ravel(xx)
        yy = np.ravel(yy)
        collision_free_point_list = []
        collision_point_list = []
        for i in range(len(xx)):
            if xx[i] >= 0 and yy[i] >= 0:
                quadrant = 1
                is_point_feasible = yy[i] <= self.f1(xx[i])
            elif xx[i] < 0 and yy[i] >= 0:
                quadrant = 2
                is_point_feasible = yy[i] <= self.f2(xx[i])
            elif xx[i] < 0 and yy[i] < 0:
                quadrant = 3
                is_point_feasible = yy[i] >= self.f3(xx[i])
            else:
                quadrant = 4
                is_point_feasible = yy[i] >= self.f4(xx[i])

            if is_point_feasible:
                collision_free_point_list.append([xx[i],yy[i]])
            else:
                collision_point_list.append([xx[i],yy[i]])
        if self.visualize:
            self.fig, self.axes1 = plt.subplots()
            self.axes1.plot(np.array(collision_free_point_list)[:,0],np.array(collision_free_point_list)[:,1],'go')
            self.axes1.plot(np.array(collision_point_list)[:,0],np.array(collision_point_list)[:,1],'ro')
            out_path = os.path.join(self.results_dir,'feasible_angles_case_'+str(self.sample_case_num)+'.png')    
            plt.savefig(out_path)
            plt.close()

        feasible_couch_angle_list = list(np.array(collision_free_point_list)[:,0])
        feasible_gantry_angle_list = list(np.array(collision_free_point_list)[:,1])
        # nonneg_feasible_couch_angle_list = copy.deepcopy(feasible_couch_angle_list)
        # nonneg_feasible_gantry_angle_list = copy.deepcopy(feasible_gantry_angle_list)
        # for i in range(len(feasible_couch_angle_list)):
        #     if nonneg_feasible_couch_angle_list[i] < 0:
        #         nonneg_feasible_couch_angle_list[i] = nonneg_feasible_couch_angle_list[i] + 360
        # for i in range(len(feasible_gantry_angle_list)):
        #     if nonneg_feasible_gantry_angle_list[i] < 0:
        #         nonneg_feasible_gantry_angle_list[i] = nonneg_feasible_gantry_angle_list[i] + 360

        return feasible_couch_angle_list, feasible_gantry_angle_list

    def compute_feasibility_of_transition(self,key1,key2):
        '''
        key1: str containing '[couch,gantry]'
        key2: str containing '[couch,gantry]'
        
        computes whether a transition between couch/gantry angles is under max_change_in_couch/max_change_in_gantry
        '''
        c1,g1 = self.convert_key_to_angles(key1)
        c2,g2 = self.convert_key_to_angles(key2)
        if np.abs(c2-c1) <= self.max_change_in_couch and np.abs(g2-g1) <= self.max_change_in_gantry:
            out = True
        else:
            out = False
        return out

    def convert_angles_to_key(self,couch_angle,gantry_angle):
        str_ca = str(couch_angle)
        str_ga = str(gantry_angle)
        str_ca_ga = ','.join([str_ca,str_ga])
        key = '['+str_ca_ga+']'
        return key

    def convert_key_to_angles(self,key):
        out = [float(x) for x in key.strip('[]').split(',')]
        couch_angle = out[0]
        gantry_angle = out[1]
        return couch_angle, gantry_angle

    def load_collision_angles(self, visualize = False):
        df = pd.read_csv(self.collision_angles_path)
        inter = 'linear'
        # quadrant 1 (positive couch and positive gantry angles)
        x1 = np.array(df['quad_1_couch'].dropna())
        y1 = np.array(df['quad_1_gantry'].dropna())
        f1 = interpolate.interp1d(x1,y1,inter,fill_value = 'extrapolate')
        # quadrant 2 (negative couch and positive gantry angles)
        x2 = np.array(df['quad_2_couch'].dropna())
        y2 = np.array(df['quad_2_gantry'].dropna())
        f2 = interpolate.interp1d(x2,y2,inter,fill_value = 'extrapolate')
        # quadrant 3 (negative couch and negative gantry angles)
        x3 = np.array(df['quad_3_couch'].dropna())
        y3 = np.array(df['quad_3_gantry'].dropna())
        f3 = interpolate.interp1d(x3,y3,inter,fill_value = 'extrapolate')
        # quadrant 4 (positive couch and negative gantry angles)
        x4 = np.array(df['quad_4_couch'].dropna())
        y4 = np.array(df['quad_4_gantry'].dropna())
        f4 = interpolate.interp1d(x4,y4,inter,fill_value = 'extrapolate')

        if visualize:
            plt.plot(np.linspace(np.min(x1),np.max(x1),1000),f1(np.linspace(np.min(x1),np.max(x1),1000)),'k-')
            plt.plot(np.linspace(np.min(x2),np.max(x2),1000),f2(np.linspace(np.min(x2),np.max(x2),1000)),'k-')
            plt.plot(np.linspace(np.min(x3),np.max(x3),1000),f3(np.linspace(np.min(x3),np.max(x3),1000)),'k-')
            plt.plot(np.linspace(np.min(x4),np.max(x4),1000),f4(np.linspace(np.min(x4),np.max(x4),1000)),'k-')


        # save interpolated angle boundaries
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4


class CoplanarBAO(BeamAngleOptimization):
    def compute_collision_free_points(self):
        couch_angles = np.linspace(-90,90,19) # spaced every 10 deg
        gantry_angles = np.linspace(-170,180,36) # spaced every 10 deg
        xx, yy = np.meshgrid(couch_angles, gantry_angles)
        xx = np.ravel(xx)
        yy = np.ravel(yy)
        collision_free_point_list = []
        collision_point_list = []
        for i in range(len(xx)):
            if xx[i] == 0:
                collision_free_point_list.append([xx[i],yy[i]])
            else:
                collision_point_list.append([xx[i],yy[i]])



        if self.visualize:
            self.fig, self.axes1 = plt.subplots()
            self.axes1.plot(np.array(collision_free_point_list)[:,0],np.array(collision_free_point_list)[:,1],'go')
            self.axes1.plot(np.array(collision_point_list)[:,0],np.array(collision_point_list)[:,1],'ro')
            out_path = os.path.join(self.results_dir,'feasible_angles_case_'+str(self.sample_case_num)+'.png')    
            plt.savefig(out_path)
            plt.close()

        feasible_couch_angle_list = list(np.array(collision_free_point_list)[:,0])
        feasible_gantry_angle_list = list(np.array(collision_free_point_list)[:,1])
        # nonneg_feasible_couch_angle_list = copy.deepcopy(feasible_couch_angle_list)
        # nonneg_feasible_gantry_angle_list = copy.deepcopy(feasible_gantry_angle_list)
        # for i in range(len(feasible_couch_angle_list)):
        #     if nonneg_feasible_couch_angle_list[i] < 0:
        #         nonneg_feasible_couch_angle_list[i] = nonneg_feasible_couch_angle_list[i] + 360
        # for i in range(len(feasible_gantry_angle_list)):
        #     if nonneg_feasible_gantry_angle_list[i] < 0:
        #         nonneg_feasible_gantry_angle_list[i] = nonneg_feasible_gantry_angle_list[i] + 360

        return feasible_couch_angle_list, feasible_gantry_angle_list

class ParallelSimplexBAO(BeamAngleOptimization):
    def run_simplex_search(self):
        self.max_iters = 15
        self.ftol = 1e-5
        self.ctol = 0
        self.ndims = 2*self.num_imrt_beams

        self.initialize_workers(num_threads=self.num_threads)
        unordered_simplex, unordered_obj_scores = self.get_initial_simplex()


        self.simplex, self.obj_scores = self.order_simplex_points(unordered_simplex,unordered_obj_scores)
        self.round_simplex()


        current_iter = 0
        done = False
        while current_iter < self.max_iters and done == False:
            current_iter, done = self.parallel_step(current_iter)

        final_angle_obj_val = self.obj_scores[0]
        final_point = self.simplex[0]
        final_point_angle_ids = set(self.simplex_angle_id_list[0])
        final_couch_angles, final_gantry_angles = self.convert_angle_id_list(final_point_angle_ids)

        print('Final couch angles:',final_couch_angles)
        print('Final gantry angles:',final_gantry_angles)
        print('Final obj value:',final_angle_obj_val)

        self.trajectory['sampled_path_couch_angles'] = final_couch_angles
        self.trajectory['sampled_path_gantry_angles'] = final_gantry_angles
        self.trajectory['optimal_obj_func_vals'] = final_angle_obj_val

        for worker in self.worker_dict['workers']:
            ray.kill(worker)

        return self.trajectory

    def get_initial_simplex(self):
        feasible_couch_angle_list = copy.deepcopy(self.feasible_couch_angle_list)
        feasible_gantry_angle_list = copy.deepcopy(self.feasible_gantry_angle_list)
        evenly_spaced_gantry_angle_list = list(np.linspace(0,360,self.num_imrt_beams+1)[:-1])
        for i, x in enumerate(evenly_spaced_gantry_angle_list):
            if x > 180:
                evenly_spaced_gantry_angle_list[i] = x - 360

        evenly_spaced_couch_angle_list = list(np.zeros(self.num_imrt_beams))
        evenly_spaced_stack = np.stack([evenly_spaced_couch_angle_list,evenly_spaced_gantry_angle_list],-1)
        evenly_spaced_angle_ids = []

        for j in range(self.num_imrt_beams):
            matching_ids = []
            for i, x in enumerate(feasible_couch_angle_list):
                if feasible_couch_angle_list[i] == evenly_spaced_couch_angle_list[j] and \
                        feasible_gantry_angle_list[i] == evenly_spaced_gantry_angle_list[j]:
                    matching_ids.append(i)
            evenly_spaced_angle_ids.append(matching_ids[0])

        evenly_spaced_angle_ids = np.array(evenly_spaced_angle_ids)

        random.seed(0)
        initial_simplex_id_list = []
        for i in range(self.ndims):
            initial_simplex_id_list.append(random.sample(list(self.angle_indices), self.num_imrt_beams))

        initial_simplex_couch_angle_list = []
        initial_simplex_gantry_angle_list = []
        for i in range(self.ndims):
            simplex_point_ids = initial_simplex_id_list[i]
            simplex_point_couch_angles = np.array([feasible_couch_angle_list[x] for x in simplex_point_ids])
            simplex_point_gantry_angles = np.array([feasible_gantry_angle_list[x] for x in simplex_point_ids])
            initial_simplex_couch_angle_list.append(simplex_point_couch_angles)
            initial_simplex_gantry_angle_list.append(simplex_point_gantry_angles)

        min_dist = np.inf
        closest_simplex_point_id = None
        for i in range(self.ndims):
            stack = np.stack([initial_simplex_couch_angle_list[i],initial_simplex_gantry_angle_list[i]],-1)
            dist = np.linalg.norm(evenly_spaced_stack - stack)
            if dist < min_dist:
                min_dist = dist
                closest_simplex_point_id = i

        assert closest_simplex_point_id is not None
        initial_simplex_id_list[closest_simplex_point_id] = evenly_spaced_angle_ids
        initial_simplex_couch_angle_list[closest_simplex_point_id] = np.array(evenly_spaced_couch_angle_list)
        initial_simplex_gantry_angle_list[closest_simplex_point_id] = np.array(evenly_spaced_gantry_angle_list)
        print('Initial simplex ids:',initial_simplex_id_list)
        print('Initial simplex couch angles:',initial_simplex_couch_angle_list)
        print('Initial simplex gantry angles:',initial_simplex_gantry_angle_list)

        spacing = self.ndims // self.num_workers
        remainder = self.ndims % self.num_workers
        indices = np.arange(self.ndims)
        initial_simplex_split_by_thread = [indices[x:x + spacing] for x in range(0, len(indices), spacing)]
        if len(initial_simplex_split_by_thread) > self.num_workers:
            remaining = np.concatenate(initial_simplex_split_by_thread[self.num_workers:])
            initial_simplex_split_by_thread = initial_simplex_split_by_thread[:self.num_workers]
            for i,rem in enumerate(list(remaining)):
                initial_simplex_split_by_thread[i] = np.concatenate([initial_simplex_split_by_thread[i], np.array([rem])])

        print('initial simplex split by thread:',initial_simplex_split_by_thread)
        results = []
        for w, worker in enumerate(self.worker_dict['workers']):
            # get per worker chunks
            current_worker_angle_indices_list = [initial_simplex_id_list[x] for x in initial_simplex_split_by_thread[w]]
            current_worker_couch_angles_list = [initial_simplex_couch_angle_list[x] for x in initial_simplex_split_by_thread[w]]
            current_worker_gantry_angles_list = [initial_simplex_gantry_angle_list[x] for x in initial_simplex_split_by_thread[w]]
            print('Worker #'+str(w)+' running the following angle indices:',current_worker_angle_indices_list)

            for i in range(len(current_worker_angle_indices_list)):
                current_indices = list(current_worker_angle_indices_list[i])
                current_couch_angles = list(current_worker_couch_angles_list[i])
                current_gantry_angles = list(current_worker_gantry_angles_list[i])

                result = worker.get_obj_function_value.remote(couch_angle = current_couch_angles,
                                                              gantry_angle = current_gantry_angles,
                                                              current_indices = current_indices)
                results.append(result)

        obj_val_list = []
        for i in range(len(results)):
            obj_func_val, _ = ray.get(results[i])
            obj_val_list.append(obj_func_val)

        unordered_simplex = []
        for i in range(len(initial_simplex_couch_angle_list)):
            simplex_point = self.format_point_from_angles(initial_simplex_couch_angle_list[i],
                                                          initial_simplex_gantry_angle_list[i])
            unordered_simplex.append(simplex_point)

        unordered_simplex_scores = obj_val_list
        print('Initial simplex scores:',unordered_simplex_scores)

        assert len(unordered_simplex) == len(obj_val_list)

        return unordered_simplex, unordered_simplex_scores


    def set_num_workers(self,num_threads):
        self.num_workers = num_threads

    def initialize_workers(self, num_threads):
        self.set_num_workers(num_threads)
        self.worker_dict = {'workers': [], }


        temp_results = []
        for i in range(num_threads):
            remote_class = ray.remote(MatRadMetaOpt)
            time.sleep(5)
            worker = remote_class.remote()
            result = worker.reset_bao.remote(self.ct_dir,
                                             self.rtst_dir,
                                             self.meta_optimized_weights,
                                             high_res=False,
                                             acceptable_iter=400,
                                             acceptable_constr_viol_tol=0.01,
                                             ipopt_max_iters=400
                                             )
            self.worker_dict['workers'].append(worker)
            temp_results.append(result)

        for i in range(len(temp_results)):
            _ = ray.get(temp_results[i])


        # precompute dij
        print('Precomputing dose influence matrix on each worker...')

        temp_results = []
        for w, worker in enumerate(self.worker_dict['workers']):
            result = worker.precompute_dij.remote(self.feasible_couch_angle_list,
                                                  self.feasible_gantry_angle_list)
            temp_results.append(result)

        for i in range(len(temp_results)):
            precompute_completed = ray.get(temp_results[i])

    def round_point_to_closest_feasible_config(self,point):
        '''
        point contains 2n angles
        '''
        couch_angles = list(point[::2])
        gantry_angles = list(point[1::2])
        assert len(couch_angles) == len(gantry_angles)

        closest_couch_angles = []
        closest_gantry_angles = []
        for i in range(len(couch_angles)):
            couch_angle = couch_angles[i]
            gantry_angle = gantry_angles[i]
            absolute_diff_couch = lambda x: np.abs(x - couch_angle)
            absolute_diff_gantry = lambda x: np.abs(x - gantry_angle)
            closest_couch_angle = min(self.feasible_couch_angle_list,key=absolute_diff_couch)
            closest_gantry_angle = min(self.feasible_gantry_angle_list,key=absolute_diff_gantry)
            closest_couch_angles.append(closest_couch_angle)
            closest_gantry_angles.append(closest_gantry_angle)

        assert len(closest_couch_angles) == len(closest_gantry_angles)
        assert len(closest_couch_angles) == len(couch_angles)
        rounded_point = self.format_point_from_angles(closest_couch_angles,closest_gantry_angles)
        return rounded_point

    def format_point_from_angles(self,couch_angle_list,gantry_angle_list):
        point = []
        for i in range(len(couch_angle_list)):
            point.append(couch_angle_list[i])
            point.append(gantry_angle_list[i])

        return np.array(point)

    def compute_feasible_angles(self):
        self.load_collision_angles()
        feasible_couch_angle_list, feasible_gantry_angle_list = self.compute_collision_free_points()
        print('Number of feasible angles: ', len(feasible_couch_angle_list))
        assert len(feasible_couch_angle_list) == len(feasible_gantry_angle_list)

        self.trajectory = {}
        self.feasible_couch_angle_list = copy.deepcopy(feasible_couch_angle_list)
        self.feasible_gantry_angle_list = copy.deepcopy(feasible_gantry_angle_list)
        self.angle_indices = np.arange(len(feasible_couch_angle_list))

    def order_simplex_points(self,unordered_simplex,unordered_obj_scores):
        '''
        Expecting all list inputs
        '''
        # order simplex points and scores
        simplex = [point for score, point in sorted(zip(unordered_obj_scores,unordered_simplex), key=lambda pair: pair[0])]
        obj_scores = [score for score, point in sorted(zip(unordered_obj_scores,unordered_simplex), key=lambda pair: pair[0])]
        return simplex, obj_scores

    def round_simplex(self):
        rounded_simplex = []
        for point in self.simplex:
            rounded_point = self.round_point_to_closest_feasible_config(point)
            rounded_simplex.append(rounded_point)
        self.simplex = rounded_simplex

    def round_to_nearest_degree_spacing(self,point):
        '''
        point contains 2n angles
        '''
        self.degree_spacing = 10
        couch_angles = list(point[::2])
        gantry_angles = list(point[1::2])
        assert len(couch_angles) == len(gantry_angles)

        closest_couch_angles = []
        closest_gantry_angles = []
        for i in range(len(couch_angles)):
            closest_couch_angle = np.round(couch_angles[i]/self.degree_spacing)*self.degree_spacing
            closest_gantry_angle = np.round(gantry_angles[i]/self.degree_spacing)*self.degree_spacing
            closest_couch_angles.append(closest_couch_angle)
            closest_gantry_angles.append(closest_gantry_angle)

        assert len(closest_couch_angles) == len(closest_gantry_angles)
        assert len(closest_couch_angles) == len(couch_angles)
        rounded_point = self.format_point_from_angles(closest_couch_angles, closest_gantry_angles)
        return rounded_point

    def check_if_point_is_feasible(self,point):
        '''
        point contains 2n angles
        '''
        couch_angles = list(point[::2])
        gantry_angles = list(point[1::2])
        assert len(couch_angles) == len(gantry_angles)
        checks = []
        try:
            angle_id_list = self.convert_angles_to_ids(couch_angles,gantry_angles)
            check_angle = True
        except:
            check_angle = False

        if check_angle:
            check_angle = len(set(angle_id_list)) == len(couch_angles)

        checks.append(check_angle)

        return np.min(checks)

    def convert_angle_id_list(self,angle_id_list):
        couch_angle_list = [self.feasible_couch_angle_list[x] for x in angle_id_list]
        gantry_angle_list = [self.feasible_gantry_angle_list[x] for x in angle_id_list]
        # print('Couch angles:',couch_angle_list)
        # print('Gantry angles:',gantry_angle_list)
        return couch_angle_list,gantry_angle_list

    def convert_angles_to_ids(self,couch_angle_list,gantry_angle_list):
        angle_ids = []

        for j in range(self.num_imrt_beams):
            matching_ids = []
            for i, x in enumerate(self.feasible_couch_angle_list):
                if self.feasible_couch_angle_list[i] == couch_angle_list[j] and \
                        self.feasible_gantry_angle_list[i] == gantry_angle_list[j]:
                    matching_ids.append(i)
            angle_ids.append(matching_ids[0])

        # print('Couch angles:',couch_angle_list,'Gantry angles:',gantry_angle_list)
        # print('Angle ids:',angle_ids)
        return angle_ids

    def parallel_step(self, current_iter):
        ''''''
        print('Peforming iteration of parallel Nelder-Mead Simplex Routine...')
        self.simplex_param = 0.5
        alpha = 2 * self.simplex_param
        gamma = 4 * self.simplex_param
        rho = self.simplex_param
        sigma = self.simplex_param

        assert len(self.simplex) == self.ndims
        assert len(self.obj_scores) == self.ndims
        for point in self.simplex:
            assert len(point) == self.ndims

        candidate_simplex = copy.deepcopy(self.simplex)
        candidate_obj_scores = copy.deepcopy(self.obj_scores)

        decisions_to_shrink = [False] * (len(self.worker_dict['workers']))

        centroid_list = []

        for worker_id in range(self.num_workers):
            # first calc centroid
            simplex_i = self.ndims - worker_id - 1
            all_but_one_points = self.simplex[:simplex_i] + self.simplex[simplex_i + 1:]
            centroid = np.mean(all_but_one_points, 0)
            centroid_list.append(centroid)

        print('Alpha:', alpha, 'Gamma:', gamma, 'Rho:', rho, 'Sigma:', sigma)

        reflection_point_list = []
        reflection_point_obj_list = []
        reflection_point_simplex_i = []
        reflection_results = []

        expansion_point_list = []
        expansion_point_obj_list = []
        expansion_point_simplex_i = []
        expansion_results = []

        oc_point_list = []
        oc_point_obj_list = []
        oc_point_simplex_i = []
        oc_results = []

        ic_point_list = []
        ic_point_obj_list = []
        ic_point_simplex_i = []
        ic_results = []

        '''Reflection'''
        for worker_id in range(self.num_workers):
            simplex_i = self.ndims - worker_id - 1
            centroid = centroid_list[worker_id]

            vec = centroid - self.simplex[simplex_i]
            reflection_point = centroid + alpha * vec
            reflection_point = self.round_to_nearest_degree_spacing(reflection_point)
            reflection_point_simplex_i.append(simplex_i)
            inside_hull_refl = self.check_if_point_is_feasible(reflection_point)
            if inside_hull_refl:
                print('Reflection point: ', reflection_point)
                current_couch_angles = list(reflection_point[::2])
                current_gantry_angles = list(reflection_point[1::2])
                angle_id_list = self.convert_angles_to_ids(current_couch_angles,current_gantry_angles)
                result_refl = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(couch_angle=current_couch_angles,
                                                              gantry_angle=current_gantry_angles,
                                                              current_indices=angle_id_list)

                reflection_results.append(result_refl)
            else:
                # result_refl = self.worker_dict['workers'][worker_id].return_inf_obj.remote(couch_angle=[],
                #                                               gantry_angle=[],
                #                                               current_indices=[])
                reflection_point = self.round_point_to_closest_feasible_config(reflection_point)
                print('Reflection point: ', reflection_point)
                current_couch_angles = list(reflection_point[::2])
                current_gantry_angles = list(reflection_point[1::2])
                angle_id_list = self.convert_angles_to_ids(current_couch_angles,current_gantry_angles)
                result_refl = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(couch_angle=current_couch_angles,
                                                              gantry_angle=current_gantry_angles,
                                                              current_indices=angle_id_list)
                reflection_results.append(result_refl)
            reflection_point_list.append(reflection_point)

        # get synchronous reflection results
        for result_id in range(len(reflection_results)):
            reflection_point_obj, _ = ray.get(reflection_results[result_id])
            simplex_i = reflection_point_simplex_i[result_id]
            reflection_point = reflection_point_list[result_id]
            is_duplicate = len(set(np.array(self.convert_angles_to_ids(reflection_point[::2],
                                                                       reflection_point[1::2]))))!=self.num_imrt_beams
            reflection_point_obj += is_duplicate*1e6
            print('Reflection point score for simplex #' + str(simplex_i) + ' :', reflection_point_obj)
            reflection_point_obj_list.append(reflection_point_obj)

        # begin simplex logic
        for worker_id in range(len(reflection_point_list)):
            simplex_i = reflection_point_simplex_i[worker_id]
            reflection_point = reflection_point_list[worker_id]
            reflection_point_obj = reflection_point_obj_list[worker_id]

            if reflection_point_obj < self.obj_scores[simplex_i - 1] and reflection_point_obj >= self.obj_scores[0]:
                candidate_simplex[simplex_i] = reflection_point
                candidate_obj_scores[simplex_i] = reflection_point_obj
            elif reflection_point_obj < self.obj_scores[0]:
                '''Expansion'''
                centroid = centroid_list[worker_id]
                vec = centroid - self.simplex[simplex_i]
                expanded_point =centroid + gamma * vec
                expanded_point = self.round_to_nearest_degree_spacing(expanded_point)
                expansion_point_simplex_i.append(simplex_i)
                inside_hull_exp = self.check_if_point_is_feasible(expanded_point)

                if inside_hull_exp:
                    print('Expanded point: ', expanded_point)
                    current_couch_angles = list(expanded_point[::2])
                    current_gantry_angles = list(expanded_point[1::2])
                    angle_id_list = self.convert_angles_to_ids(current_couch_angles, current_gantry_angles)
                    result_exp = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(
                        couch_angle=current_couch_angles,
                        gantry_angle=current_gantry_angles,
                        current_indices=angle_id_list)
                    expansion_results.append(result_exp)
                else:
                    # result_exp = self.worker_dict['workers'][worker_id].return_inf_obj.remote(couch_angle=[],
                    #                                           gantry_angle=[],
                    #                                           current_indices=[])
                    expanded_point = self.round_point_to_closest_feasible_config(expanded_point)
                    print('Expanded point: ', expanded_point)
                    current_couch_angles = list(expanded_point[::2])
                    current_gantry_angles = list(expanded_point[1::2])
                    angle_id_list = self.convert_angles_to_ids(current_couch_angles, current_gantry_angles)
                    result_exp = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(
                        couch_angle=current_couch_angles,
                        gantry_angle=current_gantry_angles,
                        current_indices=angle_id_list)
                    expansion_results.append(result_exp)
                expansion_point_list.append(expanded_point)


            elif reflection_point_obj >= self.obj_scores[simplex_i - 1] and reflection_point_obj < self.obj_scores[
                simplex_i]:
                '''Outside Contraction'''
                centroid = centroid_list[worker_id]
                vec = reflection_point - centroid
                contraction_point_oc = centroid + rho * vec
                contraction_point_oc = self.round_to_nearest_degree_spacing(contraction_point_oc)
                oc_point_simplex_i.append(simplex_i)
                inside_hull_oc = self.check_if_point_is_feasible(contraction_point_oc)
                if inside_hull_oc:
                    print('Outside contraction point: ', contraction_point_oc)
                    current_couch_angles = list(contraction_point_oc[::2])
                    current_gantry_angles = list(contraction_point_oc[1::2])
                    angle_id_list = self.convert_angles_to_ids(current_couch_angles, current_gantry_angles)
                    result_oc = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(
                        couch_angle=current_couch_angles,
                        gantry_angle=current_gantry_angles,
                        current_indices=angle_id_list)

                    oc_results.append(result_oc)
                else:
                    # result_oc = self.worker_dict['workers'][worker_id].return_inf_obj.remote(couch_angle=[],
                    #                                           gantry_angle=[],
                    #                                           current_indices=[])
                    contraction_point_oc = self.round_point_to_closest_feasible_config(contraction_point_oc)
                    print('Outside contraction point: ', contraction_point_oc)
                    current_couch_angles = list(contraction_point_oc[::2])
                    current_gantry_angles = list(contraction_point_oc[1::2])
                    angle_id_list = self.convert_angles_to_ids(current_couch_angles, current_gantry_angles)
                    result_oc = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(
                        couch_angle=current_couch_angles,
                        gantry_angle=current_gantry_angles,
                        current_indices=angle_id_list)
                    oc_results.append(result_oc)
                oc_point_list.append(contraction_point_oc)

            elif reflection_point_obj >= self.obj_scores[simplex_i]:
                '''Inside Contraction'''
                centroid = centroid_list[worker_id]
                vec = self.simplex[simplex_i] - centroid  # self.simplex[-1] - centroid
                contraction_point_ic = centroid + rho * vec
                contraction_point_ic = self.round_to_nearest_degree_spacing(contraction_point_ic)
                ic_point_simplex_i.append(simplex_i)
                inside_hull_ic = self.check_if_point_is_feasible(contraction_point_ic)

                if inside_hull_ic:
                    print('Inside contraction point: ', contraction_point_ic)
                    current_couch_angles = list(contraction_point_ic[::2])
                    current_gantry_angles = list(contraction_point_ic[1::2])
                    angle_id_list = self.convert_angles_to_ids(current_couch_angles, current_gantry_angles)
                    result_ic = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(
                        couch_angle=current_couch_angles,
                        gantry_angle=current_gantry_angles,
                        current_indices=angle_id_list)
                    ic_results.append(result_ic)
                else:
                    # result_ic = self.worker_dict['workers'][worker_id].return_inf_obj.remote(couch_angle=[],
                    #                                           gantry_angle=[],
                    #                                           current_indices=[])
                    contraction_point_ic = self.round_point_to_closest_feasible_config(contraction_point_ic)
                    print('Inside contraction point: ', contraction_point_ic)
                    current_couch_angles = list(contraction_point_ic[::2])
                    current_gantry_angles = list(contraction_point_ic[1::2])
                    angle_id_list = self.convert_angles_to_ids(current_couch_angles, current_gantry_angles)
                    result_ic = self.worker_dict['workers'][worker_id].get_obj_function_value.remote(
                        couch_angle=current_couch_angles,
                        gantry_angle=current_gantry_angles,
                        current_indices=angle_id_list)
                    ic_results.append(result_ic)
                ic_point_list.append(contraction_point_ic)


        '''Get synchronous exp, oc, and ic results'''
        for result_id in range(len(expansion_results)):
            expansion_point_obj, _ = ray.get(expansion_results[result_id])
            simplex_i = expansion_point_simplex_i[result_id]
            expansion_point = expansion_point_list[result_id]
            is_duplicate = len(set(np.array(self.convert_angles_to_ids(expansion_point[::2],
                                                                       expansion_point[1::2]))))!=self.num_imrt_beams
            expansion_point_obj += is_duplicate*1e6
            print('Expansion point score for simplex #' + str(simplex_i) + ' :', expansion_point_obj)
            expansion_point_obj_list.append(expansion_point_obj)
        for result_id in range(len(oc_results)):
            oc_point_obj, _ = ray.get(oc_results[result_id])
            simplex_i = oc_point_simplex_i[result_id]
            oc_point = oc_point_list[result_id]
            is_duplicate = len(set(np.array(self.convert_angles_to_ids(oc_point[::2],
                                                                       oc_point[1::2]))))!=self.num_imrt_beams
            oc_point_obj += is_duplicate*1e6
            print('Outside contraction point score for worker #' + str(simplex_i) + ' :', oc_point_obj)
            oc_point_obj_list.append(oc_point_obj)
        for result_id in range(len(ic_results)):
            ic_point_obj, _ = ray.get(ic_results[result_id])
            simplex_i = ic_point_simplex_i[result_id]
            ic_point = ic_point_list[result_id]
            is_duplicate = len(set(np.array(self.convert_angles_to_ids(ic_point[::2],
                                                                       ic_point[1::2]))))!=self.num_imrt_beams
            ic_point_obj += is_duplicate*1e6
            print('Inside contraction point score for worker #' + str(simplex_i) + ' :', ic_point_obj)
            ic_point_obj_list.append(ic_point_obj)

        assert len(reflection_point_list) == len(reflection_point_obj_list)
        assert len(reflection_point_list) == len(reflection_point_simplex_i)
        assert len(reflection_point_list) == len(reflection_results)

        assert len(expansion_point_list) == len(expansion_point_obj_list)
        assert len(expansion_point_list) == len(expansion_point_simplex_i)
        assert len(expansion_point_list) == len(expansion_results)

        assert len(oc_point_list) == len(oc_point_obj_list)
        assert len(oc_point_list) == len(oc_point_simplex_i)
        assert len(oc_point_list) == len(oc_results)

        assert len(ic_point_list) == len(ic_point_obj_list)
        assert len(ic_point_list) == len(ic_point_simplex_i)
        assert len(ic_point_list) == len(ic_results)

        # expansion logic
        for i in range(len(expansion_point_list)):
            simplex_i = expansion_point_simplex_i[i]
            expansion_point = expansion_point_list[i]
            expansion_point_obj = expansion_point_obj_list[i]
            check_j = None
            for j in range(len(reflection_point_list)):
                if simplex_i == reflection_point_simplex_i[j]:
                    check_j = j

            assert check_j is not None
            if expansion_point_obj < reflection_point_obj_list[check_j]:
                candidate_simplex[simplex_i] = expansion_point
                candidate_obj_scores[simplex_i] = expansion_point_obj
            else:
                candidate_simplex[simplex_i] = reflection_point_list[check_j]
                candidate_obj_scores[simplex_i] = reflection_point_obj_list[check_j]

        # Outside contraction logic
        for i in range(len(oc_point_list)):
            simplex_i = oc_point_simplex_i[i]
            oc_point = oc_point_list[i]
            oc_point_obj = oc_point_obj_list[i]
            check_j = None
            for j in range(len(reflection_point_list)):
                if simplex_i == reflection_point_simplex_i[j]:
                    check_j = j

            assert check_j is not None
            if oc_point_obj < reflection_point_obj_list[check_j]:
                candidate_simplex[simplex_i] = oc_point
                candidate_obj_scores[simplex_i] = oc_point_obj
            else:
                decisions_to_shrink[self.ndims - simplex_i - 1] = True

        # Inside contraction logic
        for i in range(len(ic_point_list)):
            simplex_i = ic_point_simplex_i[i]
            ic_point = ic_point_list[i]
            ic_point_obj = ic_point_obj_list[i]

            if ic_point_obj < self.obj_scores[simplex_i]:
                candidate_simplex[simplex_i] = ic_point
                candidate_obj_scores[simplex_i] = ic_point_obj
            else:
                decisions_to_shrink[self.ndims - simplex_i - 1] = True

        # Shrink logic
        if np.min(decisions_to_shrink) > 0:
            shrink_results = []

            spacing = (self.ndims-1) // self.num_workers
            remainder = (self.ndims-1) % self.num_workers
            indices = np.arange(1, self.ndims)
            shrink_points_split_by_thread = [indices[x:x + spacing] for x in range(0, len(indices), spacing)]
            if len(shrink_points_split_by_thread) > self.num_workers:
                remaining = np.concatenate(shrink_points_split_by_thread[self.num_workers:])
                shrink_points_split_by_thread = shrink_points_split_by_thread[:self.num_workers]
                for i,rem in enumerate(list(remaining)):
                    shrink_points_split_by_thread[i] = np.concatenate([shrink_points_split_by_thread[i], np.array([rem])])

            all_worker_simplex_i = []
            for w, worker in enumerate(self.worker_dict['workers']):
                # get per worker chunks
                simplex_i = [x for x in shrink_points_split_by_thread[w]]
                all_worker_simplex_i += simplex_i
                for si in simplex_i:
                    vec = candidate_simplex[si] - candidate_simplex[0]
                    candidate_simplex[si] = candidate_simplex[0] + sigma * vec
                    candidate_simplex[si] = self.round_to_nearest_degree_spacing(candidate_simplex[si])
                    candidate_simplex[si] = self.round_point_to_closest_feasible_config(candidate_simplex[si])
                    print('Shrink point ' + str(i + 1) + ': ', candidate_simplex[si])

                current_worker_point_list = [candidate_simplex[x] for x in simplex_i]
                current_worker_couch_angles_list = []
                current_worker_gantry_angles_list =[]
                current_worker_angle_ids_list = []
                for point in current_worker_point_list:
                    current_worker_couch_angles_list.append(point[::2])
                    current_worker_gantry_angles_list.append(point[1::2])
                    current_worker_angle_ids_list.append(np.array(self.convert_angles_to_ids(point[::2],point[1::2])))

                for i in range(len(current_worker_point_list)):
                    point = current_worker_point_list[i]
                    inside_hull_shrink_point = self.check_if_point_is_feasible(point)
                    current_indices = list(current_worker_angle_ids_list[i])
                    current_couch_angles = list(current_worker_couch_angles_list[i])
                    current_gantry_angles = list(current_worker_gantry_angles_list[i])
                    if inside_hull_shrink_point:
                        result = worker.get_obj_function_value.remote(couch_angle=current_couch_angles,
                                                                      gantry_angle=current_gantry_angles,
                                                                      current_indices=current_indices)
                        shrink_results.append(result)
                    else:
                        # result = worker.return_inf_obj.remote(couch_angle=[],
                        #                                      gantry_angle=[],
                        #                                      current_indices=[])
                        point = self.round_point_to_closest_feasible_config(point)
                        current_indices = list(np.array(self.convert_angles_to_ids(point[::2],point[1::2])))
                        current_couch_angles = list(point[::2])
                        current_gantry_angles = list(point[1::2])
                        result = worker.get_obj_function_value.remote(couch_angle=current_couch_angles,
                                                                      gantry_angle=current_gantry_angles,
                                                                      current_indices=current_indices)
                        shrink_results.append(result)

            for i,si in enumerate(all_worker_simplex_i):
                obj_func_val, _ = ray.get(shrink_results[i])
                candidate_obj_scores[si] = obj_func_val


        candidate_simplex, candidate_obj_scores = self.order_simplex_points(candidate_simplex,candidate_obj_scores)
        self.simplex = copy.deepcopy(candidate_simplex)
        self.obj_scores = copy.deepcopy(candidate_obj_scores)
        self.round_simplex()

        for i,point in enumerate(self.simplex):
            is_duplicate = len(set(np.array(self.convert_angles_to_ids(point[::2], point[1::2]))))!=self.num_imrt_beams
            if is_duplicate and self.obj_scores[i] < 1e6:
                self.obj_scores[i] += 1e6

        self.simplex, self.obj_scores = self.order_simplex_points(self.simplex, self.obj_scores)
        simplex_angle_id_list = []
        contains_duplicates = []
        for i,point in enumerate(self.simplex):
            simplex_angle_id_list.append(np.array(self.convert_angles_to_ids(point[::2], point[1::2])))
            is_duplicate = len(set(np.array(self.convert_angles_to_ids(point[::2], point[1::2]))))!=self.num_imrt_beams
            contains_duplicates.append(is_duplicate)

        self.simplex_angle_id_list = simplex_angle_id_list


        print('Simplex points:',self.simplex)
        print('Obj values:',self.obj_scores)
        print('Angle Ids:',simplex_angle_id_list)
        print('Contains Duplicates:',contains_duplicates)
        print('Current iteration:',current_iter)
        done = self.check_termination()

        return current_iter+1, done

    def check_termination(self):
        '''calculate distances'''
        distances = []
        for i in range(1, self.ndims):
            # distances.append(np.max(np.abs(self.projected_simplex[i] - self.projected_simplex[0])))
            distances.append(np.linalg.norm(self.simplex[i] - self.simplex[0]))
        max_dist = np.max(distances)
        print('Distances: ', distances)

        score_diff = []
        for i in range(1, self.ndims):
            score_diff.append(np.abs(self.obj_scores[i] - self.obj_scores[0]))
        '''check if algo should terminate'''
        if np.max(score_diff) <= self.ftol or max_dist <= self.ctol:
            done = True
        else:
            done = False

        return done


class ParallelSimplexCoplanarBAO(ParallelSimplexBAO):
    def compute_collision_free_points(self):
        couch_angles = np.linspace(-90,90,19) # spaced every 10 deg
        gantry_angles = np.linspace(-170,180,36) # spaced every 4 deg
        xx, yy = np.meshgrid(couch_angles, gantry_angles)
        xx = np.ravel(xx)
        yy = np.ravel(yy)
        collision_free_point_list = []
        collision_point_list = []
        for i in range(len(xx)):
            if xx[i] == 0:
                collision_free_point_list.append([xx[i],yy[i]])
            else:
                collision_point_list.append([xx[i],yy[i]])



        if self.visualize:
            self.fig, self.axes1 = plt.subplots()
            self.axes1.plot(np.array(collision_free_point_list)[:,0],np.array(collision_free_point_list)[:,1],'go')
            self.axes1.plot(np.array(collision_point_list)[:,0],np.array(collision_point_list)[:,1],'ro')
            out_path = os.path.join(self.results_dir,'feasible_angles_case_'+str(self.sample_case_num)+'.png')
            plt.savefig(out_path)
            plt.close()

        feasible_couch_angle_list = list(np.array(collision_free_point_list)[:,0])
        feasible_gantry_angle_list = list(np.array(collision_free_point_list)[:,1])

        return feasible_couch_angle_list, feasible_gantry_angle_list

if __name__ == "__main__":
    import pdb
    memory = 12000000000 * 9
    object_store_memory = 2500000000 * 8
    ray.init(memory=memory, object_store_memory=object_store_memory, ignore_reinit_error=True, log_to_driver=False)

    weights = np.array([149.00156388,  94.74809166,  56.80080192,  36.96814374,36.96814374,  62.17189567,  63.3413594 ])
    data_dir = './clean_prostate_data/isodose/usable/'
    results_dir = 'trash'

    trajectory_searcher = ParallelSimplexCoplanarBAO(meta_optimized_weights = weights,
                            data_dir = data_dir,
                            sample_case_num = 0,
                            visualize = True,
                            results_dir = results_dir,
                            num_threads=6)


    # trajectory_searcher.generate_cost_map()
    # trajectory_dict = trajectory_searcher.get_trajectory()
    trajectory_dict = trajectory_searcher.run_simplex_search()
    couch_angle_list = trajectory_dict['sampled_path_couch_angles']
    gantry_angle_list = trajectory_dict['sampled_path_gantry_angles']
    print(trajectory_dict)