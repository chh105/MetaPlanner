
import gym
import matlab.engine
import numpy as np
import os
import scipy.special as sc
import textdistance
import copy
import psutil
from gym.utils import seeding
import random

import glob
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import pandas as pd
from scipy.stats import wilcoxon

import ray


os.environ['matRad'] = '../matRad-dev_VMAT/'

CI_RANGE = [0.85,0.95]
HI_RANGE = [5,8]
R90_RANGE = [1.1,1.5]
R50_RANGE = [3.5,4.5]

NUM_STRUCTURES = 6 # not including automatically generated structures
# MAX_DOSES = [82,82,50,50,82,5,74]
MAX_DOSES = [30,20,15,15,82,4]

PRESCRIBED_DOSE = 74
NORMALIZED_DOSE = 74
RING_DIST = 10

CPU_USAGE_THRESH = 8 # threshold in percent
CPU_USAGE_INTERVAL = np.linspace(4,10,16)#[2,5] # check usage over __ sec

BINARY_SEARCH_THRESH = 0.02

class MatRadWeightedOpt(gym.Env):
    def __init__(self, env_config=None):
        '''
        switcher = {
            '0': 'square underdosing',
            '1': 'square overdosing',
            '2': 'square deviation',
            '3': 'mean',
            '4': 'EUD'
        }
        '''
        self.current_step = 0
        self.seed()
        self._set_cpu_affinity()
        self.constr_dict = {
            'name_list': ['rectum'] + \
                         ['bladder'] + \
                         ['fh r'] + \
                         ['fh l'] + \
                         ['ptv'] + \
                         ['body'],
            'obj_type_list': [1] + \
                             [1] + \
                             [1] + \
                             [1] + \
                             [2] + \
                             [1],
            'param_list': [[0],
                           [0],
                           [30],
                           [30],
                           [PRESCRIBED_DOSE],
                           [30]],
            'obj_num_list': list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1),
            'priority_list': [2] + [2] + [2] + [2] + [1] + [2],
            'obj_weight_list': [0] + \
                               [0] + \
                               [0] + \
                               [0] + \
                               [0] + \
                               [0],
            'preference_tier_list': [2] + \
                                    [2] + \
                                    [3] + \
                                    [3] + \
                                    [1] + \
                                    [3]
        }


    def set_ids(self):
        self.adjusted_ids = [0, 1, 2, 3, 4, 5]
        self.dvh_ids = [0, 1, 2, 3, 4, 5]
        self.score_ids = [0, 1, 2, 3, 4, 5]

    def reset_weighted_opt(self,
                           ctDir,
                           rtStDir,
                           couch_angle_list=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           gantry_angle_list=[0, 40, 80, 120, 160, 200, 240, 280, 320],
                           high_res=False,
                           acceptable_iter=200,
                           acceptable_constr_viol_tol=0.01,
                           ipopt_max_iters=200):

        self.ctDir = ctDir
        self.rtStDir = rtStDir
        self.eng = matlab.engine.start_matlab()
        self.eng.eval('addpath(genpath("' + os.environ['matRad'] + '"));')
        self.eng.eval('clear all; close all; clc;', nargout=0)
        self._set_ipopt_acceptable_iter(acceptable_iter=acceptable_iter)
        self._set_ipopt_acceptable_constr_viol_tol(acceptable_constr_viol_tol=acceptable_constr_viol_tol)
        self._set_max_iters(max_iters=ipopt_max_iters)
        self.done = False
        self.prev_sf = 1.0
        self.current_step = 0

        self.curr_param_list = copy.deepcopy(self.constr_dict['param_list'])

        self._import_dicom_with_angles(ctDir,
                                       rtStDir,
                                       couch_angle_list=couch_angle_list,
                                       gantry_angle_list=gantry_angle_list,
                                       high_res=high_res)

        self.remove_unwanted_structs()
        # self.create_ring_struct()
        self.checklist = list(np.arange(len(self.constr_dict['name_list'])))
        self.set_ids()

        self._set_all_structure_priorities(priority=np.max(np.array(self.constr_dict['priority_list'])) + 1)

        self._construct_objectives_and_constraints(checklist=self.checklist)

        num_items = []
        for key, val in self.constr_dict.items():
            num_items.append(len(val))

        assert np.min(num_items) == np.max(num_items)

    def reset_weighted_opt_vmat(self,
                           ctDir,
                           rtStDir,
                           high_res=False,
                           acceptable_iter=200,
                           acceptable_constr_viol_tol=0.01,
                           ipopt_max_iters=200,
                           additional_vmat_couch_angles = [],
                           additional_vmat_gantry_angles = [],
                           additional_fmo_angles = []):

        self.ctDir = ctDir
        self.rtStDir = rtStDir
        self.eng = matlab.engine.start_matlab()
        self.eng.eval('addpath(genpath("' + os.environ['matRad'] + '"));')
        self.eng.eval('clear all; close all; clc;', nargout=0)
        self._set_ipopt_acceptable_iter(acceptable_iter=acceptable_iter)
        self._set_ipopt_acceptable_constr_viol_tol(acceptable_constr_viol_tol=acceptable_constr_viol_tol)
        self._set_max_iters(max_iters=ipopt_max_iters)
        self.done = False
        self.prev_sf = 1.0
        self.current_step = 0

        self.curr_param_list = copy.deepcopy(self.constr_dict['param_list'])

        self._import_dicom_vmat(ctDir,
                               rtStDir,
                               high_res=high_res)

        self.remove_unwanted_structs()
        # self.create_ring_struct()
        self.checklist = list(np.arange(len(self.constr_dict['name_list'])))
        self.set_ids()

        self._set_all_structure_priorities(priority=np.max(np.array(self.constr_dict['priority_list'])) + 1)

        self._construct_objectives_and_constraints(checklist=self.checklist)

        num_items = []
        for key, val in self.constr_dict.items():
            num_items.append(len(val))

        assert np.min(num_items) == np.max(num_items)

        if additional_vmat_couch_angles:
            assert len(additional_vmat_couch_angles) == len(additional_vmat_gantry_angles)
            for i in range(len(additional_vmat_couch_angles)):
                self.eng.eval('pln.propStf.couchAngles(end+1) = '+str(additional_vmat_couch_angles[i])+';',nargout=0)
                self.eng.eval('pln.propStf.gantryAngles(end+1) = '+str(additional_vmat_gantry_angles[i])+';',nargout=0)
                self.eng.eval('pln.propStf.DAOGantryAngles(end+1) = '+str(additional_vmat_gantry_angles[i])+';',nargout=0)
            for i in range(len(additional_fmo_angles)):
                self.eng.eval('pln.propStf.FMOGantryAngles(end+1) = '+str(additional_fmo_angles[i])+';',nargout=0)

            self.eng.eval('pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);',nargout=0)
            self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);',nargout=0)
            self.eng.eval('stf = matRad_generateStf(ct,cst,pln);',nargout=0)
        else:
            dao_spacing = int(self.eng.eval('pln.propOpt.VMAToptions.maxDAOGantryAngleSpacing', nargout=1))
            gantry_spacing = int(self.eng.eval('pln.propOpt.VMAToptions.maxGantryAngleSpacing', nargout=1))
            fmo_spacing = int(self.eng.eval('pln.propOpt.VMAToptions.maxFMOGantryAngleSpacing', nargout=1))
            offset_dao = dao_spacing // 2
            offset_gantry = gantry_spacing // 2
            offset_fmo = fmo_spacing // 2
            offset = offset_dao + offset_fmo + 360
            self.eng.eval('pln.propStf.couchAngles = [pln.propStf.couchAngles,pln.propStf.couchAngles];', nargout=0)
            self.eng.eval(
                'pln.propStf.gantryAngles = [pln.propStf.gantryAngles pln.propStf.gantryAngles+' + str(offset) + '];',
                nargout=0)
            self.eng.eval(
                'pln.propStf.DAOGantryAngles = [pln.propStf.DAOGantryAngles pln.propStf.DAOGantryAngles+' + str(
                    offset) + '];', nargout=0)
            self.eng.eval(
                'pln.propStf.FMOGantryAngles = [pln.propStf.FMOGantryAngles pln.propStf.FMOGantryAngles+' + str(
                    offset) + '];', nargout=0)
            self.eng.eval('pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);', nargout=0)
            self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);',
                          nargout=0)
            self.eng.eval('stf = matRad_generateStf(ct,cst,pln);', nargout=0)



    def run_seq_and_dao(self):
        self.eng.eval('run_seq_and_dao',nargout=0)


    def create_ring_struct(self):

        print('Creating ring structure around targets...')
        ring_priority = 2
        '''Combined Ring'''
        self.constr_dict['name_list'] += ['mrad_all_target_combined_ring']
        self.constr_dict['obj_type_list'] += [1]
        self.constr_dict['param_list'] += [[30]]
        self.constr_dict['obj_num_list'] += list(np.arange(1) + 1)
        self.constr_dict['priority_list'] += [ring_priority]
        self.constr_dict['obj_weight_list'] += [0]
        self.constr_dict['preference_tier_list'] += [3]
        # ring structs always added to end of cst
        self.constr_dict['struct_num_list'] += [np.max(self.constr_dict['struct_num_list']) + 3]
        ''''''
        self.adjusted_ids.append(6)
        self.score_ids.append(6)

        target_id = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        expansion_radius = 5
        self.eng.eval(
            '[cst] = createRingStructure( ct, cst, ' + str([target_id]) + ',' + str([expansion_radius]) + ', 2);',
            nargout=0)

    def _set_cpu_affinity(self, available_cpus=list(np.arange(os.cpu_count()))):

        # '''set available cpus'''
        # print("Number of CPUs:", os.cpu_count())

        viable_cpus = []
        usage_interv = CPU_USAGE_INTERVAL
        usage_thresh = CPU_USAGE_THRESH
        while len(viable_cpus) < 1 and usage_thresh < 100:
            interv = usage_interv[int(self.np_random.uniform(0, len(usage_interv)))]
            cpu_usages = np.array(psutil.cpu_percent(interval=interv, percpu=True))
            cpu_ids = np.arange(len(cpu_usages))[:]
            cpu_usages = cpu_usages[:]
            viable_cpus = cpu_ids[cpu_usages < usage_thresh]
            usage_thresh += 30

        viable_cpus = [x for x in viable_cpus if x in available_cpus]

        if len(viable_cpus) == 0:
            idx = int(self.np_random.uniform(0, len(available_cpus)))
            affinity_mask = {available_cpus[idx]}
        else:
            idx = int(self.np_random.uniform(0, len(viable_cpus)))
            cpu_id = int(viable_cpus[idx])
            affinity_mask = {cpu_id}
        pid = 0
        os.sched_setaffinity(pid, affinity_mask)
        affinity = os.sched_getaffinity(pid)

        # Print the result
        print("Now, process is eligible to run on:", affinity)

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_max_iters(self,max_iters = 400):

        self.eng.eval('global max_iter_limit;',nargout=0)
        self.eng.eval('max_iter_limit = '+str(max_iters)+';',nargout=0)

    def _set_ipopt_acceptable_iter(self,acceptable_iter = 3):

        self.eng.eval('global acceptable_iter_limit;',nargout=0)
        self.eng.eval('acceptable_iter_limit = '+str(acceptable_iter)+';',nargout=0)

    def _set_ipopt_acceptable_constr_viol_tol(self,acceptable_constr_viol_tol = 0.001):

        self.eng.eval('global acceptable_constr_viol_tol_limit;',nargout=0)
        self.eng.eval('acceptable_constr_viol_tol_limit = '+str(acceptable_constr_viol_tol)+';',nargout=0)

    def _import_dicom_with_angles(self, ctDir, rtStDir, couch_angle_list, gantry_angle_list, high_res = False):

        self.eng.eval('ctDir = "'+ctDir+'";', nargout=0)
        self.eng.eval('rtStDir = "'+rtStDir+'";', nargout=0)

        mm_res = 5
        bixel_width = 7
        if high_res:
            mm_res = 3.5
            bixel_width = 5


        gantry_angle_str = [str(x) for x in gantry_angle_list]
        couch_angle_str = [str(x) for x in couch_angle_list]
        str_of_gantry_angles = ','.join(gantry_angle_str)
        str_of_couch_angles = ','.join(couch_angle_str)
        self.eng.eval('gantryAngles = ['+str_of_gantry_angles+'];', nargout=0)
        self.eng.eval('couchAngles = ['+str_of_couch_angles+'];', nargout=0)

        self.eng.eval('[ct, cst, pln, stf] = import_dicom_with_predefined_angles( ctDir, rtStDir, couchAngles, gantryAngles, '+str(mm_res)+', '+str(bixel_width)+' );',nargout=0)


    def _import_dicom_vmat(self, ctDir, rtStDir, high_res = False):

        self.eng.eval('ctDir = "'+ctDir+'";', nargout=0)
        self.eng.eval('rtStDir = "'+rtStDir+'";', nargout=0)

        mm_res = 5
        bixel_width = 7
        if high_res:
            mm_res = 3.5
            bixel_width = 5

        self.eng.eval('[ct, cst, pln, stf] = import_dicom_vmat( ctDir, rtStDir, '+str(mm_res)+', '+str(bixel_width)+' );',nargout=0)

    def _get_struct_names(self):
        names = self.eng.eval('cst(:,2)',nargout=1)
        return names

    def _find_closest_struct_name(self,name='',search_list = [], thresh = 0.85):
        rat = textdistance.RatcliffObershelp()
        idx = None
        score = 0
        for i,search_name in enumerate(search_list):
            new_score = rat.normalized_similarity(name.lower(),search_name.lower())
            if new_score>score:
                score = new_score
                idx = i

        # if idx is None or score<thresh:
        if idx is None:
            return None,None

        return idx, search_list[idx]

    def _return_struct_num_list(self,name_list = []):
        search_list = self._get_struct_names()
        struct_num_list = []
        self.searched_name_list = []
        for i,name in enumerate(name_list):
            idx, searched_name = self._find_closest_struct_name(name=name,search_list=search_list)
            print('Name: ',name,' Searched Name: ', searched_name)
            assert idx is not None, 'Could not find structure index!'
            struct_num_list.append(idx+1)
            self.searched_name_list.append(searched_name)
        return struct_num_list

    def _set_all_structure_priorities(self, priority = None):
        n = int(self.eng.eval('size(cst,1);',nargout=1))
        for i in range(n):
            self.eng.eval('cst{'+str(i+1)+',5}.Priority = '+str(priority)+';', nargout=0)

    def _clear_all_obj(self):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        num_struct = self.eng.eval('size(cst,1);',nargout = 1)
        print('Number of structures: ',num_struct)
        for i in range(1,int(num_struct)+1):
            self.eng.eval('cst{'+str(i)+',6} = [];', nargout=0)

    def _add_obj(self, struct_num = None, obj_type = None, is_first_obj_bool = True):
        ''' Assumes structure cell array is stored in 'cst'
        :param struct_num: which structure to add objective to
        :param obj_type: integer chosen from the following [0: EUD, 1: Max DVH, 2: Mean Dose, 3: Min DVH, 4: Squared Deviation, 5: Squared Overdosing,
                          6: Squared Underdosing, 7: Min/Max Dose Constraint, 8: DVH Constraint, 9: EUD Constraint,
                          10: Mean Dose Constraint]
        :param is_first_obj_bool: boolean for whether the added objective is the first one or not
        :return:
        '''
        switcher = {
            '0': 'square underdosing',
            '1': 'square overdosing',
            '2': 'square deviation',
            '3': 'mean',
            '4': 'EUD'
        }

        self.eng.eval('cstIndex = '+str(struct_num)+';', nargout=0)
        self.eng.eval("cst{cstIndex, 6}(end+1,1).type = '" + switcher[str(obj_type)]+"';", nargout=0)
        self.eng.eval('cst{cstIndex, 6}(end,1).dose = 0;', nargout=0)
        self.eng.eval('cst{cstIndex, 6}(end,1).penalty = 0;', nargout=0)
        self.eng.eval('cst{cstIndex, 6}(end,1).EUD = nan;', nargout=0)
        self.eng.eval('cst{cstIndex, 6}(end,1).volume = nan;', nargout=0)
        self.eng.eval("cst{cstIndex, 6}(end,1).robustness = 'none';", nargout=0)

    def _change_obj_params(self,struct_num = None, obj_num = None, param_val_list = []):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        self.eng.eval('cst{'+str(struct_num)+',6}('+str(obj_num)+').dose = '
                      +str(param_val_list[0])+';', nargout=0)

    def _change_structure_priority(self, struct_num = None, priority = None):
        self.eng.eval('cst{'+str(struct_num)+',5}.Priority = '+str(priority)+';', nargout=0)

    def _change_obj_weight(self,struct_num = None, obj_num = None, penalty = None):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        self.eng.eval('cst{'+str(struct_num)+',6}('+str(obj_num)+').penalty = '+str(penalty)+';', nargout=0)

    def _construct_objectives_and_constraints(self, checklist=None):

        self._clear_all_obj()

        if checklist is None:
            checklist = copy.deepcopy(self.adjusted_constraint_ids) + [1, 8, 13, 15, 20]

        for c in range(len(self.constr_dict['name_list'])):

            if c not in checklist and self.constr_dict['priority_list'][c] != 1:
                continue

            self._add_obj(struct_num=self.constr_dict['struct_num_list'][c],
                          obj_type=self.constr_dict['obj_type_list'][c])
            self._change_obj_params(struct_num=self.constr_dict['struct_num_list'][c],
                                    obj_num=self.constr_dict['obj_num_list'][c],
                                    param_val_list=self.constr_dict['param_list'][c])
            self._change_structure_priority(struct_num=self.constr_dict['struct_num_list'][c],
                                            priority=self.constr_dict['priority_list'][c])

            if self.constr_dict['obj_weight_list'][c] is not None:
                weight = self.constr_dict['obj_weight_list'][c]
                self._change_obj_weight(struct_num=self.constr_dict['struct_num_list'][c],
                                        obj_num=self.constr_dict['obj_num_list'][c],
                                        penalty=weight)

    def _run_optimizer(self):
        self.eng.eval('run_optimizer_gym;', nargout=0)

    def _run_dose_calc(self):

        self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);',nargout=0)
        self.eng.eval('run_dose_calc_gym;', nargout=0)

    def remove_unwanted_structs(self):
        '''Remove unwanted structures'''
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('Struct_num_list: ', struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')

        unique_structs = list(sorted(set(struct_num_list), key=struct_num_list.index))
        assert NUM_STRUCTURES == len(unique_structs)

        self.eng.eval('struct_ids = ' + str(unique_structs) + ';', nargout=0)
        self.eng.eval('cst_small = cst(struct_ids,:);', nargout=0)
        self.eng.eval('cst = cst_small;', nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('New Struct_num_list: ', struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')

        self.constr_dict['struct_num_list'] = struct_num_list
        assert len(self.constr_dict['struct_num_list']) == len(
            self.constr_dict['name_list']), 'Mismatch struct_num_list and name_list'

    def set_obj_weights(self, weights):

        weights = np.array(weights)
        for i, weight in enumerate(weights):
            c = self.adjusted_ids[i]
            self._change_obj_weight(struct_num=self.constr_dict['struct_num_list'][c],
                                    obj_num=self.constr_dict['obj_num_list'][c],
                                    penalty=weight)

    def get_mean_constraint_vals_from_nonoverlapping_structs(self):
        self.eng.eval('non_overlap_qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose);', nargout=0)
        means = []
        for cst_id in self.nonoverlapping_oar_struct_num_list:
            mu = self.eng.eval('non_overlap_qi('+str(cst_id)+').mean', nargout=1)
            num_of_fractions = self.eng.eval('pln.numOfFractions',nargout=1)
            print('Nonoverlapping oar:',self.eng.eval('non_overlap_qi('+str(cst_id)+').name', nargout=1))
            print('Mean:',mu*num_of_fractions)
            print('# of Fractions:',num_of_fractions)
            means.append(mu*num_of_fractions)

        return np.array(means)

    def get_mean_of_struct(self,cst_id):
        self.eng.eval('struct_qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose);', nargout=0)
        mu = self.eng.eval('struct_qi('+str(cst_id)+').mean', nargout=1)
        return mu


    def run_wo_and_get_constraint_vals(self, weights):
        self.set_obj_weights(weights)
        # run FMO
        try:
            self._run_optimizer()
        except:
            raise ValueError('Problem running optimizer')
        constraint_vals = self.get_mean_constraint_vals_from_nonoverlapping_structs()
        self.constraint_vals = copy.deepcopy(constraint_vals)

        return constraint_vals

    def run_wo(self,weights):
        self.set_obj_weights(weights)
        # run FMO
        try:
            self._run_optimizer()
        except:
            raise ValueError('Problem running optimizer')

        self.current_step += 1
        if self.current_step % 5 == 0:
            self.eng.eval('close all;',nargout=0)

    def render(self,visual=False):
        if visual:
            self.eng.eval('matRadGUI',nargout=0)

    def get_ci_and_hi(self,ptv_struct_num = None, pres_dose = PRESCRIBED_DOSE):
        # num_of_fractions = self.eng.eval('pln.numOfFractions', nargout=1)
        # ref_dose = pres_dose / num_of_fractions
        if ptv_struct_num is None:
            target_id = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        else:
            target_id = ptv_struct_num

        self.eng.eval('struct_qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose);', nargout=0)
        ref_dose = self.eng.eval('struct_qi(' + str(target_id) + ').D_95;', nargout=1)

        self.eng.eval('ptv_qi = calc_conformity_and_homogeneity(cst,pln,resultGUI.physicalDose,' + \
                      str(ref_dose) + ');', nargout=0)
        ci_val = self.eng.eval('ptv_qi(' + str(target_id) + ').CI', nargout=1)
        hi_val = self.eng.eval('ptv_qi(' + str(target_id) + ').HI', nargout=1)
        print('CI from plan:', ci_val)
        print('HI from plan:', hi_val)
        return ci_val,hi_val

    def get_r50_and_r90(self,ptv_struct_num = None, pres_dose = PRESCRIBED_DOSE):
        # num_of_fractions = self.eng.eval('pln.numOfFractions', nargout=1)
        # ref_dose = pres_dose / num_of_fractions

        if ptv_struct_num is None:
            target_id = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        else:
            target_id = ptv_struct_num

        self.eng.eval('struct_qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose);', nargout=0)
        ref_dose = self.eng.eval('struct_qi(' + str(target_id) + ').D_95;', nargout=1)

        self.eng.eval('dose_spill_qi = calc_dose_spill_indices(cst,pln,resultGUI.physicalDose,' + \
                      str(ref_dose) + ');', nargout=0)
        r50_val = self.eng.eval('dose_spill_qi(' + str(target_id) + ').R50', nargout=1)
        r90_val = self.eng.eval('dose_spill_qi(' + str(target_id) + ').R90', nargout=1)
        print('R50 from plan:', r50_val)
        print('R90 from plan:', r90_val)
        return r50_val, r90_val

    def get_ptv_stats_from_treatment_plan(self, ctDir, rtStDir, rtDoseDir):
        self.eng = matlab.engine.start_matlab()
        # self.eng.eval('cd ' + os.environ['matRad'],nargout=0)
        self.eng.eval('addpath(genpath("' + os.environ['matRad'] + '"));')
        # self.eng.eval('set(0,"DefaultFigureVisible","off")',nargout=0)
        self.eng.eval('clear all; close all; clc;', nargout=0)
        self.eng.eval('ctDir = "' + ctDir + '";', nargout=0)
        self.eng.eval('rtStDir = "' + rtStDir + '";', nargout=0)
        self.eng.eval('rtDoseDir = "' + rtDoseDir + '";', nargout=0)
        self.eng.eval(
            '[ct, cst, pln, stf, resultGUI] = import_dicom_return_stf_and_result_gui( ctDir, rtStDir, rtDoseDir);',
            nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('Struct_num_list: ', struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')

        '''Remove unwatned structures'''
        unique_structs = list(sorted(set(struct_num_list), key=struct_num_list.index))
        assert NUM_STRUCTURES == len(unique_structs)

        self.eng.eval('struct_ids = ' + str(unique_structs) + ';', nargout=0)
        self.eng.eval('cst_small = cst(struct_ids,:);', nargout=0)
        self.eng.eval('cst = cst_small;', nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('New Struct_num_list: ', struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')
        ''''''
        self.constr_dict['struct_num_list'] = struct_num_list
        assert len(self.constr_dict['struct_num_list']) == len(
            self.constr_dict['name_list']), 'Mismatch struct_num_list and name_list'
        self._set_all_structure_priorities(priority=np.max(np.array(self.constr_dict['priority_list'])) + 1)

        # assuming prescribed dose is at D95
        self.eng.eval('qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose,[],[95]);', nargout=0)
        num_structs = self.eng.eval('size(qi,2)', nargout=1)
        struct_num_set = list(
            sorted(set(self.constr_dict['struct_num_list']), key=self.constr_dict['struct_num_list'].index))

        # #assuming ptv is the second to last
        # ptv_struct_num = struct_num_set[-2]
        ptv_struct_num = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        print('PTV structure num:', ptv_struct_num)
        prescribed_dose = self.eng.eval('qi(' + str(ptv_struct_num) + ').D_95', nargout=1)
        num_of_fractions = self.eng.eval('pln.numOfFractions', nargout=1)
        prescribed_dose = prescribed_dose * num_of_fractions
        print('Prescribed Dose:', prescribed_dose)

        self._clear_all_obj()
        self._add_obj(struct_num=ptv_struct_num,
                      obj_type=4)
        self._change_obj_params(struct_num=ptv_struct_num,
                                obj_num=1,
                                param_val_list=[prescribed_dose])
        self._change_structure_priority(struct_num=ptv_struct_num,
                                        priority=1)

        self.ref_prescribed_dose = prescribed_dose
        self.ptv_struct_num = ptv_struct_num
        ci_val, hi_val = self.get_ci_and_hi(ptv_struct_num=ptv_struct_num, pres_dose=prescribed_dose)
        r50_val, r90_val = self.get_r50_and_r90(ptv_struct_num=ptv_struct_num, pres_dose=prescribed_dose)

        return ci_val, hi_val, r50_val, r90_val

    def get_obj_function_value(self, couch_angle, gantry_angle, current_indices=None, debug=True):
        '''
        Parameters
        ----------
        couch_angle: list
            expects a list containing at least 1 entry
        gantry_angle: list
            expects a list containing at least 1 entry
        current_indices: list
            expects a list containing at least 1 entry

        '''
        assert isinstance(couch_angle, list)
        assert isinstance(gantry_angle, list)
        assert len(couch_angle) >= 1
        assert len(gantry_angle) >= 1
        assert len(couch_angle) == len(gantry_angle)

        gantry_angle_str = [str(x) for x in gantry_angle]
        couch_angle_str = [str(x) for x in couch_angle]
        str_of_gantry_angles = ','.join(gantry_angle_str)
        str_of_couch_angles = ','.join(couch_angle_str)
        self.eng.eval('pln.propStf.gantryAngles = [' + str_of_gantry_angles + '];', nargout=0)
        self.eng.eval('pln.propStf.couchAngles = [' + str_of_couch_angles + '];', nargout=0)

        self.eng.eval('pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);', nargout=0)
        self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);',
                      nargout=0)

        self.eng.eval('stf = matRad_generateStf(ct,cst,pln);', nargout=0)
        if current_indices == None:
            self._run_dose_calc()
        else:
            debug_dict = self.get_precomp_dij(current_indices)

        try:
            self._run_optimizer()
            val = self.eng.eval('resultGUI.finalObjectiveFunctionValue', nargout=1)
        except:
            crash_log = {}
            crash_log['couch_angles'] = couch_angle
            crash_log['gantry_angles'] = gantry_angle
            np.save('crash_log.npy', crash_log)

        # print('Couch angles: ',couch_angle)
        # print('Gantry angles: ',gantry_angle)
        # print('Objective function value:',val)

        if debug == True:
            return val, debug_dict
        else:
            return val

    def get_precomp_dij(self, current_indices, sanity_check=False):
        '''
        Parameters
        ----------
        current_indices: list
            expects a list containing python indices for accessing the dij structure

        Returns
        -------

        '''
        if sanity_check:
            self._run_dose_calc()
            self.eng.eval('temp_dij = dij;', nargout=0)

        assert isinstance(current_indices, list)
        assert len(current_indices) >= 1

        # convert python indices to matlab
        current_indices = list(np.array(current_indices) + 1)

        self.eng.eval('clear dij;', nargout=0)
        self.eng.eval('clear physical_dose;', nargout=0)

        self.eng.eval('dij.radiationMode = precomp_dij.radiationMode;', nargout=0)
        self.eng.eval('dij.numOfBeams = ' + str(len(current_indices)) + ';', nargout=0)
        self.eng.eval('dij.numOfVoxels = precomp_dij.numOfVoxels;', nargout=0)
        self.eng.eval('dij.resolution = precomp_dij.resolution;', nargout=0)
        self.eng.eval('dij.dimensions = precomp_dij.dimensions;', nargout=0)
        self.eng.eval('dij.numOfScenarios = precomp_dij.numOfScenarios;', nargout=0)
        self.eng.eval('dij.weightToMU = precomp_dij.weightToMU;', nargout=0)
        self.eng.eval('dij.scaleFactor = precomp_dij.scaleFactor;', nargout=0)

        if len(current_indices) == 1:
            self.eng.eval('dij.numOfRaysPerBeam = precomp_dij.numOfRaysPerBeam(' + str(current_indices[0]) + ');',
                          nargout=0)
        else:
            current_indices_str = [str(x) for x in current_indices]
            str_of_current_indices = ','.join(current_indices_str)
            self.eng.eval('dij.numOfRaysPerBeam = precomp_dij.numOfRaysPerBeam([' + str_of_current_indices + ']);',
                          nargout=0)

        # currently assigned to have same number of bixels and rays, but this may not be true
        self.eng.eval('dij.totalNumOfBixels = sum(dij.numOfRaysPerBeam);', nargout=0)
        self.eng.eval('dij.totalNumOfRays = sum(dij.numOfRaysPerBeam);', nargout=0)

        len_dij = 0
        self.eng.eval('dij.bixelNum = [];', nargout=0)
        self.eng.eval('dij.rayNum = [];', nargout=0)
        self.eng.eval('dij.beamNum = [];', nargout=0)
        self.eng.eval('physical_dose = [];', nargout=0)
        self.eng.eval('dij.couchAngles = [];', nargout=0)
        self.eng.eval('dij.gantryAngles = [];', nargout=0)

        for count, i in enumerate(current_indices):
            offset = self.eng.eval('sum(precomp_dij.numOfRaysPerBeam(1:' + str(i - 1) + '))', nargout=1)
            interval_len = self.eng.eval('precomp_dij.numOfRaysPerBeam(' + str(i) + ');', nargout=1)

            precomp_dij_start = offset + 1
            precomp_dij_end = offset + interval_len
            precomp_dij_indices = str(precomp_dij_start) + ':' + str(precomp_dij_end)

            self.eng.eval('dij.bixelNum = [dij.bixelNum;precomp_dij.bixelNum(' + precomp_dij_indices + ',1)];',
                          nargout=0)
            self.eng.eval('dij.rayNum = [dij.rayNum;precomp_dij.rayNum(' + precomp_dij_indices + ',1)];', nargout=0)
            self.eng.eval('dij.beamNum = [dij.beamNum;' + str(count + 1) + '*ones(' + str(interval_len) + ',1)];',
                          nargout=0)
            self.eng.eval('physical_dose = [physical_dose,precomp_dij.physicalDose{1}(:,' + precomp_dij_indices + ')];',
                          nargout=0)
            self.eng.eval('dij.couchAngles = [dij.couchAngles,precomp_dij.feasible_couch_angle_list(' + str(i) + ')]',
                          nargout=0)
            self.eng.eval(
                'dij.gantryAngles = [dij.gantryAngles,precomp_dij.feasible_gantry_angle_list(' + str(i) + ')]',
                nargout=0)
            len_dij = self.eng.eval('size(dij.bixelNum,1)', nargout=1)

        self.eng.eval('dij.physicalDose = {physical_dose};', nargout=0)
        debug_dict = {}
        debug_dict['numOfRaysPerBeam'] = self.eng.eval('dij.numOfRaysPerBeam', nargout=1)
        debug_dict['totalNumOfBixels'] = self.eng.eval('dij.totalNumOfBixels', nargout=1)
        debug_dict['totalNumOfRays'] = self.eng.eval('dij.totalNumOfRays', nargout=1)
        debug_dict['bixelNum_dims'] = self.eng.eval('size(dij.bixelNum)', nargout=1)
        debug_dict['rayNum_dims'] = self.eng.eval('size(dij.rayNum)', nargout=1)
        debug_dict['beamNum_dims'] = self.eng.eval('size(dij.beamNum)', nargout=1)
        debug_dict['physical_dose_dims'] = self.eng.eval('size(dij.physicalDose{1})', nargout=1)

        if sanity_check:
            bixel_check = self.eng.eval('isequal(dij.bixelNum,temp_dij.bixelNum);', nargout=1)
            ray_check = self.eng.eval('isequal(dij.rayNum,temp_dij.rayNum);', nargout=1)
            beam_check = self.eng.eval('isequal(dij.beamNum,temp_dij.beamNum);', nargout=1)
            dose_check = self.eng.eval(
                'full(sum(dij.physicalDose{1}(:)-temp_dij.physicalDose{1}(:))^2/sum(dij.physicalDose{1}(:))^2);',
                nargout=1)
            gantry_check = self.eng.eval('isequal(dij.gantryAngles,pln.propStf.gantryAngles);', nargout=1)
            couch_check = self.eng.eval('isequal(dij.couchAngles,pln.propStf.couchAngles);', nargout=1)

            assert bixel_check
            assert ray_check
            assert beam_check
            assert dose_check <= 1e-3
            assert gantry_check
            assert couch_check

        return debug_dict

    def precompute_dij(self, feasible_couch_angle_list, feasible_gantry_angle_list):
        '''
        Parameters
        ----------
        feasible_couch_angle_list: list
            list containing all feasible couch angles
        feasible_gantry_angle_list: list
            list containing all feasible gantry angles

        Returns
        -------

        '''
        assert isinstance(feasible_couch_angle_list, list)
        assert isinstance(feasible_gantry_angle_list, list)
        assert len(feasible_couch_angle_list) >= 1
        assert len(feasible_gantry_angle_list) >= 1
        assert len(feasible_couch_angle_list) == len(feasible_gantry_angle_list)

        self.eng.eval('clear precomp_dij;', nargout=0)
        gantry_angle_str = [str(x) for x in feasible_gantry_angle_list]
        couch_angle_str = [str(x) for x in feasible_couch_angle_list]
        str_of_gantry_angles = ','.join(gantry_angle_str)
        str_of_couch_angles = ','.join(couch_angle_str)
        self.eng.eval('pln.propStf.gantryAngles = [' + str_of_gantry_angles + '];', nargout=0)
        self.eng.eval('pln.propStf.couchAngles = [' + str_of_couch_angles + '];', nargout=0)
        self.eng.eval('pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);', nargout=0)
        self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);',
                      nargout=0)
        self.eng.eval('stf = matRad_generateStf(ct,cst,pln);', nargout=0)
        self._run_dose_calc()

        self.eng.eval('precomp_dij = dij;', nargout=0)
        self.eng.eval('precomp_dij.feasible_gantry_angle_list = [' + str_of_gantry_angles + '];', nargout=0)
        self.eng.eval('precomp_dij.feasible_couch_angle_list = [' + str_of_couch_angles + '];', nargout=0)

        return True


class MatRadMetaOpt(MatRadWeightedOpt):
    def reset_pops(self,
                   ctDir,
                   rtStDir,
                   couch_angle_list=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   gantry_angle_list=[0, 40, 80, 120, 160, 200, 240, 280, 320],
                   high_res=False,
                   acceptable_iter=200,
                   acceptable_constr_viol_tol=0.01,
                   ipopt_max_iters=200,
                   vmat=False,
                   additional_vmat_couch_angles = [],
                   additional_vmat_gantry_angles = [],
                   additional_fmo_angles = []):

        if vmat:
            self.reset_weighted_opt_vmat(ctDir,
                                        rtStDir,
                                        high_res=high_res,
                                        acceptable_iter=acceptable_iter,
                                        acceptable_constr_viol_tol=acceptable_constr_viol_tol,
                                        ipopt_max_iters=ipopt_max_iters,
                                        additional_vmat_couch_angles = additional_vmat_couch_angles,
                                        additional_vmat_gantry_angles = additional_vmat_gantry_angles,
                                        additional_fmo_angles = additional_fmo_angles)


        else:
            self.reset_weighted_opt(ctDir,
                               rtStDir,
                               couch_angle_list=couch_angle_list,
                               gantry_angle_list=gantry_angle_list,
                               high_res=high_res,
                               acceptable_iter=acceptable_iter,
                               acceptable_constr_viol_tol=acceptable_constr_viol_tol,
                               ipopt_max_iters=ipopt_max_iters)

        self._run_dose_calc()



    def reset_bao(self,
                   ctDir,
                   rtStDir,
                   meta_optimized_weights,
                   couch_angle_list=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   gantry_angle_list=[0, 40, 80, 120, 160, 200, 240, 280, 320],
                   high_res=False,
                   acceptable_iter=50,
                   acceptable_constr_viol_tol=0.001,
                   ipopt_max_iters=50):

        self.reset_weighted_opt(ctDir,
                                rtStDir,
                                couch_angle_list=couch_angle_list,
                                gantry_angle_list=gantry_angle_list,
                                high_res=high_res,
                                acceptable_iter=acceptable_iter,
                                acceptable_constr_viol_tol=acceptable_constr_viol_tol,
                                ipopt_max_iters=ipopt_max_iters)

        weights = np.array(meta_optimized_weights)
        self.set_obj_weights(weights)

        return True


    def perform_wo_projection(self,weights,structure_min_doses=[],structure_max_doses=[]):
        weights = np.array(weights)
        self.run_wo(weights)

        sf = self.calc_score(structure_min_doses,structure_max_doses)
        results = {}
        results['pareto_optimal_point'] = np.array(weights)
        results['sf'] = sf
        results['dvh'] = self.get_dvh_of_structs()
        results['struct_mean_array'] = self.get_struct_mean_array()
        return results

    def wo_projection_no_score(self,weights):
        weights = np.array(weights)
        self.run_wo(weights)


        results = {}
        results['pareto_optimal_point'] = np.array(weights)
        results['dvh'] = self.get_dvh_of_structs()
        results['struct_mean_array'] = self.get_struct_mean_array()
        return results

    def get_struct_mean_array(self):
        struct_num_list = [self.constr_dict['struct_num_list'][x] for x in self.score_ids]
        struct_mean_list = []

        for struct_num in struct_num_list:
            struct_mean = self.get_mean_of_struct(cst_id=struct_num)
            struct_mean_list.append(struct_mean)

        return np.array(struct_mean_list)

    def binary_search(self, start_coord, end_coord, search_thresh=BINARY_SEARCH_THRESH, bound_vertices_list=None):
        raise NotImplementedError()

    def approx_binary_search(self, start_coord, end_coord, conv_hull_points_list=[], search_thresh=1e-8,
                             bound_vertices_list=None):
        raise NotImplementedError()


    def _eval_point(self, point):
        raise NotImplementedError()

    def calc_score(self, structure_min_doses = [], structure_max_doses = [], base = 2.):
        '''
        Parameters
        ----------
        structure_min_doses: array
            1d array of lowest mean dose values for each structure
        structure_max_doses: array
            1d array of highest mean dose values for each structure
        '''

        if not structure_min_doses:
            num_fractions = int(self.eng.eval('pln.numOfFractions', nargout=1))
            structure_min_doses = np.array([MAX_DOSES[x] for x in self.score_ids])
            structure_min_doses = structure_min_doses/num_fractions
        if not structure_max_doses:
            num_fractions = int(self.eng.eval('pln.numOfFractions', nargout=1))
            structure_max_doses = np.array([MAX_DOSES[x] for x in self.score_ids])*1.1
            structure_max_doses = structure_max_doses/num_fractions

        assert len(structure_min_doses)==len(structure_max_doses)
        assert len(structure_min_doses)==len(self.score_ids)
        score_dict = {}
        # initialize score dictionary
        for i,id in enumerate(self.score_ids):
            preference_tier = float(self.constr_dict['preference_tier_list'][id])
            score_dict[str(preference_tier)] = []
            if preference_tier == 1:
                score_dict[str(preference_tier-1)] = []


        for i,id in enumerate(self.score_ids):
            preference_tier = float(self.constr_dict['preference_tier_list'][id])
            struct_num = self.constr_dict['struct_num_list'][id]
            struct_min_dose = structure_min_doses[i]
            struct_max_dose = structure_max_doses[i]

            if preference_tier == 1: # preference 1 assigned to ptvs
                ci_val, hi_val = self.get_ci_and_hi(ptv_struct_num=struct_num)
                r50_val, r90_val = self.get_r50_and_r90(ptv_struct_num=struct_num)
                # --- PTV Terms ---
                ci_term = (CI_RANGE[-1] - ci_val) / (CI_RANGE[-1] - CI_RANGE[0])
                hi_term = (hi_val - HI_RANGE[0]) / (HI_RANGE[-1] - HI_RANGE[0])
                r50_term = (r50_val - R50_RANGE[0]) / (R50_RANGE[-1] - R50_RANGE[0])
                r90_term = (r90_val - R90_RANGE[0]) / (R90_RANGE[-1] - R90_RANGE[0])
                ptv_term = list(np.clip(np.array([hi_term]),0,1))
                dose_spill_term = list(np.clip(np.array([ci_term]),0,1))
                score_dict[str(preference_tier-1)] += ptv_term
                score_dict[str(preference_tier)] += dose_spill_term

            else: # preference >1 assigned to oars
                oar_mean = self.get_mean_of_struct(cst_id=struct_num)
                oar_mean = (oar_mean - struct_min_dose)/(struct_max_dose - struct_min_dose)
                oar_term = np.clip(oar_mean,0,1)
                score_dict[str(preference_tier)].append(oar_term)


        print('Score dict:',score_dict)

        numer = []
        denom = []
        for preference_tier,val_list in score_dict.items():
            # compute average of values in each tier
            tier_score = np.mean(val_list)
            tier_weight = base**(-float(preference_tier)+1)

            numer.append(tier_weight*tier_score)
            denom.append(tier_weight)

        score = np.sum(numer)/np.sum(denom)
        return score


    def find_nearest_dose_given_volume_point(self, volume_array, vol, dose_array):
        volume_array = np.asarray(volume_array)
        idx = (np.abs(volume_array - vol)).argmin()
        return dose_array[idx]

    def get_dvh_of_structs(self):
        self.set_ids()
        cst_ids = [self.constr_dict['struct_num_list'][x] for x in self.dvh_ids]

        self.eng.eval('cst_dvh = cst('+str(cst_ids)+',:);', nargout=0)
        self.eng.eval('[dvh,dvh_qi] = matRad_indicatorWrapper(cst_dvh,pln,resultGUI);', nargout=0)

        x = self.eng.eval('dvh(' + str(1) + ').doseGrid', nargout=1)
        dvh = np.zeros((len(cst_ids),len(np.squeeze(x)),2))

        for ind, cst_id in enumerate(cst_ids):
            x = self.eng.eval('dvh(' + str(ind + 1) + ').doseGrid', nargout=1)
            y = self.eng.eval('dvh(' + str(ind + 1) + ').volumePoints', nargout=1)

            x = np.squeeze(np.array(x))
            y = np.squeeze(np.array(y))
            assert len(x)==len(y), 'DVH x and y dimension mismatch'
            dvh[ind, :, 0] = x
            dvh[ind, :, 1] = y

        assert np.max(np.isnan(dvh)) == 0, 'DVH has NaN values'
        return dvh

    def rescale_dvh(self,in_dvh, rescale_factor = None, ptv_dvh_id = None):

        if ptv_dvh_id is None:
            names_of_dvh_structs = [self.constr_dict['name_list'][x] for x in self.dvh_ids]
            target_id = names_of_dvh_structs.index('ptv')
        else:
            target_id = ptv_dvh_id


        if rescale_factor is None:
            num_of_fractions = int(self.eng.eval('pln.numOfFractions', nargout=1))
            found_dose = self.find_nearest_dose_given_volume_point(in_dvh[target_id, :, 1], 95,
                                                                   in_dvh[target_id, :, 0] * num_of_fractions)
            rescale_factor = NORMALIZED_DOSE / found_dose
            print('Target id:', target_id)
            print('Found dose:', found_dose)

        print('Normalized dose:', NORMALIZED_DOSE)
        print('Prescription dose:', PRESCRIBED_DOSE)
        print('Rescaling factor:', rescale_factor)

        in_dvh = np.squeeze(in_dvh)
        for i in range(in_dvh.shape[0]):
            in_dvh[i, :, 0] = in_dvh[i, :, 0] * rescale_factor


        self.rescale_factor = rescale_factor
        return in_dvh

    def get_structure_means(self):
        mean_vals = self.eng.eval('[dvh_qi(:).mean]', nargout = 1)
        mean_vals = np.array(mean_vals)*self.rescale_factor
        return mean_vals

    def return_inf_score(self,weights):
        weights = np.array(weights)

        sf = np.inf
        results = {}
        results['pareto_optimal_point'] = weights
        results['sf'] = sf
        results['dvh'] = None
        results['struct_mean_array'] = None
        return results

    def return_inf_obj(self,couch_angle, gantry_angle, current_indices = None, debug = True):
        val = 1e6
        debug_dict = {}

        if debug == True:
            return val, debug_dict
        else:
            return val