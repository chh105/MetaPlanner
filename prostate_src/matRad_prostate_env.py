
import gym
import matlab.engine
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.special as sc
import textdistance
import copy
import psutil
from gym.utils import seeding
from scipy import interpolate
import time
import pandas as pd
from scipy.stats import wilcoxon

import ray


os.environ['matRad'] = '../matRad-master2021/'

NUM_STRUCTURES = 6
DVH_SHAPE = (NUM_STRUCTURES,1000,2)
DVH_HISTORY_LEN = 1
PRESCRIBED_DOSE = 74
DOSE_MAX = np.round(1.05*PRESCRIBED_DOSE)
MIN_PTV_DOSE = np.round(0.975*PRESCRIBED_DOSE)
MAX_PTV_DOSE = np.round(1.05*PRESCRIBED_DOSE)
NORMALIZED_DOSE = 78
SEED_POINT = [DOSE_MAX,DOSE_MAX,DOSE_MAX,DOSE_MAX,50]

CPU_USAGE_THRESH = 8 # threshold in percent
CPU_USAGE_INTERVAL = np.linspace(4,10,16)#[2,5] # check usage over __ sec

BINARY_SEARCH_THRESH = 0.02


class MatRad(gym.Env):    

    def __init__(self, env_config = None):

        self.current_step = 0
        
        self.seed()

        self.constr_dict = {
            'name_list': ['rectum'] * 7 + ['bladder'] * 5 + ['fh r']*2 + ['fh l']*2 + ['ptv'] * 3 + ['body']*5,
            'obj_type_list': [9] + [7] + [8] * 5 + [9] + [7] + [8] * 3 + [9] + [7] + [9] + [7] + [4] + [7] + [8] + [9] + [7] + [8] * 3,
            'param_list': [[1,0,DOSE_MAX], [0, DOSE_MAX, 1], [30, 0, 80], [47, 0, 55], [65, 0, 40], [70, 0, 25], [75, 0, 10],
                           [1,0,DOSE_MAX], [0, DOSE_MAX, 1], [30, 0, 80], [47, 0, 55], [70, 0, 30],
                           [1,0,DOSE_MAX], [0, 50, 1],
                           [1,0,DOSE_MAX], [0, 50, 1],
                           [PRESCRIBED_DOSE], [MIN_PTV_DOSE, MAX_PTV_DOSE, 1], [PRESCRIBED_DOSE, 95, 100],
                           [5,0,50],[0, DOSE_MAX, 1],[8, 0, 20],[15, 0, 15],[25, 0, 10]],
            'obj_num_list': list(np.arange(7)+1)+\
                            list(np.arange(5)+1)+\
                            list(np.arange(2)+1)+\
                            list(np.arange(2)+1)+\
                            list(np.arange(3)+1)+\
                            list(np.arange(5)+1),
            'priority_list': [2]*7+[2]*5+[3]*2+[3]*2+[1]*3+[4]*5,
            'obj_weight_list': [None]*7+[None]*5+[None]*2+[None]*2+[1]+[None]*2+[None]*5,
            'eud_weights': [1]*7+[1]*5+[0.15]*2+[0.15]*2+[None]*3+[5]*5,
            # 'tier_list': [2]*7+[2]*5+[4]*2+[4]*2+[1]*3+[3]*5,
        }
        self.adjusted_constraint_ids = [0,7,12,14,19]
        self.max_dose = SEED_POINT
        self.dvh_history = np.zeros((DVH_HISTORY_LEN,)+DVH_SHAPE)
        self.done = False
        self.prev_sf = 1.0
        # self._set_cpu_affinity([0,1,2,3,4,5,6,7])
        # self._set_cpu_affinity([8,9,10,11,12,13,14,15])
        self._set_cpu_affinity()

    def _set_cpu_affinity(self,available_cpus = list(np.arange(os.cpu_count()))):

        # '''set available cpus'''
        # print("Number of CPUs:", os.cpu_count()) 

        viable_cpus = []
        usage_interv = CPU_USAGE_INTERVAL
        usage_thresh = CPU_USAGE_THRESH
        while len(viable_cpus)<1 and usage_thresh < 100:
            interv = usage_interv[int(self.np_random.uniform(0,len(usage_interv)))]
            cpu_usages = np.array(psutil.cpu_percent(interval=interv,percpu=True))
            cpu_ids = np.arange(len(cpu_usages))[:]
            cpu_usages = cpu_usages[:]
            viable_cpus = cpu_ids[cpu_usages<usage_thresh]
            usage_thresh += 30

        viable_cpus = [x for x in viable_cpus if x in available_cpus]

        if len(viable_cpus)==0:
            idx = int(self.np_random.uniform(0,len(available_cpus)))
            affinity_mask = {available_cpus[idx]}
        else:
            idx = int(self.np_random.uniform(0,len(viable_cpus)))
            cpu_id = int(viable_cpus[idx])
            affinity_mask = {cpu_id} 
        pid = 0
        os.sched_setaffinity(pid, affinity_mask) 
        affinity = os.sched_getaffinity(pid) 
          
        # Print the result 
        print("Now, process is eligible to run on:", affinity) 

    def _get_dvh(self, get_names = True, compress_dvh = True):
        
        dvh = np.zeros(DVH_SHAPE)
        names = []
        self.unique_structs = list(sorted(set(self.constr_dict['struct_num_list']), key = self.constr_dict['struct_num_list'].index))
        assert NUM_STRUCTURES==len(self.unique_structs)
        if compress_dvh:
            self.eng.eval('struct_ids = '+str(self.unique_structs)+';',nargout=0)
            self.eng.eval('cst_temp = cst(struct_ids,:);',nargout=0)
            self.eng.eval('[dvh,qi] = matRad_indicatorWrapper(cst_temp,pln,resultGUI);',nargout=0)

            count = 0
            for ind, struct_id in enumerate(self.unique_structs):
                x = self.eng.eval('dvh('+str(ind+1)+').doseGrid',nargout=1)
                y = self.eng.eval('dvh('+str(ind+1)+').volumePoints',nargout=1)

                x = np.squeeze(np.array(x))[:DVH_SHAPE[1]]
                y = np.squeeze(np.array(y))[:DVH_SHAPE[1]]
                dvh[count,:,0] = x
                dvh[count,:,1] = y
                if get_names:
                    name = self.eng.eval('qi(' + str(ind+1) + ').name', nargout=1)
                    names.append(name)
                # plt.plot(dvh[count,:,0],dvh[count,:,1])
                count += 1
        else:
            if get_names:
                self.eng.eval('[dvh,qi] = matRad_indicatorWrapper(cst,pln,resultGUI);',nargout=0)
            else:
                self.eng.eval('calc_dvh_script;',nargout=0)

            count = 0
            for ind in self.unique_structs:
                x = self.eng.eval('dvh('+str(ind)+').doseGrid',nargout=1)
                y = self.eng.eval('dvh('+str(ind)+').volumePoints',nargout=1)

                x = np.squeeze(np.array(x))[:DVH_SHAPE[1]]
                y = np.squeeze(np.array(y))[:DVH_SHAPE[1]]
                dvh[count,:,0] = x
                dvh[count,:,1] = y
                if get_names:
                    name = self.eng.eval('qi(' + str(ind) + ').name', nargout=1)
                    names.append(name)
                # plt.plot(dvh[count,:,0],dvh[count,:,1])
                count += 1

        # plt.show()
        assert np.max(np.isnan(dvh))==0, 'DVH has NaN values'
        print('Structure names: ',names)
        return dvh, names

    def _import_dicom(self, ctDir, rtStDir, rtDoseDir = '', rtPlanDir = '',high_res = False):
        self.eng.eval('ctDir = "'+ctDir+'";', nargout=0)
        self.eng.eval('rtStDir = "'+rtStDir+'";', nargout=0)
        if not rtDoseDir:
            self.eng.eval('rtDoseDir = "'+rtDoseDir+'";', nargout=0)
        if not rtPlanDir:
            self.eng.eval('rtPlanDir = "'+rtPlanDir+'";', nargout=0)

        mm_res = 5
        bixel_width = 7
        if high_res:
            mm_res = 3.5
            bixel_width = 5

        self.eng.eval('[ct, cst, pln] = import_dicom_without_stf( ctDir, rtStDir, '+str(mm_res)+', '+str(bixel_width)+' );',nargout=0)

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


    def get_obj_function_value(self, couch_angle, gantry_angle, current_indices = None, debug = True):
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
        self.eng.eval('pln.propStf.gantryAngles = ['+str_of_gantry_angles+'];', nargout=0)
        self.eng.eval('pln.propStf.couchAngles = ['+str_of_couch_angles+'];', nargout=0)

        self.eng.eval('pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);', nargout = 0)
        self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);', nargout=0)

        self.eng.eval('stf = matRad_generateStf(ct,cst,pln);',nargout = 0)
        if current_indices == None:
            self._run_dose_calc()
        else:
            debug_dict = self.get_precomp_dij(current_indices)

        try:
            self._run_optimizer()
            val = self.eng.eval('resultGUI.usedOptimizer.finalObjectiveFunctionValue',nargout = 1)
        except:
            crash_log = {}
            crash_log['couch_angles'] = couch_angle
            crash_log['gantry_angles'] = gantry_angle
            np.save('crash_log.npy',crash_log)



        # print('Couch angles: ',couch_angle)
        # print('Gantry angles: ',gantry_angle)
        # print('Objective function value:',val)

        if debug == True:
            return val, debug_dict
        else:
            return val

    def get_precomp_dij(self, current_indices, sanity_check = False):
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
            self.eng.eval('temp_dij = dij;', nargout = 0)
        

        assert isinstance(current_indices, list)
        assert len(current_indices) >= 1

        #convert python indices to matlab
        current_indices = list(np.array(current_indices)+1)

        self.eng.eval('clear dij;', nargout = 0)
        self.eng.eval('clear physical_dose;', nargout = 0)

        self.eng.eval('dij.doseGrid = precomp_dij.doseGrid;', nargout = 0)
        self.eng.eval('dij.ctGrid = precomp_dij.ctGrid;', nargout = 0)
        self.eng.eval('dij.numOfScenarios = precomp_dij.numOfScenarios;', nargout = 0)
        self.eng.eval('dij.numOfBeams = '+str(len(current_indices))+';', nargout = 0)

        if len(current_indices) == 1:
            self.eng.eval('dij.numOfRaysPerBeam = precomp_dij.numOfRaysPerBeam('+str(current_indices[0])+');',
                          nargout = 0)
        else:
            current_indices_str = [str(x) for x in current_indices]
            str_of_current_indices = ','.join(current_indices_str)
            self.eng.eval('dij.numOfRaysPerBeam = precomp_dij.numOfRaysPerBeam(['+str_of_current_indices+']);',
                          nargout = 0)

        # currently assigned to have same number of bixels and rays, but this may not be true
        self.eng.eval('dij.totalNumOfBixels = sum(dij.numOfRaysPerBeam);',nargout = 0)
        self.eng.eval('dij.totalNumOfRays = sum(dij.numOfRaysPerBeam);',nargout = 0)

        len_dij = 0
        self.eng.eval('dij.bixelNum = [];',nargout = 0)
        self.eng.eval('dij.rayNum = [];',nargout = 0)
        self.eng.eval('dij.beamNum = [];',nargout = 0)
        self.eng.eval('physical_dose = [];',nargout = 0)
        self.eng.eval('dij.couchAngles = [];',nargout = 0)
        self.eng.eval('dij.gantryAngles = [];',nargout = 0)
            
        for count, i in enumerate(current_indices):
            offset = self.eng.eval('sum(precomp_dij.numOfRaysPerBeam(1:'+str(i-1)+'))',nargout = 1) 
            interval_len = self.eng.eval('precomp_dij.numOfRaysPerBeam('+str(i)+');',nargout = 1)

            precomp_dij_start = offset+1
            precomp_dij_end = offset+interval_len
            precomp_dij_indices = str(precomp_dij_start)+':'+str(precomp_dij_end)

            self.eng.eval('dij.bixelNum = [dij.bixelNum;precomp_dij.bixelNum('+precomp_dij_indices+',1)];',nargout = 0)
            self.eng.eval('dij.rayNum = [dij.rayNum;precomp_dij.rayNum('+precomp_dij_indices+',1)];',nargout = 0)
            self.eng.eval('dij.beamNum = [dij.beamNum;'+str(count+1)+'*ones('+str(interval_len)+',1)];',nargout = 0)
            self.eng.eval('physical_dose = [physical_dose,precomp_dij.physicalDose{1}(:,'+precomp_dij_indices+')];',nargout = 0)
            self.eng.eval('dij.couchAngles = [dij.couchAngles,precomp_dij.feasible_couch_angle_list('+str(i)+')]',nargout = 0)
            self.eng.eval('dij.gantryAngles = [dij.gantryAngles,precomp_dij.feasible_gantry_angle_list('+str(i)+')]',nargout = 0)
            len_dij = self.eng.eval('size(dij.bixelNum,1)',nargout = 1)

        self.eng.eval('dij.physicalDose = {physical_dose};',nargout = 0)
        debug_dict = {}
        debug_dict['numOfRaysPerBeam'] = self.eng.eval('dij.numOfRaysPerBeam',nargout = 1)
        debug_dict['totalNumOfBixels'] = self.eng.eval('dij.totalNumOfBixels',nargout = 1)
        debug_dict['totalNumOfRays'] = self.eng.eval('dij.totalNumOfRays',nargout = 1)
        debug_dict['bixelNum_dims'] = self.eng.eval('size(dij.bixelNum)',nargout = 1)
        debug_dict['rayNum_dims'] = self.eng.eval('size(dij.rayNum)',nargout = 1)
        debug_dict['beamNum_dims'] = self.eng.eval('size(dij.beamNum)',nargout = 1)
        debug_dict['physical_dose_dims'] = self.eng.eval('size(dij.physicalDose{1})',nargout = 1)

        if sanity_check:
            bixel_check = self.eng.eval('isequal(dij.bixelNum,temp_dij.bixelNum);', nargout = 1)
            ray_check = self.eng.eval('isequal(dij.rayNum,temp_dij.rayNum);', nargout = 1)
            beam_check = self.eng.eval('isequal(dij.beamNum,temp_dij.beamNum);', nargout = 1)
            dose_check = self.eng.eval('full(sum(dij.physicalDose{1}(:)-temp_dij.physicalDose{1}(:))^2/sum(dij.physicalDose{1}(:))^2);', nargout = 1)
            gantry_check = self.eng.eval('isequal(dij.gantryAngles,pln.propStf.gantryAngles);', nargout = 1)
            couch_check = self.eng.eval('isequal(dij.couchAngles,pln.propStf.couchAngles);', nargout = 1)
           
            assert bixel_check
            assert ray_check
            assert beam_check
            assert dose_check <= 1e-3
            assert gantry_check
            assert couch_check

        return debug_dict

    def precompute_dij(self,feasible_couch_angle_list,feasible_gantry_angle_list):
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

        self.eng.eval('clear precomp_dij;',nargout = 0)
        gantry_angle_str = [str(x) for x in feasible_gantry_angle_list]
        couch_angle_str = [str(x) for x in feasible_couch_angle_list]
        str_of_gantry_angles = ','.join(gantry_angle_str)
        str_of_couch_angles = ','.join(couch_angle_str)
        self.eng.eval('pln.propStf.gantryAngles = ['+str_of_gantry_angles+'];', nargout=0)
        self.eng.eval('pln.propStf.couchAngles = ['+str_of_couch_angles+'];', nargout=0)
        self.eng.eval('pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);', nargout = 0)
        self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);', nargout=0)
        self.eng.eval('stf = matRad_generateStf(ct,cst,pln);',nargout = 0)
        self._run_dose_calc()

        self.eng.eval('precomp_dij = dij;',nargout = 0)
        self.eng.eval('precomp_dij.feasible_gantry_angle_list = ['+str_of_gantry_angles+'];',nargout = 0)
        self.eng.eval('precomp_dij.feasible_couch_angle_list = ['+str_of_couch_angles+'];',nargout = 0)

        return True

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
            '0': 'struct(DoseObjectives.matRad_EUD)',
            '1': 'struct(DoseObjectives.matRad_MaxDVH)',
            '2': 'struct(DoseObjectives.matRad_MeanDose)',
            '3': 'struct(DoseObjectives.matRad_MinDVH)',
            '4': 'struct(DoseObjectives.matRad_SquaredDeviation)',
            '5': 'struct(DoseObjectives.matRad_SquaredOverdosing)',
            '6': 'struct(DoseObjectives.matRad_SquaredUnderdosing)',
            '7': 'struct(DoseConstraints.matRad_MinMaxDose)',
            '8': 'struct(DoseConstraints.matRad_MinMaxDVH)',
            '9': 'struct(DoseConstraints.matRad_MinMaxEUD)',
            '10': 'struct(DoseConstraints.matRad_MinMaxMeanDose)'
        }

        self.eng.eval('cstIndex = '+str(struct_num)+';', nargout=0)
        # if is_first_obj_bool:
        #     self.eng.eval('cst{cstIndex, 6}{1} = '+switcher[str(obj_type)], nargout=0)
        # else:
        #     self.eng.eval('cst{cstIndex, 6}(end+1) = '+switcher[str(obj_type)], nargout=0)
        self.eng.eval('cst{cstIndex, 6}{end+1} = ' + switcher[str(obj_type)]+';', nargout=0)

    def _clear_all_obj(self):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        num_struct = self.eng.eval('size(cst,1);',nargout = 1)
        print('Number of structures: ',num_struct)
        for i in range(1,int(num_struct)+1):
            self.eng.eval('cst{'+str(i)+',6} = [];', nargout=0)

    def _change_obj_params(self,struct_num = None, obj_num = None, param_val_list = []):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        # example: cst{2,6}{1}.parameters = {47,55}
        self.eng.eval('cst{'+str(struct_num)+',6}{'+str(obj_num)+'}.parameters = {'
                      +','.join(str(x) for x in param_val_list)+'};', nargout=0)

    def _change_obj_weight(self,struct_num = None, obj_num = None, penalty = None):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        # example: cst{2,6}{1}.parameters = {47,55}
        self.eng.eval('cst{'+str(struct_num)+',6}{'+str(obj_num)+'}.penalty = '+str(penalty)+';', nargout=0)

    def _get_obj_weight(self,struct_num = None, obj_num = None):
        '''
        Assumes structure cell array is stored in 'cst'
        :return:
        '''
        # example: cst{2,6}{1}.parameters = {47,55}
        weight = self.eng.eval('cst{'+str(struct_num)+',6}{'+str(obj_num)+'}.penalty;', nargout=1)
        return weight

    def _change_structure_priority(self, struct_num = None, priority = None):
        self.eng.eval('cst{'+str(struct_num)+',5}.Priority = '+str(priority)+';', nargout=0)

    def _set_all_structure_priorities(self, priority = None):
        n = int(self.eng.eval('size(cst,1);',nargout=1))
        for i in range(n):
            self.eng.eval('cst{'+str(i+1)+',5}.Priority = '+str(priority)+';', nargout=0)

    def _rem_unconstr_struct(self):
        '''
        Assumes structure cell array is stored in 'cst'. Finds rows with empty constraints and removes them from 'cst'
        :return:
        '''
        num_struct = self.eng.eval('size(cst,1);',nargout = 1)
        num_struct = int(num_struct)
        ind_list = []
        for i in range(1,num_struct+1):
            flag = self.eng.eval('isempty(cst{'+str(i)+',6});', nargout=1)
            flag = int(flag)
            if flag:
                ind_list.append(i)

        self.eng.eval('cst('+str(ind_list)+',:)=[];', nargout=0)

        new_num_struct = self.eng.eval('size(cst,1);',nargout = 1)
        print('Number of structures: ',new_num_struct)

    def _set_dose_res(self):
        self.eng.eval('set_dose_res',nargout = 0)

    def _run_dose_calc(self):

        self.eng.eval('pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);',nargout=0)
        self.eng.eval('run_dose_calc_gym;', nargout=0)

    def _run_optimizer(self):
        self.eng.eval('run_optimizer_gym;', nargout=0)

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

    def _rerun(self, checklist = None):

        self._clear_all_obj()

        if checklist is None:
            checklist = copy.deepcopy(self.adjusted_constraint_ids) + [1,8,13,15,20]
        
        for c in range(len(self.constr_dict['name_list'])):


            # if self.constr_dict['priority_list'][c]!=1 and self.constr_dict['obj_type_list'][c]!=9 and c != self.adjusted_constraint_ids[-1]+1: # and self.constr_dict['name_list'][c]!='body':
            #     continue

            if c not in checklist and self.constr_dict['priority_list'][c]!=1:
                continue

            self._add_obj(struct_num=self.constr_dict['struct_num_list'][c],
                          obj_type=self.constr_dict['obj_type_list'][c])
            self._change_obj_params(struct_num=self.constr_dict['struct_num_list'][c],
                                    obj_num=self.constr_dict['obj_num_list'][c],
                                    param_val_list=self.constr_dict['param_list'][c])
            self._change_structure_priority(struct_num=self.constr_dict['struct_num_list'][c],
                                            priority=self.constr_dict['priority_list'][c])


            if self.constr_dict['priority_list'][c]!=1 and self.constr_dict['obj_type_list'][c]==5:
                weight = 1
                self._change_obj_weight(struct_num=self.constr_dict['struct_num_list'][c],
                                        obj_num=self.constr_dict['obj_num_list'][c],
                                        penalty=weight)

    def _construct_objectives_and_constraints(self, checklist = None):

        self._clear_all_obj()

        if checklist is None:
            checklist = copy.deepcopy(self.adjusted_constraint_ids) + [1,8,13,15,20]
        
        for c in range(len(self.constr_dict['name_list'])):

            if c not in checklist and self.constr_dict['priority_list'][c]!=1:
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
    def _set_max_iters(self,max_iters = 400):

        self.eng.eval('global max_iter_limit;',nargout=0)
        self.eng.eval('max_iter_limit = '+str(max_iters)+';',nargout=0)

    def _set_ipopt_acceptable_iter(self,acceptable_iter = 3):

        self.eng.eval('global acceptable_iter_limit;',nargout=0)
        self.eng.eval('acceptable_iter_limit = '+str(acceptable_iter)+';',nargout=0)

    def _set_ipopt_acceptable_constr_viol_tol(self,acceptable_constr_viol_tol = 0.001):

        self.eng.eval('global acceptable_constr_viol_tol_limit;',nargout=0)
        self.eng.eval('acceptable_constr_viol_tol_limit = '+str(acceptable_constr_viol_tol)+';',nargout=0)


    def reset_trajectory_searcher(self,
              ctDir,
              rtStDir,
              high_res = False,
              acceptable_iter = 3,
              acceptable_constr_viol_tol = 0.001,
              ipopt_max_iters = 50):

        '''
        configs
        '''
        # # scoring weights eud obj 5:1
        # self.constr_dict = {
        #     'name_list': ['rectum'] + ['bladder'] + ['fh r'] + ['fh l'] + ['ptv'] + ['body'],
        #     'obj_type_list': [0] + [0] + [0] + [0] + [0] + [0],
        #     'param_list': [[0,1],[0,1],[0,1],[0,1],[PRESCRIBED_DOSE,-20],[0,5]],
        #     'obj_num_list': list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1),
        #     'priority_list': [2]+[2]+[3]+[3]+[1]+[4],
        #     'obj_weight_list': [0.2]+[0.2]+[0.03]+[0.03]+[1]+[1]
        # }

        # # scoring weights eud obj 1:1
        # self.constr_dict = {
        #     'name_list': ['rectum'] + ['bladder'] + ['fh r'] + ['fh l'] + ['ptv'] + ['body'],
        #     'obj_type_list': [0] + [0] + [0] + [0] + [0] + [0],
        #     'param_list': [[0, 1], [0, 1], [0, 1], [0, 1], [PRESCRIBED_DOSE, -20], [0, 5]],
        #     'obj_num_list': list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1),
        #     'priority_list': [2] + [2] + [3] + [3] + [1] + [4],
        #     'obj_weight_list': [1] + [1] + [0.15] + [0.15] + [1] + [1]
        # }

        # scoring weights eud obj ptv constr
        self.constr_dict = {
            'name_list': ['rectum'] + ['bladder'] + ['fh r'] + ['fh l'] + ['ptv'],
            'obj_type_list': [0] + [0] + [0] + [0] + [8],
            'param_list': [[0, 8], [0, 8], [0, 12], [0, 12], [PRESCRIBED_DOSE, 95, 100]],
            'obj_num_list': list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1) + \
                            list(np.arange(1) + 1),
            'priority_list': [2] + [2] + [2] + [2] + [1],
            'obj_weight_list': [1] + [1] + [1] + [1] + [None] + [1]
        }



        # # ptv only oar constrained
        # self.constr_dict = {
        #     'name_list': ['rectum'] + ['bladder'] + ['fh r'] + ['fh l'] + ['ptv'] + ['body'],
        #     'obj_type_list': [9] + [9] + [9] + [9] + [4] + [9],
        #     'param_list': [[1, 0, 2.5026*8.73126],
        #                    [1, 0, 1.7591*8.73126],
        #                    [1, 0, 1*8.73126],
        #                    [1, 0, 1.0020*8.73126],
        #                    [PRESCRIBED_DOSE],
        #                    [5, 0, 2.5884*8.73126]],
        #     'obj_num_list': list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1)+\
        #                     list(np.arange(1)+1),
        #     'priority_list': [2]+[2]+[3]+[3]+[1]+[4],
        #     'obj_weight_list': [None]+[None]+[None]+[None]+[1]+[None]
        # }

        # # ptv only oar constrained
        # # tg 166: np.array([44,44,16.39667,17.68667,22.622])/min(np.array([44,44,16.39667,17.68667,22.622]))
        # self.constr_dict = {
        #     'name_list': ['rectum'] + ['bladder'] + ['fh r'] + ['fh l'] + ['ptv'],
        #     'obj_type_list': [9] + [9] + [9] + [9] + [4],
        #     'param_list': [[8, 0, 44], [8, 0, 44], [12, 0, 16.39667], [12, 0, 17.68667], [PRESCRIBED_DOSE]],
        #     'obj_num_list': list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1) + \
        #                     list(np.arange(1) + 1),
        #     'priority_list': [2] + [2] + [3] + [3] + [1],
        #     'obj_weight_list': [None] + [None] + [None] + [None] + [1]
        # }
        '''
        end of various configs
        '''
        checklist = list(np.arange(len(self.constr_dict['name_list'])))


        self.ctDir = ctDir
        self.rtStDir = rtStDir
        self.eng = matlab.engine.start_matlab()
        # self.eng.eval('cd ' + os.environ['matRad'],nargout=0)
        self.eng.eval('addpath(genpath("'+os.environ['matRad']+'"));')
        # self.eng.eval('set(0,"DefaultFigureVisible","off")',nargout=0)
        self.eng.eval('clear all; close all; clc;',nargout=0)
        self._set_ipopt_acceptable_iter(acceptable_iter = acceptable_iter)
        self._set_ipopt_acceptable_constr_viol_tol(acceptable_constr_viol_tol = acceptable_constr_viol_tol)
        self._set_max_iters(max_iters = ipopt_max_iters)
        self.dvh_history = np.zeros((DVH_HISTORY_LEN,)+DVH_SHAPE)
        self.done = False
        self.prev_sf = 1.0
        self.current_step = 0

        self.curr_param_list = copy.deepcopy(self.constr_dict['param_list'])



        self._import_dicom(ctDir, rtStDir, high_res = high_res)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')

        '''Remove unwanted structures'''
        unique_structs = list(sorted(set(struct_num_list), key = struct_num_list.index))
        
        self.eng.eval('struct_ids = '+str(unique_structs)+';',nargout=0)
        self.eng.eval('cst_small = cst(struct_ids,:);',nargout=0)
        self.eng.eval('cst = cst_small;',nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('New Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')
        ''''''
        self.constr_dict['struct_num_list'] = struct_num_list
        assert len(self.constr_dict['struct_num_list']) == len(self.constr_dict['name_list']), 'Mismatch struct_num_list and name_list'
        self._set_all_structure_priorities(priority=np.max(np.array(self.constr_dict['priority_list']))+1)
        self._construct_objectives_and_constraints(checklist = checklist)

        return True
    
    def reset_pops(self,
              ctDir,
              rtStDir,
              couch_angle_list = [0,0,0,0,0,0,0,0,0], 
              gantry_angle_list = [0,40,80,120,160,200,240,280,320],
              high_res = False,
              acceptable_iter = 1,
              acceptable_constr_viol_tol=0.01,
              ipopt_max_iters = 400):


        self.ctDir = ctDir
        self.rtStDir = rtStDir
        self.eng = matlab.engine.start_matlab()
        # self.eng.eval('cd ' + os.environ['matRad'],nargout=0)
        self.eng.eval('addpath(genpath("'+os.environ['matRad']+'"));')
        # self.eng.eval('set(0,"DefaultFigureVisible","off")',nargout=0)
        self.eng.eval('clear all; close all; clc;',nargout=0)
        self._set_ipopt_acceptable_iter(acceptable_iter = acceptable_iter)
        self._set_ipopt_acceptable_constr_viol_tol(acceptable_constr_viol_tol=acceptable_constr_viol_tol)
        self._set_max_iters(max_iters = ipopt_max_iters)
        self.dvh_history = np.zeros((DVH_HISTORY_LEN,)+DVH_SHAPE)
        self.done = False
        self.prev_sf = 1.0
        self.current_step = 0

        self.curr_param_list = copy.deepcopy(self.constr_dict['param_list'])


        self._import_dicom_with_angles(ctDir, 
                                       rtStDir,  
                                       couch_angle_list = couch_angle_list,
                                       gantry_angle_list = gantry_angle_list,
                                       high_res = high_res)

        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')

        '''Remove unwatned structures'''
        unique_structs = list(sorted(set(struct_num_list), key = struct_num_list.index))
        assert NUM_STRUCTURES==len(unique_structs)
        
        self.eng.eval('struct_ids = '+str(unique_structs)+';',nargout=0)
        self.eng.eval('cst_small = cst(struct_ids,:);',nargout=0)
        self.eng.eval('cst = cst_small;',nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('New Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')
        ''''''
        self.constr_dict['struct_num_list'] = struct_num_list
        assert len(self.constr_dict['struct_num_list']) == len(self.constr_dict['name_list']), 'Mismatch struct_num_list and name_list'
        self._set_all_structure_priorities(priority=np.max(np.array(self.constr_dict['priority_list']))+1)
        self._run_dose_calc()        
        self._rerun()

    def render(self,visual=False):
        print('Prev sf: ',self.prev_sf)
        if visual:
            self.eng.eval('matRadGUI',nargout=0)

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _eval_point(self, point):
        
        point = np.round(point,decimals=2)
        adjusted_constraint_ids = copy.deepcopy(self.adjusted_constraint_ids)
        max_dose = copy.deepcopy(self.max_dose)
        # try:
        #     print('Struct_num_list:',self.constr_dict['struct_num_list'])
        # except:
        #     self.reset(self.ctDir, self.rtStDir)
        for constr_idx, c in enumerate(adjusted_constraint_ids):
            dose_val = point[constr_idx]
            self.curr_param_list[c][-1] = dose_val
            self._change_obj_params(struct_num=self.constr_dict['struct_num_list'][c],
                                    obj_num=self.constr_dict['obj_num_list'][c],
                                    param_val_list=self.curr_param_list[c])
            print('Name: ',self.constr_dict['name_list'][c],'Number: ',self.constr_dict['struct_num_list'][c],'CURRENT DOSE VALUE: ',self.curr_param_list[c][-1])
            
        try:
            self._run_optimizer()
        except:
            self.eng.eval('disp(cst{1,6})',nargout=0)
            raise ValueError('Problem running optimizer')
        dvh, names = self._get_dvh()
        return dvh

    def binary_search(self, start_coord, end_coord, search_thresh = BINARY_SEARCH_THRESH, bound_vertices_list = None):
        

        top_point = start_coord
        bot_point = end_coord
        mid_point = (top_point + bot_point)/2
        dist = np.sqrt(np.sum((top_point/self.max_dose - bot_point/self.max_dose)**2))
        recheck_count = 0
        while dist > search_thresh:  
            dvh = self._eval_point(mid_point)
            self.dvh_history[:-1,...] = self.dvh_history[1:,...]
            self.dvh_history[-1,...] = dvh
            # sf = self._calc_sf(dvh_hist=self.dvh_history)
            sf = self._calc_eud_score(point_coords = mid_point, 
                                       bound_vertices_list = bound_vertices_list)
        
            print('sf:',sf)
            '''Check if midpoint is infeasible and update search range'''
            if sf < 1:
                top_point = mid_point
                mid_point = (top_point + bot_point)/2
                recheck_count = 0
            # elif recheck_count == 0:
            #     mid_point = (4*mid_point + bot_point)/5
            #     recheck_count += 1
            else:
                bot_point = mid_point
                mid_point = (top_point + bot_point)/2
                recheck_count = 0

            dist = np.sqrt(np.sum((top_point/self.max_dose - bot_point/self.max_dose)**2))
            self.current_step += 1

            if self.current_step % 5 == 0:
                self.eng.eval('close all;',nargout=0)

        bin_search_results = {}
        bin_search_results['pareto_optimal_point'] = top_point

        dvh = self._eval_point(bin_search_results['pareto_optimal_point'])
        self.dvh_history[:-1,...] = self.dvh_history[1:,...]
        self.dvh_history[-1,...] = dvh
        # sf = self._calc_sf(dvh_hist=self.dvh_history)
        sf = self._calc_eud_score(point_coords = bin_search_results['pareto_optimal_point'], 
                                   bound_vertices_list = bound_vertices_list)
        bin_search_results['sf'] = sf
        bin_search_results['dvh'] = dvh

        # last_good_point, last_good_sf, last_good_dvh = self._forward_search_to_nearest_border(bin_search_results['pareto_optimal_point'],
        #                                                                                   bin_search_results['sf'],
        #                                                                                   bin_search_results['dvh'])
        # bin_search_results['pareto_optimal_point'] = last_good_point
        # bin_search_results['sf'] = last_good_sf
        # bin_search_results['dvh'] = last_good_dvh

        return bin_search_results

    def in_hull(self, points, x):
        from scipy.optimize import linprog
        points = np.array(points)
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        try:
            lp = linprog(c, A_eq=A, b_eq=b)
            return lp.success
        except:
            return False

    def approx_binary_search(self, start_coord, end_coord, conv_hull_points_list = [], search_thresh = 1e-8, bound_vertices_list = None):
        

        top_point = start_coord
        bot_point = end_coord
        mid_point = (top_point + bot_point)/2
        dist = np.sqrt(np.sum((top_point/self.max_dose - bot_point/self.max_dose)**2))
        recheck_count = 0
        while dist > search_thresh:  
            if self.in_hull(np.array(conv_hull_points_list),mid_point):
                    top_point = mid_point
                    mid_point = (top_point + bot_point)/2
                    recheck_count = 0
            else:
                bot_point = mid_point
                mid_point = (top_point + bot_point)/2
                recheck_count = 0

            dist = np.sqrt(np.sum((top_point/self.max_dose - bot_point/self.max_dose)**2))
            self.current_step += 1

            if self.current_step % 5 == 0:
                self.eng.eval('close all;',nargout=0)

        bin_search_results = {}
        bin_search_results['pareto_optimal_point'] = top_point

        dvh = None
        # sf = self._calc_sf(dvh_hist=self.dvh_history)
        sf = self._calc_approx_eud_score(point_coords = bin_search_results['pareto_optimal_point'])
        bin_search_results['sf'] = sf
        bin_search_results['dvh'] = dvh


        return bin_search_results


    def _calc_eud_score(self, point_coords, bound_vertices_list = None, status = None):
        assert len(self.constr_dict['name_list']) == len(self.constr_dict['obj_type_list'])
        assert len(self.constr_dict['obj_type_list']) == len(self.constr_dict['param_list'])
        assert len(self.constr_dict['param_list']) == len(self.constr_dict['obj_num_list'])
        assert len(self.constr_dict['obj_num_list']) == len(self.constr_dict['eud_weights'])

        # bound_vertices_array =  np.array(bound_vertices_list)
        # bound_mins = np.min(bound_vertices_array,axis=0)
        # normalized_point_coords = (np.array(point_coords)-bound_mins)/(DOSE_MAX-bound_mins) # range 0 to 1
        
        normalized_point_coords = point_coords/DOSE_MAX

        if status is None:
            status = int(self.eng.eval('resultGUI.info.status',nargout=1))
            
        iters_run = int(self.eng.eval('resultGUI.info.iter',nargout=1))
        num_max_iters = int(self.eng.eval('resultGUI.usedOptimizer.options.max_iter',nargout=1))
        # if iters_run >= num_max_iters or (status != 0 and status != 1): #status = 5 is for early termination of infeasible problems
        if iters_run >= num_max_iters or (status != 0 and status != 1):
            is_infeasible_plan = True
            print('Run for iters: ',iters_run,'Max iters:',num_max_iters)
        else:
            is_infeasible_plan = False

        if is_infeasible_plan:
            score = 1
        else:
            numer_list = []
            denom_list = []
            for i, coord in enumerate(normalized_point_coords):
                idx = self.adjusted_constraint_ids[i]
                weight = self.constr_dict['eud_weights'][idx]
                numer = coord*weight
                denom = weight
                numer_list.append(numer)
                denom_list.append(denom)

            score = np.sum(numer_list,axis=0)/np.sum(denom_list,axis=0)
        return score

    def _calc_approx_eud_score(self, point_coords):
        assert len(self.constr_dict['name_list']) == len(self.constr_dict['obj_type_list'])
        assert len(self.constr_dict['obj_type_list']) == len(self.constr_dict['param_list'])
        assert len(self.constr_dict['param_list']) == len(self.constr_dict['obj_num_list'])
        assert len(self.constr_dict['obj_num_list']) == len(self.constr_dict['eud_weights'])

        # bound_vertices_array =  np.array(bound_vertices_list)
        # bound_mins = np.min(bound_vertices_array,axis=0)
        # normalized_point_coords = (np.array(point_coords)-bound_mins)/(DOSE_MAX-bound_mins) # range 0 to 1
        
        normalized_point_coords = point_coords/DOSE_MAX

        
        numer_list = []
        denom_list = []
        for i, coord in enumerate(normalized_point_coords):
            idx = self.adjusted_constraint_ids[i]
            weight = self.constr_dict['eud_weights'][idx]
            numer = coord*weight
            denom = weight
            numer_list.append(numer)
            denom_list.append(denom)

        score = np.sum(numer_list,axis=0)/np.sum(denom_list,axis=0)
        return score

    def _calc_sf(self, dvh_hist = np.zeros((DVH_HISTORY_LEN,)+DVH_SHAPE), status = None, use_ptv_ci_and_hi = False):
        '''
        :param struct_names: list of names (str)
        :param dvh: dvh from which to calculate sf
        :return: sf
        '''

        def find_nearest_dose_given_volume_point(volume_array, vol, dose_array):
            volume_array = np.asarray(volume_array)
            idx = (np.abs(volume_array - vol)).argmin()
            return dose_array[idx]

        def find_idx_of_nearest_val_in_arr(arr, val):
            arr = np.asarray(arr)
            idx = (np.abs(arr - val)).argmin()
            return idx

        def solve_reg_lower_gamma_for_k_and_theta(upper_90_ci,constr_val):
            possible_ks = np.linspace(1e-4,10000,100000)
            possible_thetas = upper_90_ci/(sc.gammaincinv(possible_ks,0.95)+1e-10)
            possible_ps = sc.gammainc(possible_ks,constr_val/possible_thetas)
            idx = find_idx_of_nearest_val_in_arr(possible_ps, 0.5)
            k_val = possible_ks[idx]
            theta_val = possible_thetas[idx]
            return k_val, theta_val

        def solve_reg_upper_gamma_for_k_and_theta(lower_90_ci,constr_val):
            possible_ks = np.linspace(1e-4,10000,100000)
            possible_thetas = lower_90_ci/(sc.gammainccinv(possible_ks,0.95)+1e-10)
            possible_ps = sc.gammaincc(possible_ks,constr_val/possible_thetas)
            idx = find_idx_of_nearest_val_in_arr(possible_ps, 0.5)
            k_val = possible_ks[idx]
            theta_val = possible_thetas[idx]
            return k_val, theta_val

        def sigmoid_fun(x,l = 1,reverse = False):
            if reverse:
                out = -1/(1+np.exp(-l*(x)))+1
            else:
                out = 1/(1+np.exp(-l*(x)))
            return out
        def piecewise_sigmoid(x,l1 = 1,l2 = 1,reverse = False):
            try:
                out = np.zeros(x.shape)
                for i in range(len(x)):
                    if x[i] <= 0:
                        out[i] = sigmoid_fun(x[i],l = l1,reverse = reverse)
                    elif x[i] > 0:
                        out[i] = sigmoid_fun(x[i],l = l2,reverse = reverse)
            except:
                if x <= 0:
                        out = sigmoid_fun(x,l = l1,reverse = reverse)
                elif x > 0:
                    out = sigmoid_fun(x,l = l2,reverse = reverse)
            return out

        switcher = {
            '0': 'DoseObjectives.matRad_EUD',
            '1': 'DoseObjectives.matRad_MaxDVH',
            '2': 'DoseObjectives.matRad_MeanDose',
            '3': 'DoseObjectives.matRad_MinDVH',
            '4': 'DoseObjectives.matRad_SquaredDeviation',
            '5': 'DoseObjectives.matRad_SquaredOverdosing',
            '6': 'DoseObjectives.matRad_SquaredUnderdosing',
            '7': 'DoseConstraints.matRad_MinMaxDose',
            '8': 'DoseConstraints.matRad_MinMaxDVH',
            '9': 'DoseConstraints.matRad_MinMaxEUD',
            '10': 'DoseConstraints.matRad_MinMaxMeanDose'
        }
        num_of_fractions = self.eng.eval('pln.numOfFractions',nargout = 1)



        data_dict = {}
        for n in range(dvh_hist.shape[0]):
            constr_count = 0
            for struct_num, obj_type_val in enumerate(self.constr_dict['obj_type_list']):
                dvh_idx = self.unique_structs.index(self.constr_dict['struct_num_list'][struct_num])
                classname = switcher[str(obj_type_val)]
                priority = self.constr_dict['priority_list'][struct_num]
                # print('Structure Num: ',struct_num)
                # print('classname: ',classname)
                # print('priority: ',priority)
                # print('DVH Index: ',dvh_idx)

                if priority > 1: #for oars
                    if classname==switcher['8']: #assuming priority of ptv is 1 and priority of oars > 1
                        constr_count += 1
                        dose_constr, vmin, vmax = self.constr_dict['param_list'][struct_num]
                        plan_value = find_nearest_dose_given_volume_point(dvh_hist[n, dvh_idx, :, 1], vmax,dvh_hist[n, dvh_idx, :, 0])
 
                        constr_type = 2 #dvh constraints are type 2
                        data_dict['dvh_'+str(n+1)+'_constraint_'+str(constr_count)] = [priority,
                                                                                       dose_constr/num_of_fractions,
                                                                                       constr_type,
                                                                                       plan_value]

                    if classname==switcher['7']:
                        constr_count += 1
                        dose_min, dose_max, compute_flag = self.constr_dict['param_list'][struct_num]
                        plan_max = find_nearest_dose_given_volume_point(dvh_hist[n, dvh_idx, :, 1], 0,dvh_hist[n, dvh_idx, :, 0]) #plan max defined as max dose received to 2% of volume

                        constr_type = 1 #max constraints are type 1
                        data_dict['dvh_'+str(n+1)+'_constraint_'+str(constr_count)] = [priority,
                                                                                       dose_max/num_of_fractions,
                                                                                       constr_type,
                                                                                       plan_max]
                elif priority == 1: #for ptv
                    if classname==switcher['7']:
                        constr_count += 1
                        dose_min, dose_max, compute_flag = self.constr_dict['param_list'][struct_num]
                        plan_max = find_nearest_dose_given_volume_point(dvh_hist[n, dvh_idx, :, 1], 0,dvh_hist[n, dvh_idx, :, 0]) #plan max defined as max dose received to 2% of volume
                        
                        constr_type = 1 #max constraints are type 1
                        data_dict['dvh_'+str(n+1)+'_constraint_'+str(constr_count)] = [priority,
                                                                                       dose_max/num_of_fractions,
                                                                                       constr_type,
                                                                                       plan_max]
                        constr_count += 1
                        
                        plan_min = find_nearest_dose_given_volume_point(dvh_hist[n, dvh_idx, :, 1], 98, dvh_hist[n, dvh_idx, :, 0])  # plan min defined as min dose received to 95% of volume
                        
                        constr_type = 0 #min constraints are type 0
                        data_dict['dvh_'+str(n+1)+'_constraint_'+str(constr_count)] = [priority,
                                                                                       dose_min/num_of_fractions,
                                                                                       constr_type,
                                                                                       plan_min]



        if DVH_HISTORY_LEN>1:
            raise NotImplementedError()
        else:

            data = np.zeros((dvh_hist.shape[0],constr_count,len(data_dict['dvh_1_constraint_1'])))
            for n in range(dvh_hist.shape[0]):
                for c in range(constr_count):
                    data[n,c,:] = np.array(data_dict['dvh_'+str(n+1)+'_constraint_'+str(c+1)])

            print('DVH History Shape: ',dvh_hist.shape)
            plan_vals = np.squeeze(data[:,:,-1])
            constr_val = np.squeeze(data[-1,:,-3])
            constr_type = np.squeeze(data[-1,:,-2])

            

            '''Compute Metric'''
            plan_val_current_dvh = np.squeeze(data[-1,:,-1])
            priority_current_dvh = np.squeeze(data[-1,:,-4])
            numer = []
            score_dict = {
                'in_vals': [],
                'priorities': [],
                'constr_num': [],
                'constr_type': [],
                'uw_scores': [],
                'ptv_check': [0],
                'body_check': [0]
            }
            ptv_scores = []
            ptv_idx = []

            if use_ptv_ci_and_hi:
                struct_num_set = list(sorted(set(self.constr_dict['struct_num_list']), key = self.constr_dict['struct_num_list'].index))
        
                #assuming ptv is the second to last
                ptv_struct_num = struct_num_set[-2]
                print('PTV structure num:', ptv_struct_num)
                conformity_idx, homogeneity_idx = self.get_ci_and_hi(ptv_struct_num = ptv_struct_num)
                conformity_score = np.abs(1-conformity_idx)
                homogeneity_score = homogeneity_idx/100
                ptv_score = (conformity_score+homogeneity_score)/2

            if status is None:
                status = int(self.eng.eval('resultGUI.info.status',nargout=1))
            
            iters_run = int(self.eng.eval('resultGUI.info.iter',nargout=1))
            num_max_iters = int(self.eng.eval('resultGUI.usedOptimizer.options.max_iter',nargout=1))
            if iters_run >= num_max_iters or status == 5: #status = 5 is for early termination of infeasible problems
                score_dict['ptv_check'].append(1)
                print('Run for iters: ',iters_run,'Max iters:',num_max_iters)
            for c in range(constr_count):
                
                in_val = num_of_fractions*(plan_val_current_dvh[c]-constr_val[c])   
                score_dict['priorities'].append(priority_current_dvh[c])
                score_dict['constr_num'].append(c)
                score_dict['constr_type'].append(constr_type[c])
                if constr_type[c]==2:
                    if priority_current_dvh[c]==1: #if ptv use a steeper sigmoid
                        uw_score = 1-piecewise_sigmoid(in_val,l1 = 500,l2 = 1, reverse = True)
                        score_dict['uw_scores'].append(uw_score)
                    else: #if oar use a flatter sigmoid
                        uw_score = 1-piecewise_sigmoid(in_val,l1 = 0.02,l2 = 3, reverse = True)
                        score_dict['uw_scores'].append(uw_score)
                elif constr_type[c]==1:
                    if priority_current_dvh[c]==1: #if ptv use a steeper sigmoid
                        uw_score = 1-piecewise_sigmoid(in_val,l1 = 500,l2 = 1, reverse = True)
                        score_dict['uw_scores'].append(uw_score)
                    else: #if oar use a flatter sigmoid
                        uw_score = 1-piecewise_sigmoid(in_val,l1 = 3,l2 = 3, reverse = True)
                        score_dict['uw_scores'].append(uw_score)
                elif constr_type[c]==0:
                    if priority_current_dvh[c]==1: #if ptv use a steeper sigmoid 
                        uw_score = 1-piecewise_sigmoid(in_val,l1 = 500,l2 = 2, reverse = False)
                        score_dict['uw_scores'].append(uw_score)
                    else: #if oar use a flatter sigmoid (this one is unused)
                        uw_score = 1-piecewise_sigmoid(in_val,l1 = 500,l2 = 0.5, reverse = False)
                        score_dict['uw_scores'].append(uw_score)
                score_dict['in_vals'].append(in_val)


            print('Constraints: ',data[-1,:,-3]*num_of_fractions, 'Plan values: ',data[-1,:,-1]*num_of_fractions, \
                  'Unweighted score values: ',score_dict['uw_scores'])

 
            score_dict['unique_idxs'] = list(sorted(set(score_dict['priorities']), key = score_dict['priorities'].index))
            print('Unique structure priorities: ',score_dict['unique_idxs'])
            if max(score_dict['ptv_check']) == 0: # and max(score_dict['body_check']) == 0:

                data_by_priority = {}            
                data_by_priority['numer'] = []
                data_by_priority['denom'] = []
                for k, unique_idx in enumerate(score_dict['unique_idxs']):
                    data_by_priority['locations_for_p'+str(unique_idx)] = [i for i,x in enumerate(score_dict['priorities']) if x == unique_idx]
                    data_by_priority['uw_scores_for_p'+str(unique_idx)] = [score_dict['uw_scores'][i] for i,x in enumerate(score_dict['priorities']) if x == unique_idx]
                for k, unique_idx in enumerate(score_dict['unique_idxs']):
                    data_by_priority['score_for_p'+str(unique_idx)] = np.mean(data_by_priority['uw_scores_for_p'+str(unique_idx)])
                    if unique_idx != 1: #if not ptv
                        numer = 2**(-unique_idx+1)*(data_by_priority['score_for_p'+str(unique_idx)]) #l=0.3 is about 15 Gy interval
                        denom = 2**(-unique_idx+1)
                        data_by_priority['numer'].append(numer)
                        data_by_priority['denom'].append(denom)

                print('Numer: ',data_by_priority['numer'], 'Denom: ',data_by_priority['denom'])
                sf = np.sum(data_by_priority['numer'],axis=0)/np.sum(data_by_priority['denom'],axis=0)
                
                
            else:
                sf = 1
               


            self.score_dict = score_dict

            if use_ptv_ci_and_hi:
                sf = np.sqrt(ptv_score**2+sf**2)
        return sf

    def score_ground_truth_dose_distribution(self, ctDir,rtStDir,rtDoseDir):
        from scipy import interpolate
        import pandas as pd
        
        def find_nearest_dose_given_volume_point(volume_array, vol, dose_array):
            volume_array = np.asarray(volume_array)
            idx = (np.abs(volume_array - vol)).argmin()
            return dose_array[idx]


        self.eng = matlab.engine.start_matlab()
        # self.eng.eval('cd ' + os.environ['matRad'],nargout=0)
        self.eng.eval('addpath(genpath("'+os.environ['matRad']+'"));')
        # self.eng.eval('set(0,"DefaultFigureVisible","off")',nargout=0)
        self.eng.eval('clear all; close all; clc;',nargout=0)
        self.eng.eval('ctDir = "'+ctDir+'";',nargout=0)
        self.eng.eval('rtStDir = "'+rtStDir+'";',nargout=0)
        self.eng.eval('rtDoseDir = "'+rtDoseDir+'";',nargout=0)
        self.eng.eval('get_physician_dvh_script',nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')
        unique_structs = list(sorted(set(struct_num_list), key = struct_num_list.index))
        self.unique_structs = copy.deepcopy(unique_structs)
        self.constr_dict['struct_num_list'] = copy.deepcopy(struct_num_list)
        assert NUM_STRUCTURES==len(unique_structs)

        structure_key = ['Rectum','Bladder','FH R','FH L','PTV','Body']
        df = pd.read_csv('./clean_prostate_data/isodose/roi_id.csv')
        case_num = int(ctDir.split('/')[-2])
        print('Case number:',case_num)
        df_match = df.loc[df['Patient ID']==case_num]
        struct_intermediate_ids = [df_match[x].values[0] for x in structure_key]
        struct_ids = []
        for i in range(len(struct_intermediate_ids)):
            for j in range(int(self.eng.eval('numel(fieldnames(doseInfo.DVHSequence))',nargout = 1))):   
                struct_intermediate_num = self.eng.eval('doseInfo.DVHSequence.Item_'+str(j+1)+'.DVHReferencedROISequence.Item_1.ReferencedROINumber',nargout=1)
                if struct_intermediate_ids[i] == int(struct_intermediate_num):
                    struct_ids.append(j+1)
        assert len(struct_ids) == len(structure_key)
        ptv_struct_num = struct_ids[-2]
        dvh = np.zeros(DVH_SHAPE)
        dvh_hist = np.zeros((1,)+DVH_SHAPE)
        # rescale_phys = np.round(1.05*PRESCRIBED_DOSE)/int(self.eng.eval('doseInfo.DVHSequence.Item_'+str(ptv_struct_num)+'.DVHMeanDose',nargout=1))
        # print('Rescale phys:',rescale_phys)
        for i,struct_id in enumerate(struct_ids):
            dvh_max_dose = int(self.eng.eval('doseInfo.DVHSequence.Item_'+str(struct_id)+'.DVHMaximumDose',nargout=1))
            dvh_num_bins = int(self.eng.eval('doseInfo.DVHSequence.Item_'+str(struct_id)+'.DVHNumberOfBins',nargout=1))
            num_of_fractions = int(self.eng.eval('pln.numOfFractions',nargout = 1))
            dose_grid = np.squeeze(np.linspace(0,dvh_max_dose/num_of_fractions,dvh_num_bins))

            volume_points_raw = self.eng.eval('doseInfo.DVHSequence.Item_'+str(struct_id)+'.DVHData(2:2:end)',nargout=1)
            volume_points_max = np.max(self.eng.eval('doseInfo.DVHSequence.Item_'+str(struct_id)+'.DVHData(2:2:end)',nargout=1))
            volume_points = np.squeeze(volume_points_raw/volume_points_max*100)

            print('Dim of dose grid:',len(dose_grid))
            print('Dim of dose volume points:',len(volume_points))
            f = interpolate.interp1d(dose_grid, volume_points,kind='linear')
            new_dose_grid = np.linspace(0,dvh_max_dose/num_of_fractions,dvh.shape[1])
            new_volume_points = f(new_dose_grid)
            dvh[i,:,0] = new_dose_grid
            dvh[i,:,1] = new_volume_points
            # plt.plot(dvh[i,:,0],dvh[i,:,1])

            if struct_id == struct_ids[-2]:
                found_dose = find_nearest_dose_given_volume_point(dvh[i,:,1],95,dvh[i,:,0]*num_of_fractions)
                rescale_factor = NORMALIZED_DOSE/found_dose
                print('Normalized dose:',NORMALIZED_DOSE)
                print('Found dose:',found_dose)
                print('Rescaling factor:',rescale_factor)

        for i in range(dvh.shape[0]):
            dvh[i,:,0] = dvh[i,:,0]*rescale_factor

        self.eng.eval('resultGUI.info.iter = 200;',nargout=0)
        self.eng.eval('resultGUI.usedOptimizer.options.max_iter = 250;',nargout=0)
        dvh_hist[0,...] = dvh
        sf = self._calc_sf(dvh_hist=dvh_hist,status = 1)
        print('sf of ground truth:',sf)
        self.rescale_factor = rescale_factor
        # plt.show()
        return sf, dvh

    def rescale_and_score_dvh(self,in_dvh,compute_ptv_struct_num = False):
        def find_nearest_dose_given_volume_point(volume_array, vol, dose_array):
            volume_array = np.asarray(volume_array)
            idx = (np.abs(volume_array - vol)).argmin()
            return dose_array[idx]

        if compute_ptv_struct_num:
            struct_num_set = list(sorted(set(self.constr_dict['struct_num_list']), key = self.constr_dict['struct_num_list'].index))
            
            # #assuming ptv is the second to last
            # ptv_struct_num = struct_num_set[-2]
            ptv_struct_num = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
            print('PTV structure num:', ptv_struct_num)

        num_of_fractions = int(self.eng.eval('pln.numOfFractions',nargout = 1))
        found_dose = find_nearest_dose_given_volume_point(in_dvh[-2,:,1],95,in_dvh[-2,:,0]*num_of_fractions)
        rescale_factor = NORMALIZED_DOSE/found_dose
        print('Normalized dose:',NORMALIZED_DOSE)
        print('Found dose:',found_dose)
        print('Rescaling factor:',rescale_factor)

        in_dvh = np.squeeze(in_dvh)
        for i in range(in_dvh.shape[0]):
            in_dvh[i,:,0] = in_dvh[i,:,0]*rescale_factor

        self.eng.eval('resultGUI.info.iter = 200;',nargout=0)
        self.eng.eval('resultGUI.usedOptimizer.options.max_iter = 250;',nargout=0)
        dvh_hist = np.zeros((1,)+DVH_SHAPE)
        dvh_hist[0,...] = in_dvh

        sf = self._calc_sf(dvh_hist=dvh_hist)
        self.rescale_factor = rescale_factor

        if compute_ptv_struct_num:
            return sf, in_dvh, ptv_struct_num
        else:
            return sf, in_dvh

    def get_structure_means(self,):
        mean_vals = self.eng.eval('[qi(:).mean]', nargout = 1)
        mean_vals = np.array(mean_vals)*self.rescale_factor
        return mean_vals

    def get_ptv_stats_from_treatment_plan(self,ctDir,rtStDir,rtDoseDir):
        self.eng = matlab.engine.start_matlab()
        # self.eng.eval('cd ' + os.environ['matRad'],nargout=0)
        self.eng.eval('addpath(genpath("'+os.environ['matRad']+'"));')
        # self.eng.eval('set(0,"DefaultFigureVisible","off")',nargout=0)
        self.eng.eval('clear all; close all; clc;',nargout=0)
        self.eng.eval('ctDir = "'+ctDir+'";',nargout=0)
        self.eng.eval('rtStDir = "'+rtStDir+'";',nargout=0)
        self.eng.eval('rtDoseDir = "'+rtDoseDir+'";',nargout=0)
        self.eng.eval('[ct, cst, pln, stf, resultGUI] = import_dicom_return_stf_and_result_gui( ctDir, rtStDir, rtDoseDir);',nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')

        '''Remove unwatned structures'''
        unique_structs = list(sorted(set(struct_num_list), key = struct_num_list.index))
        assert NUM_STRUCTURES==len(unique_structs)
        
        self.eng.eval('struct_ids = '+str(unique_structs)+';',nargout=0)
        self.eng.eval('cst_small = cst(struct_ids,:);',nargout=0)
        self.eng.eval('cst = cst_small;',nargout=0)
        struct_num_list = self._return_struct_num_list(self.constr_dict['name_list'])
        print('New Struct_num_list: ',struct_num_list)
        if None in struct_num_list:
            raise ValueError('Cannot find a structure')
        ''''''
        self.constr_dict['struct_num_list'] = struct_num_list
        assert len(self.constr_dict['struct_num_list']) == len(self.constr_dict['name_list']), 'Mismatch struct_num_list and name_list'
        self._set_all_structure_priorities(priority=np.max(np.array(self.constr_dict['priority_list']))+1)

        #assuming prescribed dose is at D95
        self.eng.eval('qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose,[],[95]);',nargout=0)
        num_structs = self.eng.eval('size(qi,2)',nargout=1)
        struct_num_set = list(sorted(set(self.constr_dict['struct_num_list']), key = self.constr_dict['struct_num_list'].index))
        
        # #assuming ptv is the second to last
        # ptv_struct_num = struct_num_set[-2]
        ptv_struct_num = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        print('PTV structure num:', ptv_struct_num)
        prescribed_dose = self.eng.eval('qi('+str(ptv_struct_num)+').D_95',nargout=1)
        num_of_fractions = self.eng.eval('pln.numOfFractions',nargout=1)
        prescribed_dose = prescribed_dose*num_of_fractions
        print('Prescribed Dose:',prescribed_dose)

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
        gi_val = self.get_gi(ptv_struct_num=ptv_struct_num, pres_dose=prescribed_dose)

        return ci_val,hi_val,gi_val

    # def get_ci_and_hi(self, ptv_struct_num):
    #     self.eng.eval('qi = matRad_calcQualityIndicators(cst,pln,resultGUI.physicalDose);',nargout=0)
    #     self.eng.eval('qi_names = fieldnames(qi);',nargout=0)
    #     conformity_name = self.eng.eval('qi_names{end-1}',nargout=1)
    #     homogeneity_name = self.eng.eval('qi_names{end}',nargout=1)
    #     conformity_idx = self.eng.eval('qi('+str(ptv_struct_num)+').'+str(conformity_name),nargout=1)
    #     homogeneity_idx = self.eng.eval('qi('+str(ptv_struct_num)+').'+str(homogeneity_name),nargout=1)
    #
    #     print('Reference dose:',self.eng.eval('cst{'+str(ptv_struct_num)+',6}',nargout=1))
    #     return conformity_idx, homogeneity_idx

    def get_ci_and_hi(self,ptv_struct_num = None, pres_dose = PRESCRIBED_DOSE):
        num_of_fractions = self.eng.eval('pln.numOfFractions', nargout=1)
        ref_dose = pres_dose / num_of_fractions
        self.eng.eval('ptv_qi = calc_conformity_and_homogeneity(cst,pln,resultGUI.physicalDose,' + \
                      str(ref_dose) + ');', nargout=0)
        if ptv_struct_num is None:
            target_id = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        else:
            target_id = ptv_struct_num
        ci_val = self.eng.eval('ptv_qi(' + str(target_id) + ').CI', nargout=1)
        hi_val = self.eng.eval('ptv_qi(' + str(target_id) + ').HI', nargout=1)
        print('CI from plan:', ci_val)
        print('HI from plan:', hi_val)
        return ci_val,hi_val


    def get_gi(self,ptv_struct_num = None, pres_dose = PRESCRIBED_DOSE):
        num_of_fractions = self.eng.eval('pln.numOfFractions', nargout=1)
        ref_dose = pres_dose / num_of_fractions
        self.eng.eval('gi_qi = calc_gradient_index(cst,pln,resultGUI.physicalDose,' + \
                      str(ref_dose) + ');', nargout=0)
        if ptv_struct_num is None:
            target_id = self.constr_dict['struct_num_list'][self.constr_dict['name_list'].index('ptv')]
        else:
            target_id = ptv_struct_num
        gi_val = self.eng.eval('gi_qi(' + str(target_id) + ').GI', nargout=1)
        print('GI from plan:', gi_val)
        return gi_val