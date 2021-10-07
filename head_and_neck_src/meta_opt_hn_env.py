from matRad_wo_hn_env import *

import numpy as np
import os
import glob
import copy
from gym.utils import seeding
import ray

MAX_WEIGHT = 1000

class POPS_algo_wo():

    def __init__(self, env_config = None):

        self.ftol = 1e-5
        self.ctol = 1
        self.simplex_param_actions = [0.3,0.4,0.5,0.6,0.7] #modify center of mass calculation to shift centroid
        self.num_constraints = 8
        self.current_step = 0
        self.max_iter = 30

        self.seed()

    def reset(self,
              sample_case_num = None,
              data_parent_dir = './clean_prostate_data/isodose/usable/',
              couch_angle_list = [0, 0, 0, 0, 0, 0, 0, 0, 0],
              gantry_angle_list = [0, 40, 80, 120, 160, 200, 240, 280, 320]):

        self.current_step = 0
        case_list = sorted([i for i in glob.glob(os.path.join(data_parent_dir,'*')) if os.path.isdir(i)])
        num_cases = len(case_list)
        if sample_case_num is None:
            sample_case_num = int(self.np_random.uniform(0,num_cases))

        self.patientDir = case_list[sample_case_num]
        print('Patient Directory: ',self.patientDir)
        self.ctDir = os.path.join(case_list[sample_case_num],'CT')
        self.rtStDir = os.path.join(case_list[sample_case_num],'RTst')
        self.rtDoseDir = os.path.join(case_list[sample_case_num],'RTDOSE')
        self.rtPlanDir = os.path.join(case_list[sample_case_num],'RTPLAN')

        self.start_coord = np.array([MAX_WEIGHT*0.75,
                                     0.5,
                                     0.5,
                                     0.5,
                                     0.5,
                                     0.5,
                                     0.5,
                                     MAX_WEIGHT*0.25-3,])
        self.start_coord_hist = []
        self.start_coord_hist.append(copy.deepcopy(self.start_coord))
        self.bound_vertices = []
        self.simplex_hist = []
        self.simplex = []
        self.projected_simplex_hist = []
        self.projected_simplex = []
        self.sf_hist = []
        self.sf_scores = []
        self.dvhs_hist = []
        self.dvhs = []
        self.bound_sf_scores = []
        self.couch_angle_list = couch_angle_list
        self.gantry_angle_list = gantry_angle_list
        self.bounded_search = True
        self.num_workers = len(self.start_coord)

        #initialize workers and bin search for each of the constraints
        unordered_simplex, unordered_projected_simplex, unordered_sf_scores, unordered_dvhs = self.initialize_all_workers()
        print('Bounded at:',self.bound_vertices)

        self.pareto_points_list = copy.deepcopy(unordered_projected_simplex)
        self.pareto_points_scores = copy.deepcopy(unordered_sf_scores)
        self.pareto_points_dvhs = copy.deepcopy(unordered_dvhs)



        self.format_initial_simplex(unordered_simplex, unordered_projected_simplex, unordered_sf_scores, unordered_dvhs )

        obs = self.get_obs(self.simplex,self.projected_simplex,self.sf_scores)


        assert self.num_workers == len(self.start_coord)
        assert self.num_workers == self.num_constraints

        return obs

    def initialize_all_workers(self):
        import time
        workers = []
        results = []
        bound_weights_list = []
        print('Now initializing workers...')
        for i in range(self.num_workers):
            remote_class = ray.remote(MatRadMetaOpt)
            worker = remote_class.remote()
            time.sleep(5)
            worker.reset_pops.remote(self.ctDir,
                                     self.rtStDir,
                                     couch_angle_list=self.couch_angle_list,
                                     gantry_angle_list=self.gantry_angle_list,
                                     acceptable_iter=150,
                                     acceptable_constr_viol_tol=0.0001,
                                     ipopt_max_iters=150)


            bound_weights = np.zeros(len(self.start_coord))
            bound_weights[i] = MAX_WEIGHT
            print('Bound weights: ', bound_weights)
            result = worker.wo_projection_no_score.remote(bound_weights)

            workers.append(worker)
            results.append(result)
            bound_weights_list.append(bound_weights)

        self.workers = workers
        unordered_simplex = []
        unordered_projected_simplex = []
        unordered_dvhs = []
        structure_means = []
        for i in range(len(results)):
            unordered_simplex.append(ray.get(results[i])['pareto_optimal_point'])
            unordered_projected_simplex.append(ray.get(results[i])['pareto_optimal_point'])
            unordered_dvhs.append(ray.get(results[i])['dvh'])
            structure_means.append(ray.get(results[i])['struct_mean_array'])

        self.bound_vertices = copy.deepcopy(bound_weights_list)
        self.bound_weights_list = copy.deepcopy(bound_weights_list)
        self.bound_dvhs = copy.deepcopy(unordered_dvhs)

        results = []
        unordered_sf_scores = []
        for i in range(self.num_workers):
            result = self.workers[i].calc_score.remote()
            results.append(result)
        for i in range(len(results)):
            unordered_sf_scores.append(ray.get(results[i]))

        self.bound_sf_scores = copy.deepcopy(unordered_sf_scores)
        print('Bound scores:',self.bound_sf_scores)

        results = []
        unordered_simplex = []
        initial_simplex = self.guess_initial_simplex()
        for i in range(self.num_workers):
            initial_simplex_point = initial_simplex[i]

            print('Initial simplex point: ', initial_simplex_point)
            result = self.workers[i].perform_wo_projection.remote(initial_simplex_point)
            results.append(result)
            unordered_simplex.append(initial_simplex_point)

        unordered_projected_simplex = []
        unordered_sf_scores = []
        unordered_dvhs = []
        for i in range(len(results)):
            unordered_projected_simplex.append(ray.get(results[i])['pareto_optimal_point'])
            unordered_sf_scores.append(ray.get(results[i])['sf'])
            unordered_dvhs.append(ray.get(results[i])['dvh'])
            print('Score:',ray.get(results[i])['sf'])
            print('Structure means:',ray.get(results[i])['struct_mean_array'])

        out_simplex = copy.deepcopy(unordered_simplex)
        out_projected_simplex = copy.deepcopy(unordered_projected_simplex)
        out_sf_scores = copy.deepcopy(unordered_sf_scores)
        out_dvhs = copy.deepcopy(unordered_dvhs)


        return out_simplex, out_projected_simplex, out_sf_scores, out_dvhs

    def guess_initial_simplex(self):
        s, ps, gs, dvhs = self.order_simplex_points(self.bound_vertices, self.bound_vertices,
                                                    self.bound_sf_scores, self.bound_dvhs)
        initial_simplex = []
        for i in range(self.num_constraints):
            initial_simplex.append(np.array(s[i]))


        # # apply initial shrink
        # for i in range(1, self.num_constraints):
        #     vec = initial_simplex[i] - initial_simplex[0]
        #     initial_simplex[i] = initial_simplex[0] + 0.5 * vec


        # apply initial shrink
        for i in range(0, self.num_constraints):
            vec = initial_simplex[i] - np.array(self.start_coord)
            initial_simplex[i] = np.array(self.start_coord) + 0.5 * vec

        return initial_simplex


    def format_initial_simplex(self,unordered_simplex, unordered_projected_simplex, unordered_sf_scores, unordered_dvhs ):
        '''
        Initial simplex is the bounds
        '''

        # order simplex points, projected simplex points, and sf scores
        s, ps, gs, dvhs = self.order_simplex_points(unordered_simplex, unordered_projected_simplex,
                                                    unordered_sf_scores, unordered_dvhs)
        self.simplex = copy.deepcopy(s)
        self.projected_simplex = copy.deepcopy(ps)
        self.sf_scores = copy.deepcopy(gs)
        self.dvhs = copy.deepcopy(dvhs)
        # self.projected_simplex_opt_weights = self.get_projected_simplex_opt_weights()

        self.simplex_hist.append(copy.deepcopy(self.simplex))
        self.projected_simplex_hist.append(copy.deepcopy(self.projected_simplex))
        self.sf_hist.append(copy.deepcopy(self.sf_scores))
        self.dvhs_hist.append(copy.deepcopy(self.dvhs))

        print('-----After ordering-----')
        print('Simplex:', self.simplex)
        # print('Projected simplex:', self.projected_simplex)
        print('Scores:', self.sf_scores)
        # print('Opt weights:', self.projected_simplex_opt_weights)

        self.initial_simplex = copy.deepcopy(self.simplex)
        self.initial_scores = copy.deepcopy(self.sf_scores)
        self.initial_projected_simplex = copy.deepcopy(self.projected_simplex)




    def step(self, action, use_parallel = True):
        if use_parallel:
            return self.parallel_step(action)
        else:
            return self.sequential_step(action)

    def point_correction(self,point):
        point = np.array(point)/np.sum(point)*MAX_WEIGHT
        return point

    def parallel_step(self, action):
        ''''''
        print('Peforming iteration of parallel Nelder-Mead Simplex Routine...')
        alpha = 2 * self.simplex_param_actions[action]
        gamma = 4 * self.simplex_param_actions[action]
        rho = self.simplex_param_actions[action]
        sigma = self.simplex_param_actions[action]

        candidate_simplex = copy.deepcopy(self.simplex)
        candidate_projected_simplex = copy.deepcopy(self.projected_simplex)
        candidate_sf_scores = copy.deepcopy(self.sf_scores)
        candidate_dvhs = copy.deepcopy(self.dvhs)

        decisions_to_shrink = [False] * (len(self.workers)-1)

        centroid_list = []


        for worker_id in range(1,len(self.workers)):
            # first calc centroid
            all_but_one_points = self.simplex[:worker_id] + self.simplex[worker_id+1:]
            centroid = np.mean(all_but_one_points,0)
            centroid_list.append(centroid)

        print('Alpha:', alpha, 'Gamma:', gamma, 'Rho:', rho, 'Sigma:', sigma)

        reflection_point_list = []
        projected_reflection_point_list = []
        reflection_point_sf_list = []
        reflection_point_dvh_list = []
        reflection_point_worker_ids = []
        reflection_results = []

        expansion_point_list = []
        projected_expansion_point_list = []
        expansion_point_sf_list = []
        expansion_point_dvh_list = []
        expansion_point_worker_ids = []
        expansion_results = []

        oc_point_list = []
        projected_oc_point_list = []
        oc_point_sf_list = []
        oc_point_dvh_list = []
        oc_point_worker_ids = []
        oc_results = []

        ic_point_list = []
        projected_ic_point_list = []
        ic_point_sf_list = []
        ic_point_dvh_list = []
        ic_point_worker_ids = []
        ic_results = []


        '''Reflection'''
        for worker_id in range(1,len(self.workers)):

            centroid = centroid_list[worker_id-1]

            vec = centroid - self.simplex[worker_id]
            reflection_point = centroid + alpha * vec
            reflection_point_list.append(reflection_point)
            reflection_point_worker_ids.append(worker_id)
            inside_hull_refl = self.in_hull(np.array(self.bound_vertices), reflection_point)
            if inside_hull_refl or not self.bounded_search:
                print('Reflection point before projection: ', reflection_point)
                result_refl = self.workers[worker_id].perform_wo_projection.remote(reflection_point)
                reflection_results.append(result_refl)
            else:
                result_refl = self.workers[worker_id].return_inf_score.remote(reflection_point)
                reflection_results.append(result_refl)

        # get synchronous reflection results
        for result_id in range(len(reflection_results)):
            projected_reflection_point = ray.get(reflection_results[result_id])['pareto_optimal_point']
            reflection_point_sf = ray.get(reflection_results[result_id])['sf']
            reflection_point_dvh = ray.get(reflection_results[result_id])['dvh']
            worker_id = reflection_point_worker_ids[result_id]
            print('Reflection point score for worker #'+str(worker_id)+' :', reflection_point_sf)
            print('Structure means for worker #'+str(worker_id)+' :', ray.get(reflection_results[result_id])['struct_mean_array'])
            projected_reflection_point_list.append(projected_reflection_point)
            reflection_point_sf_list.append(reflection_point_sf)
            reflection_point_dvh_list.append(reflection_point_dvh)

        # begin simplex logic
        for i in range(len(reflection_point_list)):
            worker_id = reflection_point_worker_ids[i]
            reflection_point = reflection_point_list[i]
            projected_reflection_point = projected_reflection_point_list[i]
            reflection_point_sf = reflection_point_sf_list[i]
            reflection_point_dvh = reflection_point_dvh_list[i]

            if reflection_point_sf < self.sf_scores[worker_id-1] and reflection_point_sf >= self.sf_scores[0]:
                candidate_simplex[worker_id] = reflection_point
                candidate_projected_simplex[worker_id] = projected_reflection_point
                candidate_sf_scores[worker_id] = reflection_point_sf
                candidate_dvhs[worker_id] = reflection_point_dvh
            elif reflection_point_sf < self.sf_scores[0]:
                '''Expansion'''
                centroid = centroid_list[worker_id - 1]
                vec = centroid - self.simplex[worker_id]
                expanded_point = centroid + gamma * vec
                expansion_point_list.append(expanded_point)
                expansion_point_worker_ids.append(worker_id)
                inside_hull_exp = self.in_hull(np.array(self.bound_vertices), expanded_point)

                if inside_hull_exp or not self.bounded_search:
                    print('Expanded point before projection: ', expanded_point)
                    result_exp = self.workers[worker_id].perform_wo_projection.remote(expanded_point)
                    expansion_results.append(result_exp)
                else:
                    result_exp = self.workers[worker_id].return_inf_score.remote(expanded_point)
                    expansion_results.append(result_exp)

            elif reflection_point_sf >= self.sf_scores[worker_id-1] and reflection_point_sf < self.sf_scores[worker_id]:
                '''Outside Contraction'''
                centroid = centroid_list[worker_id - 1]
                vec = reflection_point - centroid
                contraction_point_oc = centroid + rho * vec
                oc_point_list.append(contraction_point_oc)
                oc_point_worker_ids.append(worker_id)
                inside_hull_oc = self.in_hull(np.array(self.bound_vertices), contraction_point_oc)

                if inside_hull_oc or not self.bounded_search:
                    print('Outside contraction point before projection: ', contraction_point_oc)
                    result_oc = self.workers[worker_id].perform_wo_projection.remote(contraction_point_oc)
                    oc_results.append(result_oc)
                else:
                    result_oc = self.workers[worker_id].return_inf_score.remote(contraction_point_oc)
                    oc_results.append(result_oc)
            elif reflection_point_sf >= self.sf_scores[worker_id]:
                '''Inside Contraction'''
                centroid = centroid_list[worker_id - 1]
                vec = self.simplex[worker_id] - centroid  # self.simplex[-1] - centroid
                contraction_point_ic = centroid + rho * vec
                ic_point_list.append(contraction_point_ic)
                ic_point_worker_ids.append(worker_id)
                inside_hull_ic = self.in_hull(np.array(self.bound_vertices), contraction_point_ic)

                if inside_hull_ic or not self.bounded_search:
                    print('Inside contraction point before projection: ', contraction_point_ic)
                    result_ic = self.workers[worker_id].perform_wo_projection.remote(contraction_point_ic)
                    ic_results.append(result_ic)
                else:
                    result_ic = self.workers[worker_id].return_inf_score.remote(contraction_point_ic)
                    ic_results.append(result_ic)







        '''Get synchronous exp, oc, and ic results'''
        for result_id in range(len(expansion_results)):
            projected_expansion_point = ray.get(expansion_results[result_id])['pareto_optimal_point']
            expansion_point_sf = ray.get(expansion_results[result_id])['sf']
            expansion_point_dvh = ray.get(expansion_results[result_id])['dvh']
            worker_id = expansion_point_worker_ids[result_id]
            print('Expansion point score for worker #'+str(worker_id)+' :', expansion_point_sf)
            print('Structure means for worker #'+str(worker_id)+' :', ray.get(expansion_results[result_id])['struct_mean_array'])
            projected_expansion_point_list.append(projected_expansion_point)
            expansion_point_sf_list.append(expansion_point_sf)
            expansion_point_dvh_list.append(expansion_point_dvh)
        for result_id in range(len(oc_results)):
            projected_oc_point = ray.get(oc_results[result_id])['pareto_optimal_point']
            oc_point_sf = ray.get(oc_results[result_id])['sf']
            oc_point_dvh = ray.get(oc_results[result_id])['dvh']
            worker_id = oc_point_worker_ids[result_id]
            print('Outside contraction point score for worker #'+str(worker_id)+' :', oc_point_sf)
            print('Structure means for worker #'+str(worker_id)+' :', ray.get(oc_results[result_id])['struct_mean_array'])
            projected_oc_point_list.append(projected_oc_point)
            oc_point_sf_list.append(oc_point_sf)
            oc_point_dvh_list.append(oc_point_dvh)
        for result_id in range(len(ic_results)):
            projected_ic_point = ray.get(ic_results[result_id])['pareto_optimal_point']
            ic_point_sf = ray.get(ic_results[result_id])['sf']
            ic_point_dvh = ray.get(ic_results[result_id])['dvh']
            worker_id = ic_point_worker_ids[result_id]
            print('Inside contraction point score for worker #'+str(worker_id)+' :', ic_point_sf)
            print('Structure means for worker #'+str(worker_id)+' :', ray.get(ic_results[result_id])['struct_mean_array'])
            projected_ic_point_list.append(projected_ic_point)
            ic_point_sf_list.append(ic_point_sf)
            ic_point_dvh_list.append(ic_point_dvh)

        assert len(reflection_point_list) == len(projected_reflection_point_list)
        assert len(reflection_point_list) == len(reflection_point_sf_list)
        assert len(reflection_point_list) == len(reflection_point_dvh_list)
        assert len(reflection_point_list) == len(reflection_point_worker_ids)
        assert len(reflection_point_list) == len(reflection_results)

        assert len(expansion_point_list) == len(projected_expansion_point_list)
        assert len(expansion_point_list) == len(expansion_point_sf_list)
        assert len(expansion_point_list) == len(expansion_point_dvh_list)
        assert len(expansion_point_list) == len(expansion_point_worker_ids)
        assert len(expansion_point_list) == len(expansion_results)

        assert len(oc_point_list) == len(projected_oc_point_list)
        assert len(oc_point_list) == len(oc_point_sf_list)
        assert len(oc_point_list) == len(oc_point_dvh_list)
        assert len(oc_point_list) == len(oc_point_worker_ids)
        assert len(oc_point_list) == len(oc_results)

        assert len(ic_point_list) == len(projected_ic_point_list)
        assert len(ic_point_list) == len(ic_point_sf_list)
        assert len(ic_point_list) == len(ic_point_dvh_list)
        assert len(ic_point_list) == len(ic_point_worker_ids)
        assert len(ic_point_list) == len(ic_results)

        # expansion logic
        for i in range(len(expansion_point_list)):
            worker_id = expansion_point_worker_ids[i]
            expansion_point = expansion_point_list[i]
            projected_expansion_point = projected_expansion_point_list[i]
            expansion_point_sf = expansion_point_sf_list[i]
            expansion_point_dvh = expansion_point_dvh_list[i]
            check_j = None
            for j in range(len(reflection_point_list)):
                if worker_id == reflection_point_worker_ids[j]:
                    check_j = j

            assert check_j is not None
            if expansion_point_sf < reflection_point_sf_list[check_j]:
                candidate_simplex[worker_id] = expansion_point
                candidate_projected_simplex[worker_id] = projected_expansion_point
                candidate_sf_scores[worker_id] = expansion_point_sf
                candidate_dvhs[worker_id] = expansion_point_dvh
            else:
                candidate_simplex[worker_id] = reflection_point_list[check_j]
                candidate_projected_simplex[worker_id] = projected_reflection_point_list[check_j]
                candidate_sf_scores[worker_id] = reflection_point_sf_list[check_j]
                candidate_dvhs[worker_id] = reflection_point_dvh_list[check_j]

        # Outside contraction logic
        for i in range(len(oc_point_list)):
            worker_id = oc_point_worker_ids[i]
            oc_point = oc_point_list[i]
            projected_oc_point = projected_oc_point_list[i]
            oc_point_sf = oc_point_sf_list[i]
            oc_point_dvh = oc_point_dvh_list[i]
            check_j = None
            for j in range(len(reflection_point_list)):
                if worker_id == reflection_point_worker_ids[j]:
                    check_j = j

            assert check_j is not None
            if oc_point_sf < reflection_point_sf_list[check_j]:
                candidate_simplex[worker_id] = oc_point
                candidate_projected_simplex[worker_id] = projected_oc_point
                candidate_sf_scores[worker_id] = oc_point_sf
                candidate_dvhs[worker_id] = oc_point_dvh
            else:
                decisions_to_shrink[worker_id-1] = True

        # Inside contraction logic
        for i in range(len(ic_point_list)):
            worker_id = ic_point_worker_ids[i]
            ic_point = ic_point_list[i]
            projected_ic_point = projected_ic_point_list[i]
            ic_point_sf = ic_point_sf_list[i]
            ic_point_dvh = ic_point_dvh_list[i]

            if ic_point_sf < self.sf_scores[worker_id]:
                candidate_simplex[worker_id] = ic_point
                candidate_projected_simplex[worker_id] = projected_ic_point
                candidate_sf_scores[worker_id] = ic_point_sf
                candidate_dvhs[worker_id] = ic_point_dvh
            else:
                decisions_to_shrink[worker_id - 1] = True

        # Shrink logic
        if np.min(decisions_to_shrink) > 0:
            shrink_results = []
            for i in range(1, self.num_constraints):
                vec = candidate_simplex[i] - candidate_simplex[0]
                candidate_simplex[i] = candidate_simplex[0] + sigma * vec
                print('Shrink point ' + str(i + 1) + ' before projection: ', candidate_simplex[i])
                result = self.workers[i].perform_wo_projection.remote(candidate_simplex[i])
                shrink_results.append(result)

            for i in range(1, self.num_constraints):
                candidate_projected_simplex[i] = ray.get(shrink_results[i - 1])['pareto_optimal_point']
                candidate_sf_scores[i] = ray.get(shrink_results[i - 1])['sf']
                candidate_dvhs[i] = ray.get(shrink_results[i - 1])['dvh']

        self.simplex = copy.deepcopy(candidate_simplex)
        self.projected_simplex = copy.deepcopy(candidate_projected_simplex)
        self.sf_scores = copy.deepcopy(candidate_sf_scores)
        self.dvhs = copy.deepcopy(candidate_dvhs)

        return self.order_vertices_and_get_returns()



    def sequential_step(self, action):
        ''''''
        alpha = 2*self.simplex_param_actions[action]
        gamma = 4*self.simplex_param_actions[action]
        rho = self.simplex_param_actions[action]
        sigma = self.simplex_param_actions[action]
        print('Alpha:',alpha,'Gamma:',gamma,'Rho:',rho,'Sigma:',sigma)
        #first calc centroid
        centroid = np.zeros(self.num_constraints)
        for i in range(self.num_constraints-1):
            centroid += self.simplex[i]
        centroid = centroid/(self.num_constraints-1)


        '''Reflection'''

        vec = centroid - self.simplex[-1]
        reflection_point = centroid + alpha*vec
        inside_hull_refl = self.in_hull(np.array(self.bound_vertices),reflection_point)
        if inside_hull_refl or not self.bounded_search:
            print('Reflection point before projection: ',reflection_point)
            result_refl = self.workers[-4].perform_wo_projection.remote(reflection_point)


        '''Expansion'''

        vec = centroid - self.simplex[-1]
        expanded_point = centroid + gamma*vec
        inside_hull_exp = self.in_hull(np.array(self.bound_vertices),expanded_point)
        if inside_hull_exp or not self.bounded_search:
            print('Expanded point before projection: ',expanded_point)
            result_exp = self.workers[-3].perform_wo_projection.remote(expanded_point)


        '''Outside Contraction'''

        vec = reflection_point - centroid #self.simplex[-1] - centroid
        contraction_point_oc = centroid + rho*vec
        inside_hull_oc = self.in_hull(np.array(self.bound_vertices),contraction_point_oc)
        if inside_hull_oc or not self.bounded_search:
            print('Outside contraction point before projection: ',contraction_point_oc)
            result_oc = self.workers[-2].perform_wo_projection.remote(contraction_point_oc)


        '''Inside Contraction'''

        vec = self.simplex[-1] - centroid #self.simplex[-1] - centroid
        contraction_point_ic = centroid + rho*vec
        inside_hull_ic = self.in_hull(np.array(self.bound_vertices),contraction_point_ic)
        if inside_hull_ic or not self.bounded_search:
            print('Inside contraction point before projection: ',contraction_point_ic)
            result_ic = self.workers[-1].perform_wo_projection.remote(contraction_point_ic)

        '''Get synchronous results'''
        if inside_hull_refl or not self.bounded_search:
            # evaluate reflection point
            projected_reflection_point = ray.get(result_refl)['pareto_optimal_point']
            reflection_point_sf = ray.get(result_refl)['sf']
            reflection_point_dvh = ray.get(result_refl)['dvh']
            print('Reflection point score:',reflection_point_sf)
            print('Structure means:',ray.get(result_refl)['struct_mean_array'])

            self.pareto_points_list.append(projected_reflection_point)
            self.pareto_points_scores.append(reflection_point_sf)
            self.pareto_points_dvhs.append(reflection_point_dvh)

        if inside_hull_exp or not self.bounded_search:
            # evaluate expanded point
            projected_expanded_point = ray.get(result_exp)['pareto_optimal_point']
            expanded_point_sf = ray.get(result_exp)['sf']
            expanded_point_dvh = ray.get(result_exp)['dvh']
            print('Expanded point score:',expanded_point_sf)
            print('Structure means:',ray.get(result_exp)['struct_mean_array'])

            self.pareto_points_list.append(projected_expanded_point)
            self.pareto_points_scores.append(expanded_point_sf)
            self.pareto_points_dvhs.append(expanded_point_dvh)

        if inside_hull_oc or not self.bounded_search:
            # evaluate contraction point
            projected_contraction_point_oc = ray.get(result_oc)['pareto_optimal_point']
            contraction_point_sf_oc = ray.get(result_oc)['sf']
            contraction_point_dvh_oc = ray.get(result_oc)['dvh']
            print('Outside contraction point score:',contraction_point_sf_oc)
            print('Structure means:',ray.get(result_oc)['struct_mean_array'])

            self.pareto_points_list.append(projected_contraction_point_oc)
            self.pareto_points_scores.append(contraction_point_sf_oc)
            self.pareto_points_dvhs.append(contraction_point_dvh_oc)

        if inside_hull_ic or not self.bounded_search:
            # evaluate contraction point
            projected_contraction_point_ic = ray.get(result_ic)['pareto_optimal_point']
            contraction_point_sf_ic = ray.get(result_ic)['sf']
            contraction_point_dvh_ic = ray.get(result_ic)['dvh']
            print('Inside contraction point score:',contraction_point_sf_ic)
            print('Structure means:',ray.get(result_ic)['struct_mean_array'])

            self.pareto_points_list.append(projected_contraction_point_ic)
            self.pareto_points_scores.append(contraction_point_sf_ic)
            self.pareto_points_dvhs.append(contraction_point_dvh_ic)



        '''Reflection'''
        if inside_hull_refl or not self.bounded_search:
            print('Reflection point is inside the hull')
        else:
            projected_reflection_point = None
            reflection_point_sf = 1e10
            print('Reflection point outside bounded area: ',reflection_point)
        if reflection_point_sf < self.sf_scores[-2] and reflection_point_sf >= self.sf_scores[0]:
            self.simplex[-1] = reflection_point
            self.projected_simplex[-1] = projected_reflection_point
            self.sf_scores[-1] = reflection_point_sf
            self.dvhs[-1] = reflection_point_dvh

            return self.order_vertices_and_get_returns()

        elif reflection_point_sf < self.sf_scores[0]:
            '''Begin Expansion'''
            if inside_hull_exp or not self.bounded_search:
                print('Expansion point is inside the hull')
            else:
                projected_expanded_point = None
                expanded_point_sf = 1e10

            if expanded_point_sf < reflection_point_sf:
                self.simplex[-1] = expanded_point
                self.projected_simplex[-1] = projected_expanded_point
                self.sf_scores[-1] = expanded_point_sf
                self.dvhs[-1] = expanded_point_dvh
                return self.order_vertices_and_get_returns()
            else:
                self.simplex[-1] = reflection_point
                self.projected_simplex[-1] = projected_reflection_point
                self.sf_scores[-1] = reflection_point_sf
                self.dvhs[-1] = reflection_point_dvh
                return self.order_vertices_and_get_returns()

        elif reflection_point_sf >= self.sf_scores[-2] and reflection_point_sf < self.sf_scores[-1]:
            '''Begin Outside Contraction'''
            if inside_hull_oc or not self.bounded_search:
                print('Outside contraction point is inside the hull')
            else:
                projected_contraction_point_oc = None
                contraction_point_sf_oc = 1e10
            if contraction_point_sf_oc < reflection_point_sf: # contraction_point_sf < self.sf_scores[-1]:
                self.simplex[-1] = contraction_point_oc
                self.projected_simplex[-1] = projected_contraction_point_oc
                self.sf_scores[-1] = contraction_point_sf_oc
                self.dvhs[-1] = contraction_point_dvh_oc
                return self.order_vertices_and_get_returns()
            else:
                '''Shrink'''
                results = []
                for i in range(1,self.num_constraints):
                    vec = self.simplex[i] - self.simplex[0]
                    self.simplex[i] = self.simplex[0] + sigma*vec
                    print('Shrink point '+str(i+1)+' before projection: ',self.simplex[i])
                    result = self.workers[i].perform_wo_projection.remote(self.simplex[i])
                    results.append(result)

                for i in range(1,self.num_constraints):
                    self.projected_simplex[i] = ray.get(results[i-1])['pareto_optimal_point']
                    self.sf_scores[i] = ray.get(results[i-1])['sf']
                    self.dvhs[i] = ray.get(results[i-1])['dvh']

                return self.order_vertices_and_get_returns()

        elif reflection_point_sf >= self.sf_scores[-1]:
            '''Begin Inside Contraction'''
            if inside_hull_ic or not self.bounded_search:
                print('Inside contraction point is inside the hull')
            else:
                projected_contraction_point_ic = None
                contraction_point_sf_ic = 1e10
            if contraction_point_sf_ic < self.sf_scores[-1]:# contraction_point_sf < self.sf_scores[-1]:
                self.simplex[-1] = contraction_point_ic
                self.projected_simplex[-1] = projected_contraction_point_ic
                self.sf_scores[-1] = contraction_point_sf_ic
                self.dvhs[-1] = contraction_point_dvh_ic
                return self.order_vertices_and_get_returns()
            else:
                '''Shrink'''
                results = []
                for i in range(1,self.num_constraints):
                    vec = self.simplex[i] - self.simplex[0]
                    self.simplex[i] = self.simplex[0] + sigma*vec
                    print('Shrink point '+str(i+1)+' before projection: ',self.simplex[i])
                    result = self.workers[i].perform_wo_projection.remote(self.simplex[i])
                    results.append(result)

                for i in range(1,self.num_constraints):
                    self.projected_simplex[i] = ray.get(results[i-1])['pareto_optimal_point']
                    self.sf_scores[i] = ray.get(results[i-1])['sf']
                    self.dvhs[i] = ray.get(results[i-1])['dvh']

                return self.order_vertices_and_get_returns()

    def order_vertices_and_get_returns(self):
        '''order vertices'''
        s, ps, gs, dvhs = self.order_simplex_points(self.simplex, self.projected_simplex, self.sf_scores, self.dvhs)
        self.simplex = s
        self.projected_simplex = ps
        self.sf_scores = gs
        self.dvhs = dvhs
        # self.projected_simplex_opt_weights = self.get_projected_simplex_opt_weights()

        self.simplex_hist.append(copy.deepcopy(self.simplex))
        self.projected_simplex_hist.append(copy.deepcopy(self.projected_simplex))
        self.sf_hist.append(copy.deepcopy(self.sf_scores))
        self.dvhs_hist.append(copy.deepcopy(self.dvhs))

        print('-----After ordering-----')
        print('Simplex:', self.simplex)
        # print('Projected simplex:', self.projected_simplex)
        print('Scores:', self.sf_scores)
        # print('Opt weights:', self.projected_simplex_opt_weights)

        '''calculate distances'''
        distances = []
        for i in range(1, self.num_constraints):
            # distances.append(np.max(np.abs(self.projected_simplex[i] - self.projected_simplex[0])))
            distances.append(np.max(np.abs(self.simplex[i] - self.simplex[0])))
        max_dist = np.max(distances)
        print('Distances: ', distances)

        score_diff = []
        for i in range(1, self.num_constraints):
            score_diff.append(np.abs(self.sf_scores[i] - self.sf_scores[0]))
        '''check if algo should terminate'''
        if np.max(score_diff) <= self.ftol or max_dist <= self.ctol:
            done, out_dict = self.terminate()
        elif self.current_step == self.max_iter - 1:
            done, out_dict = self.terminate()
        else:
            done = False
            out_dict = {}

        obs = self.get_obs(self.simplex, self.projected_simplex, self.sf_scores)
        self.current_step += 1
        print('Current step:', self.current_step)
        print('Current Patient Dir:', self.patientDir)

        return obs, -100, done, {'num_steps_run': self.current_step,
                                 'out_dict': out_dict,
                                 'patient_dir': self.patientDir,
                                 'bound_vertices': self.bound_vertices,
                                 'bound_sf_scores': self.bound_sf_scores,
                                 'final_simplex': self.simplex,
                                 'initial_simplex': self.initial_simplex,
                                 'initial_scores': self.initial_scores,
                                 'initial_projected_simplex': self.initial_projected_simplex,
                                 'final_projected_simplex': self.projected_simplex,
                                 'final_sf_scores': self.sf_scores,
                                 'final_dvhs': self.dvhs,
                                 'couch_angle_list': self.couch_angle_list,
                                 'gantry_angle_list': self.gantry_angle_list,
                                 'pareto_points_list': self.pareto_points_list,
                                 'bound_weights_list': self.bound_weights_list}

    def terminate(self):
        done = True
        '''project and order vertices'''
        s, ps, gs, dvhs = self.order_simplex_points(self.simplex, self.projected_simplex, self.sf_scores, self.dvhs)
        self.simplex = s
        self.projected_simplex = ps
        self.sf_scores = gs
        self.dvhs = dvhs
        # self.projected_simplex_opt_weights = self.get_projected_simplex_opt_weights()

        self.simplex_hist.append(copy.deepcopy(self.simplex))
        self.projected_simplex_hist.append(copy.deepcopy(self.projected_simplex))
        self.sf_hist.append(copy.deepcopy(self.sf_scores))
        self.dvhs_hist.append(copy.deepcopy(self.dvhs))

        print('-----After ordering-----')
        print('Simplex:', self.simplex)
        # print('Projected simplex:', self.projected_simplex)
        print('Scores:', self.sf_scores)
        # print('Opt weights:', self.projected_simplex_opt_weights)
        ''''''
        out_point_idx = 0
        out_dict = {}
        out_dict['out_point'] = copy.deepcopy(self.projected_simplex[out_point_idx])
        out_dict['out_sf'] = copy.deepcopy(self.sf_scores[out_point_idx])
        out_dict['out_dvh'] = copy.deepcopy(self.dvhs[out_point_idx])

        out_dict['out_point_opt_weights'] = copy.deepcopy(self.projected_simplex[out_point_idx])

        for actor in self.workers:
            ray.kill(actor)
        return done, out_dict

    def get_projected_simplex_opt_weights(self):
        projected_simplex_opt_weights = []
        for point in self.simplex:
            opt_weights = self.get_opt_weights_from_dose_constraint_vals(point, self.bound_vertices,
                                                                         self.bound_weights_list)
            projected_simplex_opt_weights.append(opt_weights)

        return projected_simplex_opt_weights

    def get_opt_weights_from_dose_constraint_vals(self,start_coord, bound_vertices_list, bound_weights_list):
        mass_weights = self.mass_weights_given_point(bound_vertices_list, start_coord)
        point_pieces = []
        for i, w in enumerate(bound_weights_list):
            point_pieces.append(mass_weights[i] * w)
        opt_weights = np.array(np.sum(np.array(point_pieces), 0)) / np.sum(mass_weights)
        return opt_weights

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




    def get_obs(self,simplex_points,projected_simplex_points,sf_scores):
        '''
        Expecting all list inputs
        '''
        simplex_obs = np.concatenate(simplex_points,axis = -1)
        projected_obs = np.concatenate(projected_simplex_points,axis = -1)
        bound_obs = np.concatenate(self.bound_vertices,axis = -1)
        obs = np.concatenate([simplex_obs,projected_obs,bound_obs],axis = -1)
        obs = np.concatenate([obs,sf_scores],axis = -1)
        return obs

    def order_simplex_points(self,unordered_simplex,unordered_projected_simplex,unordered_sf_scores,unordered_dvhs):
        '''
        Expecting all list inputs
        '''
        # order simplex points, projected simplex points, and sf scores 
        simplex = [point for score, point in sorted(zip(unordered_sf_scores,unordered_simplex), key=lambda pair: pair[0])]
        projected_simplex = [point for score, point in sorted(zip(unordered_sf_scores,unordered_projected_simplex), key=lambda pair: pair[0])]
        sf_scores = [score for score, point in sorted(zip(unordered_sf_scores,unordered_projected_simplex), key=lambda pair: pair[0])]
        dvhs = [point for score, point in sorted(zip(unordered_sf_scores,unordered_dvhs), key=lambda pair: pair[0])]
        return simplex, projected_simplex, sf_scores, dvhs

    def get_end_point(self,rayDirection,rayPoint):
        from scipy.optimize import minimize, Bounds, NonlinearConstraint
        psi = rayPoint
        unit_ray = rayDirection/np.linalg.norm(rayDirection)
        end_point_f = lambda x: unit_ray*x+psi
        obj_f = lambda x: np.linalg.norm(unit_ray*x+psi)
        ineq_constr = lambda x: np.min(unit_ray*x+psi)
        constraints = NonlinearConstraint(ineq_constr,0,np.inf)
        x0 = np.array([0])
        res = minimize(obj_f,x0,constraints=constraints)
        x_star = res.x

        end_point = end_point_f(x_star)-np.min(end_point_f(x_star))
        print('Nonnegative End point:',end_point)

        return end_point


    # def mass_weights_given_point(self,vertex_coords,point_coords):
    #     a = np.stack(list(vertex_coords), axis=-1)
    #     b = point_coords
    #     weights = np.linalg.pinv(a)@b
    #     return weights

    # def in_hull(self, points, x, tol = 1e-3):
    #     from scipy.spatial import ConvexHull
    #     centroid = np.mean(points, 0)
    #     points_hull_array = np.array(list(points) + [centroid])
    #     hull = ConvexHull(points_hull_array, qhull_options='QJ')
    #     return all([np.dot(eq[:-1], x) + eq[-1] <= tol for eq in hull.equations])

    def in_hull(self, points, x):
        from scipy.optimize import linprog
        points = np.array(points)
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success

    def mass_weights_given_point(self, bound_vertices_list, point_coords):
        from scipy.optimize import minimize, Bounds, NonlinearConstraint
        a = np.stack(list(bound_vertices_list), axis=-1)
        b = point_coords
        obj_f = lambda x: np.linalg.norm(a @ x - b)
        c = lambda x: x
        constraints = NonlinearConstraint(c, 0., 1.)
        x0 = np.multiply(b, 0)
        res = minimize(obj_f, x0, constraints=constraints)
        weights = res.x
        return np.maximum(np.array(weights) / np.sum(weights),0)

    def find_closest_point_in_hull(self,vertex_coords,point_coords):
        weights = self.mass_weights_given_point(vertex_coords,point_coords)
        point_pieces = []
        for i,v in enumerate(vertex_coords):
            point_pieces.append(weights[i]*v)
        return np.array(np.sum(np.array(point_pieces),0))/np.sum(weights)


    def get_back_projection_end_point(self, rayDirection, rayPoint):
        from scipy.optimize import minimize, Bounds, NonlinearConstraint
        from scipy.spatial import ConvexHull
        psi = rayPoint
        unit_ray = rayDirection / np.linalg.norm(rayDirection)
        centroid_bounds = np.mean(self.bound_vertices, 0)
        bounds_hull_array = np.array(self.bound_vertices + [centroid_bounds])
        hull = ConvexHull(bounds_hull_array,qhull_options = 'QJ')
        end_point_f = lambda x: unit_ray * x + psi
        obj_f = lambda x: np.linalg.norm(centroid_bounds - (unit_ray * x + psi))
        # constraints
        c1 = lambda x: np.min(unit_ray * x + psi)
        c2 = lambda x: x
        constraints = [NonlinearConstraint(c1, 0., np.inf),
                       NonlinearConstraint(c2, 0., np.inf)]
        for eq in hull.equations:
            c = lambda x: -1*(np.dot(eq[:-1], unit_ray * x + psi) + eq[-1])
            constraints.append(NonlinearConstraint(c, 0., np.inf))
        # ----------
        x0 = np.array([0])
        res = minimize(obj_f, x0, constraints=constraints)
        x_star = res.x
        end_point = self.find_closest_point_in_hull(self.bound_vertices,end_point_f(x_star))
        print('Backproj End point:', end_point)
        return end_point
