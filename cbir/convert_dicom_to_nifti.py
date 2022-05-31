import os
import glob
import matlab.engine
from scipy.ndimage import zoom
import numpy as np
import nibabel as nib
import sys

sys.path.append('..')

from head_and_neck_src import matRad_wo_hn_env
from prostate_src import matRad_wo_prostate_env

def label_encoding(struct_names,labels,masks,max_label=4):
    '''
    :param struct_names: list of structure names
    :param labels: list of integer labels (background has label of 0)
    :param masks: list of binary arrays
    :param max_label: int largest label value
    :return: label encoded array
    '''
    dims = masks[0].shape
    out = np.zeros(dims)
    for l in range(1,max_label+1):
        mask_l = np.zeros(dims)
        matching_ids = np.where(np.array(labels)==l)[0]
        for i in matching_ids: # assumes labels is 1d
            mask_l += masks[i]
        mask_l = np.clip(mask_l,0,1)*l
        out += mask_l
        out = np.clip(out,0,l)
    return np.round(out)


if __name__ == "__main__":
    os.environ['matRad'] = '../matRad-dev_VMAT/'

    output_dim = [128,128,128]
    data_parent_dir = './provided-data/dicom_data/'
    output_parent_dir = './provided-data/nifti_data/'
    body_regions = ['prostate','head_and_neck']

    ''''''
    for body_region in body_regions:
        data_ext = body_region+'_dicoms'
        data_folder = os.path.join(data_parent_dir,data_ext)
        output_ext = body_region+'_niftis'
        output_folder = os.path.join(output_parent_dir,output_ext)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        case_list = sorted([i for i in glob.glob(os.path.join(data_folder, '*')) if os.path.isdir(i)])
        num_cases = len(case_list)
        print('Num cases: ',num_cases)

        for sample_case_num in range(num_cases):

            ctDir = os.path.join(case_list[sample_case_num], 'CT')
            rtStDir = os.path.join(case_list[sample_case_num], 'RTst')
            rtDoseDir = os.path.join(case_list[sample_case_num],'RTDOSE')
            print('CT path:',ctDir)
            print('RTst path:',rtStDir)
            print('RTDOSE path:',rtDoseDir)

            try:
                if body_region == body_regions[0]: #prostate
                    struct_names = ['rectum','bladder','fh r','fh l','ptv','body']
                    labels = [2,2,2,2,4,1]
                    masks = []
                    matrad_env = matRad_wo_prostate_env.MatRadMetaOpt()
                    # matrad_env.reset_pops(ctDir,
                    #                       rtStDir,
                    #                       couch_angle_list=[0],
                    #                       gantry_angle_list=[0],
                    #                       high_res=True,
                    #                       acceptable_iter=200,
                    #                       acceptable_constr_viol_tol=0.0001,
                    #                       ipopt_max_iters=200, )

                    _ = matrad_env.get_ptv_stats_from_treatment_plan(
                        ctDir, rtStDir, rtDoseDir, num_fracs = 1)
                elif body_region == body_regions[1]: #head and neck
                    struct_names = ['ptv_52_final','ptv_56_final','ptv_70_final','spinal_cord','brainstem','parotid_r',
                                    'parotid_l','oral_cavity','larynx','body']
                    labels = [3,3,4,2,2,2,2,2,2,1]
                    masks = []
                    matrad_env = matRad_wo_hn_env.MatRadMetaOpt()
                    # matrad_env.reset_pops(ctDir,
                    #                       rtStDir,
                    #                       couch_angle_list=[0],
                    #                       gantry_angle_list=[0],
                    #                       high_res=True,
                    #                       acceptable_iter=200,
                    #                       acceptable_constr_viol_tol=0.0001,
                    #                       ipopt_max_iters=200, )

                    _ = matrad_env.get_ptv_stats_from_treatment_plan(
                        ctDir, rtStDir, rtDoseDir, num_fracs = 1)
                assert len(struct_names) == len(labels)
                ct_HU = np.squeeze(matrad_env.eng.eval('ct.cubeHU{:}'))
                curr_dim = list(np.squeeze(matrad_env.eng.eval('ct.cubeDim')))
                zoom_factors = np.array(output_dim) / np.array(curr_dim)
                resized_ct_HU = zoom(ct_HU, zoom_factors, order = 0)

                # load and resize segmentations
                for row_num,s in enumerate(struct_names):
                    print('loading data for ',str(matrad_env.eng.eval('cst{'+str(row_num+1)+',2}')))
                    print('corresponding structure name: ',s)
                    mask_data = np.squeeze(matrad_env.eng.eval('volVecToMask(ct,cst{'+str(row_num+1)+',4}{:})'))
                    resized_mask = np.round(zoom(mask_data, zoom_factors, order = 0))
                    masks.append(resized_mask)

                # load and resize dose distribution
                dose_distr = np.squeeze(matrad_env.eng.eval('resultGUI.physicalDose'))
                resized_dose_distr = zoom(dose_distr, zoom_factors, order=0)

                # apply label encoding
                out_mask = label_encoding(struct_names,labels,masks)
                assert out_mask.shape == resized_ct_HU.shape, print(out_mask.shape,resized_ct_HU.shape)
                assert np.max(out_mask) == np.max(labels), print(np.max(out_mask),np.max(labels))

                output_data = np.zeros(resized_ct_HU.shape+(3,))
                output_data[:,:,:,0] = resized_ct_HU
                output_data[:,:,:,1] = out_mask
                output_data[:,:,:,2] = resized_dose_distr

                nii_img = nib.Nifti1Image(output_data, affine=None)
                output_file_path = os.path.join(output_folder, case_list[sample_case_num].split('/')[-1] + '.nii.gz')
                print('Saving nifti to path:',output_file_path)
                nib.save(nii_img, output_file_path)

            except:
                pass


