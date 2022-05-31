import glob
import os
import copy
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

class ManualDataloader():
    def __init__(self, folder_pth = './provided-data/unlabelled_data/',
                 output_folder = ''):
        if not output_folder:
            output_folder = os.path.join(folder_pth, 'labels')
        self.folder_pth = folder_pth
        self.output_folder = output_folder
        filenames = [i for i in glob.glob(os.path.join(folder_pth, 'pt_*')) if os.path.isdir(i)]
        filenames += sorted([i for i in glob.glob(os.path.join(folder_pth, '*.nii.gz'))])
        self.list_case_id = copy.deepcopy(filenames)
        print('Num cases:',len(filenames))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print('Creating folder for storing labels...')

    '''Helper Functions'''
    def normalize_csv_ct(self, CT, shift=-1024, min_val=-200, max_val=200):
        CT = CT + shift
        CT = np.clip(CT, a_min=min_val, a_max=max_val).astype(np.float32)
        # CT = (CT-min_val)/(max_val-min_val)
        return CT

    def load_csv_file(self, file_name):
        # Load the file as a csv
        loaded_file_df = pd.read_csv(file_name, index_col=0)

        # If the csv is voxel dimensions read it with numpy
        if 'voxel_dimensions.csv' in file_name:
            loaded_file = np.loadtxt(file_name)
        # Check if the data has any values
        elif loaded_file_df.isnull().values.any():
            # Then the data is a vector, which we assume is for a mask of ones
            loaded_file = np.array(loaded_file_df.index).squeeze()
        else:
            # Then the data is a matrix of indices and data points
            loaded_file = {'indices': np.array(loaded_file_df.index).squeeze(),
                           'data': np.array(loaded_file_df['data']).squeeze()}

        return loaded_file

    def load_ct(self, patient_dir, min_val=-1024, max_val=1500):
        CT_csv = self.load_csv_file(patient_dir + '/ct.csv')
        CT = np.zeros((128, 128, 128), dtype=np.int16)
        indices_ = np.int64(CT_csv['indices'])
        data_ = np.int16(CT_csv['data'])
        np.put(CT, indices_, data_)

        CT = self.normalize_csv_ct(CT)

        # Data in OpenKBP dataset is (x, y, -z) output is (x, y, z)
        CT = CT[:, :, ::-1]
        return CT

    def load_dose(self, patient_dir):
        dose_csv = self.load_csv_file(patient_dir + '/dose.csv')
        dose = np.zeros((128, 128, 128), dtype=np.float32)
        indices_ = np.int64(dose_csv['indices'])
        data_ = np.float32(dose_csv['data'])
        np.put(dose, indices_, data_)

        dose = dose[:, :, ::-1]
        return dose

    def label_encoding(self, struct_names, labels, masks, max_label=3):
        '''
        :param struct_names: list of structure names
        :param labels: list of integer labels (background has label of 0)
        :param masks: list of binary arrays
        :param max_label: int largest label value
        :return: label encoded array
        '''
        assert len(struct_names) == len(labels)
        assert len(struct_names) == len(masks)
        dims = masks[0].shape
        out = np.zeros(dims)
        for l in range(1, max_label + 1):
            mask_l = np.zeros(dims)
            matching_ids = np.where(np.array(labels) == l)[0]
            for i in matching_ids:  # assumes labels is 1d
                mask_l += masks[i]
            mask_l = np.clip(mask_l, 0, 1) * l
            out += mask_l
            out = np.clip(out, 0, l)
        return np.round(out)

    def load_oars(self, patient_dir):
        # labels = [3,2,2,1,1,1,1,1,1,1]
        # structure_names = ['PTV70',
        #                        'PTV63',
        #                        'PTV56',
        #                        'Brainstem',
        #                        'SpinalCord',
        #                        'RightParotid',
        #                        'LeftParotid',
        #                        'Esophagus',
        #                        'Larynx',
        #                        'Mandible']
        labels = [3,2,2,1,1,1,1,1]
        structure_names = ['PTV70',
                               'PTV63',
                               'PTV56',
                               'Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Larynx',]
        mask_list = []
        for structure_name in structure_names:
            structure_csv_file = patient_dir + '/' + structure_name + '.csv'
            if os.path.exists(structure_csv_file):
                structure_csv = self.load_csv_file(structure_csv_file)
                structure = np.zeros((128, 128, 128), dtype=np.uint8)
                np.put(structure, structure_csv, np.uint8(1))

                structure = structure[:, :, ::-1]
                mask_list.append(structure)
            else:
                structure = np.zeros((128, 128, 128), dtype=np.uint8)
                mask_list.append(structure)

        out_mask = self.label_encoding(structure_names, labels, mask_list)

        return out_mask

    def load_img(self, filepath, load_dose=False):
        patient_dir = filepath
        # if path is a dir with csv files
        if os.path.isdir(patient_dir):
            ct = self.load_ct(patient_dir)
            seg = self.load_oars(patient_dir)
            dose = self.load_dose(patient_dir)
            mc_img = np.zeros(seg.shape + (2,))
            if load_dose:
                mc_img[:, :, :, 0] = dose
                mc_img[:, :, :, 1] = seg
            else:
                mc_img[:, :, :, 0] = ct
                mc_img[:, :, :, 1] = seg
            mc_img = mc_img.transpose([3, 0, 1, 2])
        else:  # assume file is nifti
            mc_img = self.load_nii(patient_dir, load_dose=load_dose)
            assert len(mc_img.shape) == 4, print(mc_img.shape)
            mc_img = mc_img.transpose([3, 0, 1, 2])
            mc_img = self.remove_couch_from_ct(mc_img)
            mc_img = self.remove_oars(mc_img)

        # mc_img = self.convert_ct_to_seg(mc_img)
        mc_img[0, :, :, :] = (mc_img[0, :, :, :] - np.min(mc_img[0, :, :, :])) / (
                np.max(mc_img[0, :, :, :]) - np.min(mc_img[0, :, :, :]))  # normalize ct
        return mc_img.astype(np.float32)


    def load_nii(self, patient_dir, min_val=-200, max_val=200, load_dose=False):
        mc_img = np.squeeze(nib.load(patient_dir).dataobj)
        assert len(mc_img.shape) == 4  # dims [x,y,z,c]
        out = np.zeros(mc_img.shape[:-1] + (2,))
        seg = copy.deepcopy(mc_img[:, :, :, 1])
        CT = np.round(mc_img[:, :, :, 0])

        if load_dose:
            dose = copy.deepcopy(mc_img[:, :, :, 2])
            out[:, :, :, 0] = dose
            out[:, :, :, 1] = seg

        else:
            CT = np.clip(CT, a_min=min_val, a_max=max_val).astype(np.float32)
            # CT = (CT-min_val)/(max_val-min_val)
            out[:, :, :, 0] = CT
            out[:, :, :, 1] = seg

        return out


    def remove_couch_from_ct(self,mc_img, body_label = 1): # [c,x,y,z]
        '''
        Removes couch from ct (does not apply to csv data)
        :param mc_img: [C,X,Y,Z]
        :return:
        '''
        masked_ct = np.zeros(mc_img.shape[1:])
        x,y,z = np.where(mc_img[1,:,:,:] >= body_label)
        nx,ny,nz = np.where(mc_img[1,:,:,:] == 0)
        masked_ct[x,y,z] = copy.deepcopy(mc_img[0,:,:,:])[x,y,z]
        masked_ct[nx,ny,nz] = np.min(mc_img[0,:,:,:])
        mc_img[0,:,:,:] = masked_ct
        return mc_img


    def remove_oars(self, mc_img, oar_val = 1, amin=0):
        '''
        Removes the oar segmentations from second channel of image
        :param mc_img: [C,X,Y,Z]
        :param amin: value of background (0)
        :param amax: value of most important structure. (4-1=3) is primary ptv. (3-1=2) is non-primary ptvs

        :return:
        '''
        segmentations = mc_img[1,:,:,:]
        amax = np.max(segmentations)
        assert amax == 4
        segmentations = np.clip(segmentations - oar_val, amin, amax - oar_val)
        mc_img[1,:,:,:] = segmentations
        return mc_img

    def get_middle_factors(self,x):
        factors = []
        for i in range(1, x + 1):
            if x % i == 0:
                factors.append(i)

        id = len(factors) // 2 - 1
        factor1 = factors[id]
        factor2 = x / factor1
        return [int(factor1), int(factor2)]

    def plot_img_and_get_label(self,num_plots = 50):
        print('----->Label identifier (0/1, 0/1, 0/1, 0/1/2) has following properties:')
        print('Body site is [0: prostate/1: h&n]')
        print('Number of ptvs is [0: 1 ptv/1: >1 ptv]')
        print('Primary ptv size is [0: small/1: large]')
        print('Primary ptv location is [0: left/1: right/2: center/3: both l & r]')
        fp_list = copy.deepcopy(self.list_case_id)
        img = self.load_img(fp_list[0], load_dose=False)
        num_imgs = len(fp_list)
        num_slices = img[0].shape[-1]
        slice_spacing = np.round(np.linspace(0,num_slices-1,num_plots))
        seg_cmap = mpl.cm.gnuplot
        bounds = [0.5,1.5,2.5,3.5]
        norm = mpl.colors.BoundaryNorm(bounds,seg_cmap.N)
        r,c = self.get_middle_factors(num_plots)

        for fp in fp_list:
            fig, axs = plt.subplots(r,c)
            for j in range(len(slice_spacing)):
                img = self.load_img(fp, load_dose=False)

                m,n = np.unravel_index(j,(r,c))
                axs[m,n].imshow(img[0,:,:,int(slice_spacing[j])],
                                cmap='gray',vmin=0,vmax=1)
                axs[m,n].imshow(img[1,:,:,int(slice_spacing[j])],
                                cmap=seg_cmap,norm=norm,vmin=0,vmax=3,alpha=0.5)

                axs[m,n].set_xticks([])
                axs[m,n].set_yticks([])

            fig.suptitle('Filepath: '+fp)
            fig.canvas.draw()
            plt.pause(0.1)
            print('Filepath:',fp)
            label_id = input('Enter label id (e.g. 1112):')
            print('Retrieved label id is',label_id)
            out_fp = self.write_label_file(fp,label_id)
            print('Writing label to path:',out_fp)
            plt.close(fig)

    def write_label_file(self,input_filename,label_id):
        output_filename = os.path.join(self.output_folder,input_filename.split('.nii.gz')[0].split('/')[-1]+'.txt')
        with open(output_filename,'w') as f:
            f.write(str(label_id))
        return output_filename

if __name__ == "__main__":
    m_loader = ManualDataloader()
    m_loader.plot_img_and_get_label()