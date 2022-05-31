from open_kbp_dataloader import *
import glob
import os
import nibabel as nib
import numpy as np
import copy

class MultiChannelDataset(MyDataset):
    def __init__(self, num_samples_per_epoch, phase):
        self.phase = phase # train, val, query, database
        self.transform = {'train': self.train_transform,
                          'val': self.val_transform,
                          'query': self.val_transform,
                          'database': self.val_transform}

        train_folder_pth = './provided-data/train-pats/'
        val_folder_pth = './provided-data/validation-pats/'
        query_folder_pth = './provided-data/query-pats/'
        database_folder_pth = './provided-data/database-pats/'

        train_nifti_filenames = [os.path.join(train_folder_pth, 'pt_') + str(i) for i in range(1, 201)]
        train_nifti_filenames += sorted([i for i in glob.glob(os.path.join(train_folder_pth, '*.nii.gz'))])
        val_nifti_filenames = [os.path.join(val_folder_pth, 'pt_') + str(i) for i in range(201, 241)]
        val_nifti_filenames += sorted([i for i in glob.glob(os.path.join(val_folder_pth, '*.nii.gz'))])
        query_nifti_filenames = [i for i in glob.glob(os.path.join(query_folder_pth, 'pt_*')) if os.path.isdir(i)]
        query_nifti_filenames += sorted([i for i in glob.glob(os.path.join(query_folder_pth, '*.nii.gz'))])
        database_nifti_filenames = [i for i in glob.glob(os.path.join(database_folder_pth, 'pt_*')) if os.path.isdir(i)]
        database_nifti_filenames += sorted([i for i in glob.glob(os.path.join(database_folder_pth, '*.nii.gz'))])
        self.list_case_id = {'train': train_nifti_filenames,
                             'val': val_nifti_filenames,
                             'query': query_nifti_filenames,
                             'database': database_nifti_filenames}[phase]

        print('Number of cases in {} phase: {}'.format(phase,len(self.list_case_id)))

        if self.phase == 'train':
            random.shuffle(self.list_case_id)

        if num_samples_per_epoch == None:
            self.num_samples_per_epoch = len(self.list_case_id)
        else:
            self.num_samples_per_epoch = num_samples_per_epoch

        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            patient_dir = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            patient_dir = self.list_case_id[new_index_]

        self.patient_dir = patient_dir

        mc_img = self.load_img(patient_dir, load_dose=False)
        dose_img = self.load_img(patient_dir, load_dose=True)

        list_images = [mc_img,  # input1
                       mc_img]  # input2

        list_images = self.transform[self.phase](list_images)

        if self.phase == 'query' or self.phase == 'database':
            return list_images + [self.patient_dir]
        else:
            return list_images



    def load_img(self,filepath, load_dose=False):
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
        else: #assume file is nifti
            mc_img = self.load_nii(patient_dir,load_dose=load_dose)
            assert len(mc_img.shape) == 4, print(mc_img.shape)
            mc_img = mc_img.transpose([3, 0, 1, 2])
            mc_img = self.remove_couch_from_ct(mc_img)
            mc_img = self.remove_oars(mc_img)

        # mc_img = self.convert_ct_to_seg(mc_img)
        mc_img[0, :, :, :] = (mc_img[0, :, :, :] - np.min(mc_img[0, :, :, :])) / (
                             np.max(mc_img[0, :, :, :]) - np.min(mc_img[0, :, :, :])) # normalize ct
        return mc_img.astype(np.float32)

    def load_nii(self,patient_dir,min_val = -200, max_val = 200, load_dose = False):
        mc_img = np.squeeze(nib.load(patient_dir).dataobj)
        assert len(mc_img.shape)==4 # dims [x,y,z,c]
        out = np.zeros(mc_img.shape[:-1]+(2,))
        seg = copy.deepcopy(mc_img[:,:,:,1])
        CT = np.round(mc_img[:,:,:,0])

        if load_dose:
            dose = copy.deepcopy(mc_img[:,:,:,2])
            out[:,:,:,0] = dose
            out[:,:,:,1] = seg

        else:
            CT = np.clip(CT, a_min=min_val, a_max=max_val).astype(np.float32)
            # CT = (CT-min_val)/(max_val-min_val)
            out[:,:,:,0] = CT
            out[:,:,:,1] = seg

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

    def convert_ct_to_seg(self, mc_img, window_center = -200):
        '''
        :param mc_img: [C,X,Y,Z]
        :return:
        '''
        ct = mc_img[0,:,:,:]
        ct = np.clip(ct, window_center, window_center+1)
        ct = np.round(ct)
        ct = ct-window_center
        mc_img[0,:,:,:] = ct
        return mc_img

def get_loader(train_bs=1,
               val_bs=1,
               train_num_samples_per_epoch=200,
               val_num_samples_per_epoch=40,
               num_works=0):
    train_dataset = MultiChannelDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = MultiChannelDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    training_data_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=False)
    val_data_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)

    return training_data_loader, val_data_loader