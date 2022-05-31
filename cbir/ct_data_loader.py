from open_kbp_dataloader import *
import glob
import os
import nibabel as nib
import numpy as np

class CTDataset(MyDataset):
    def __init__(self, num_samples_per_epoch, phase):
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': self.train_transform, 'val': self.val_transform}

        train_folder_pth = './provided-data/train-pats/'
        val_folder_pth = './provided-data/validation-pats/'

        train_nifti_filenames = [os.path.join(train_folder_pth, 'pt_') + str(i) for i in range(1, 201)]
        train_nifti_filenames += sorted([i for i in glob.glob(os.path.join(train_folder_pth, '*.nii.gz'))])
        val_nifti_filenames = [os.path.join(val_folder_pth, 'pt_') + str(i) for i in range(201, 241)]
        val_nifti_filenames += sorted([i for i in glob.glob(os.path.join(val_folder_pth, '*.nii.gz'))])
        self.list_case_id = {'train': train_nifti_filenames,
                             'val': val_nifti_filenames}[phase]

        print('Number of cases in {} phase: {}'.format(phase,len(self.list_case_id)))

        random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            patient_dir = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            patient_dir = self.list_case_id[new_index_]

        # if path is a dir with csv files
        if os.path.isdir(patient_dir):
            ct = self.load_ct(patient_dir)
            ct = np.expand_dims(ct, axis=0)
            dose_dist = self.load_dose(patient_dir)
            dose_dist = np.expand_dims(dose_dist, axis=0)
            list_images = [ct, #input
                           ct] #output

            list_images = self.transform[self.phase](list_images)
        # else (nifti)
        else:
            ct = self.load_nii(patient_dir)
            ct = np.expand_dims(ct, axis=0)
            list_images = [ct,  # input
                           ct]  # output

            list_images = self.transform[self.phase](list_images)
        return list_images

    def load_nii(self,patient_dir,min_val = -1024, max_val = 1500):
        CT = np.squeeze(nib.load(patient_dir).dataobj).astype(np.int16)
        assert len(CT.shape)==3
        CT = np.clip(CT, a_min=min_val, a_max=max_val)
        CT = CT.astype(np.float32) / 1000.

        return CT

def get_loader(train_bs=1,
               val_bs=1,
               train_num_samples_per_epoch=200,
               val_num_samples_per_epoch=40,
               num_works=0):
    train_dataset = CTDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = CTDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    training_data_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=False)
    val_data_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)

    return training_data_loader, val_data_loader