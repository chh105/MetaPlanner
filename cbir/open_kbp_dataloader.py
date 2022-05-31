# This file is adapted from https://github.com/LSL000UD/RTDosePrediction/blob/main/RTDosePrediction/Src/DataPrepare/prepare_OpenKBP_C3D.py
import pandas as pd
import numpy as np
import os
import random
import copy

import torch
import torch.utils.data as data


class MyDataset(data.Dataset):
    def __init__(self, num_samples_per_epoch, phase):
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': self.train_transform, 'val': self.val_transform}

        self.list_case_id = {'train': ['./provided-data/train-pats/pt_' + str(i) for i in range(1, 201)],
                             'val': ['./provided-data/validation-pats/pt_' + str(i) for i in range(201, 241)]}[phase]

        random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            patient_dir = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            patient_dir = self.list_case_id[new_index_]

        ct = self.load_ct(patient_dir)
        ct = np.expand_dims(ct, axis=0)
        dose_dist = self.load_dose(patient_dir)
        dose_dist = np.expand_dims(dose_dist, axis=0)
        list_images = [ct, #input
                       ct] #output

        list_images = self.transform[self.phase](list_images)

        return list_images

    def __len__(self):
        return self.num_samples_per_epoch

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

    # To tensor, images should be C*Z*H*W
    def to_tensor(self, list_images):
        for image_i in range(len(list_images)):
            list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
        return list_images

    def train_transform(self, list_images):

        list_images = self.random_flip(list_images)
        list_images = self.random_transpose(list_images)


        list_images = self.to_tensor(list_images)
        return list_images

    def val_transform(self, list_images):
        list_images = self.to_tensor(list_images)
        return list_images

    def random_flip(self, list_images, p = 0.5):
        axes = [1, 2, 3]  # channel is dim 0
        for ax in axes:
            val = np.random.uniform(0, 1)
            if val >= p:
                for image_i in range(len(list_images)):
                    img = np.flip(copy.deepcopy(list_images[image_i]), axis = ax)
                    list_images[image_i] = img

        return list_images


    def random_transpose_single_img(self, image):
        transposes = [[0, 1, 2, 3],[0, 1, 3, 2],[0, 2, 1, 3],[0, 2, 3, 1],[0, 3, 2, 1],[0, 3, 1, 2]]  # channel is dim 0
        transf_img = copy.deepcopy(image)

        val = int(np.random.uniform(0, len(transposes)))
        transf_img = copy.deepcopy(transf_img).transpose(transposes[val])

        return transf_img

    def random_transpose(self, list_images):
        transposes = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 2, 1],
                      [0, 3, 1, 2]]  # channel is dim 0
        val = int(np.random.uniform(0, len(transposes)))
        transform = transposes[val]
        for i in range(len(list_images)):
            list_images[i] = copy.deepcopy(list_images[i]).transpose(transform)

        return list_images



def get_loader(train_bs=1,
               val_bs=1,
               train_num_samples_per_epoch=200,
               val_num_samples_per_epoch=40,
               num_works=0):
    train_dataset = MyDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = MyDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    training_data_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=False)
    val_data_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)

    return training_data_loader, val_data_loader

