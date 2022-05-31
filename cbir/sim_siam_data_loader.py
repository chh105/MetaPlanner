from multi_channel_data_loader import *
import glob
import os
import nibabel as nib
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt

class SimSiamDataset(MultiChannelDataset):
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

        train_nifti_filenames = [i for i in glob.glob(os.path.join(train_folder_pth, 'pt_*')) if os.path.isdir(i)]
        train_nifti_filenames += sorted([i for i in glob.glob(os.path.join(train_folder_pth, '*.nii.gz'))])
        val_nifti_filenames = [i for i in glob.glob(os.path.join(val_folder_pth, 'pt_*')) if os.path.isdir(i)]
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

        '''sort file names for triplet loss
        print('----->Label identifier (0/1, 0/1, 0/1, 0/1/2) has following properties:')
        print('Body site is [0: prostate/1: h&n]')
        print('Number of ptvs is [0: 1 ptv/1: >1 ptv]')
        print('Primary ptv size is [0: small/1: large]')
        print('Primary ptv location is [0: left/1: right/2: center/3: both l & r]')
        '''
        self.id_dim = [2, 2, 2, 4]
        self.num_classes = np.prod(self.id_dim)
        self.categorized_filenames = {}
        for i in range(self.num_classes):
            self.categorized_filenames[str(i)] = []


        for i in range(len(self.list_case_id)):
            filepath = copy.deepcopy(self.list_case_id[i])
            category = self.get_category_of_file(filepath)
            self.categorized_filenames[str(category)].append(copy.deepcopy(self.list_case_id[i]))


    def get_category_of_file(self, filepath, ext = '.nii.gz'):
        fp = filepath.split(ext)[0]
        label_file = fp.split('/')[:-1]+['labels']+[fp.split('/')[-1]]
        label_file = '/'.join(label_file) + '.txt'
        assert os.path.exists(label_file), 'Label file "'+label_file+'" does not exist'
        with open(label_file) as f:
            lines = f.readlines()
        label_id = lines[0].replace('\n','')
        label_id = label_id.replace('\t','')
        category = np.ravel_multi_index(tuple([int(x) for x in list(label_id)]), dims = self.id_dim, order='C')
        # to get label id back from category, used "np.unravel_index(np.prod(self.id_dim)-1,shape=self.id_dim,order='C')"

        return str(category)

    # To tensor, images should be C*Z*H*W
    def to_tensor(self, list_images):
        for image_i in range(len(list_images)):
            list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
        return list_images

    def random_flip_single_img(self, image, p = 0.5):
        axes = [1, 2, 3]  # channel is dim 0
        transf_img = copy.deepcopy(image)
        for ax in axes:
            val = np.random.uniform(0, 1)
            if val >= p:
                transf_img = np.flip(copy.deepcopy(transf_img), axis = ax)

        return transf_img


    def get_transform(self):
        transform = tio.Compose((
            tio.RandomAffine(scales=(0.9,1.2),degrees = 10,
                             image_interpolation = 'nearest'),
        ))
        return transform

    def train_transform(self, list_images):
        list_images = self.to_tensor(list_images)

        # transform1 = self.get_transform()
        # list_images[1] = transform1(list_images[1])

        transform2 = self.get_transform()
        for i in range(len(list_images)):
            list_images[i] = transform2(list_images[i])
        return list_images

    def val_transform(self, list_images):
        # list_images[1] = self.random_flip_single_img(list_images[1])
        # list_images[1] = self.random_transpose_single_img(list_images[1])

        list_images = self.to_tensor(list_images)
        # transform = self.get_transform()
        # list_images[1] = transform(list_images[1])
        return list_images


    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            patient_dir = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            patient_dir = self.list_case_id[new_index_]

        self.patient_dir = patient_dir

        mc_img = self.load_img(patient_dir,load_dose=False)
        dose_img = self.load_img(patient_dir,load_dose=True)

        list_images = [mc_img,  # input1
                       mc_img, # input2
                       dose_img]  # input3


        '''load positive and negative samples'''
        pos_img, neg_img = self.load_pos_and_neg_samples_using_category(patient_dir)
        # pos_img, neg_img = self.load_pos_and_neg_samples_using_dose(patient_dir)
        list_images += [pos_img, neg_img]

        ''''''
        list_images = self.transform[self.phase](list_images)

        if self.phase == 'query' or self.phase == 'database':
            return list_images + [self.patient_dir] # [img, transf_img, pos_img, neg_img, img_filepath]
        else:
            return list_images  # [img, transf_img, pos_img, neg_img]

    def load_pos_and_neg_samples_using_category(self,patient_dir):
        '''load positive and negative samples'''
        positive_category = self.get_category_of_file(patient_dir)
        negative_category = [x for x in list(self.categorized_filenames.keys()) if x != positive_category and \
                             self.categorized_filenames[x]]
        negative_category = np.random.choice(negative_category)

        filtered_filenames_pos = list(filter(lambda a: a != patient_dir, self.categorized_filenames[positive_category]))
        filtered_filenames_neg = list(filter(lambda a: a != patient_dir, self.categorized_filenames[negative_category]))
        if not filtered_filenames_pos:
            positive_sample_filepath = patient_dir
        else:
            positive_sample_filepath = np.random.choice(filtered_filenames_pos)
        negative_sample_filepath = np.random.choice(filtered_filenames_neg)

        pos_img = self.load_img(positive_sample_filepath,load_dose=False)
        neg_img = self.load_img(negative_sample_filepath,load_dose=False)

        return pos_img, neg_img

    def load_pos_and_neg_samples_using_dose(self,patient_dir):
        '''load positive and negative samples'''
        positive_category = self.get_category_of_file(patient_dir)
        negative_category = [x for x in list(self.categorized_filenames.keys()) if x != positive_category]
        negative_category = np.random.choice(negative_category)

        filtered_filenames_pos = list(filter(lambda a: a != patient_dir, self.categorized_filenames[positive_category]))
        filtered_filenames_neg = list(filter(lambda a: a != patient_dir, self.categorized_filenames[negative_category]))
        combined_list = filtered_filenames_pos + filtered_filenames_neg
        pos_dose_filepath = patient_dir
        neg_dose_filepath = np.random.choice(combined_list)

        pos_img = self.load_img(pos_dose_filepath,load_dose=True)
        neg_img = self.load_img(neg_dose_filepath,load_dose=True)

        return pos_img, neg_img

def get_loader(train_bs=1,
               val_bs=1,
               train_num_samples_per_epoch=200,
               val_num_samples_per_epoch=40,
               num_works=0):
    train_dataset = SimSiamDataset(num_samples_per_epoch=train_num_samples_per_epoch, phase='train')
    val_dataset = SimSiamDataset(num_samples_per_epoch=val_num_samples_per_epoch, phase='val')

    training_data_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works,
                                   pin_memory=False)
    val_data_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works,
                                 pin_memory=False)

    return training_data_loader, val_data_loader


def plot_img(input):
    input = input.cpu().detach().numpy()
    input = input[0,:,:,:,:]
    num_imgs = 1
    num_slices = input.shape[-1]
    slice_spacing = np.round(np.linspace(0, num_slices - 1, 20))

    for i in range(num_imgs):
        fig, axs = plt.subplots(len(slice_spacing), 2)
        for j in range(len(slice_spacing)):
            axs[j, 0].imshow(input[0, :, :, int(slice_spacing[j])], vmin=0, vmax=1)
            axs[j, 1].imshow(input[1, :, :, int(slice_spacing[j])], vmin=0, vmax=3)
            axs[j, 0].set_xlabel('x')
            axs[j, 0].set_ylabel('y')
            axs[j, 1].set_xlabel('x')
            axs[j, 1].set_ylabel('y')
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_dataset = SimSiamDataset(num_samples_per_epoch=None, phase='query')

    query_data_loader = data.DataLoader(dataset=query_dataset, batch_size=1, shuffle=True,
                                        num_workers=1,
                                        pin_memory=False)
    batch = iter(query_data_loader).next()

    input1, input2, pos_img, neg_img = batch[0].to(device).float(), batch[1].to(device).float(), \
                                       batch[2].to(device).float(), batch[3].to(device).float()
    print(batch[-1])
    plot_img(input1)

