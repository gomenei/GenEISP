import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from util import (
    get_coords_and_pixels
)

# ElecDataset
# you can choose E_s with J or E_s with epsilon

class ElecDataset(Dataset):
    def __init__(self, root, config, noise_ratio, mode, transform=None):
        """
        root: path of the pickle file
        config: includes these keys:
                - xy_as_input: bool, whether take (x, y) as the input of the model or not
                - image_input: bool, whether input the data as image or as points
                - pos_encoding: bool, whether use positional encoding or not
                - J_as_label:  bool, whether take J as the output label
                - epsilon_as_label: bool, whether take epsilon as the output label
                - dimension: int, the dimension of positional encoding
        transform: additional transform (not necessarily needed)
        """

        self.baseline = config['experiment']['baseline']
        self.image_input = config['experiment']['image_input']
        self.supervise = config['experiment']['supervise']
        self.channels = config['experiment']['channels']
        self.pos_encoding = config['advanced']['pos_encoding']
        self.d = config['advanced']['dimesions']

        super().__init__()
        # ---------- 1. load data and transform to tensor ----------
        with open(root, 'rb') as f:
            data_dict = pickle.load(f)

        # basic parameters
        self.params = data_dict['params']  # usually some scalars or hyperparameters, not necessarily need to be converted to Tensor
        self.data = data_dict['data']      # dictionary storing the actual data
        self.num_groups = self.params['num_groups'] 
        self.Mx = self.params['Mx']        # the grid size when generating the data

        # transform numpy arrays in data to torch.Tensor
        # Note: if the data is already Tensor, no need to transform
        for k, v in self.data.items():
            if isinstance(v, (list, tuple)):
                self.data[k] = torch.tensor(v, dtype=torch.float32)
            elif isinstance(v, (torch.Tensor,)):
                # already Tensor, no need to transform
                continue
            else:
                # assume v is numpy.ndarray or other types that can be transformed to Tensor
                self.data[k] = torch.from_numpy(v).float()

        # get J (complex) from its real and imaginary parts
        if 'J_real' in self.data and 'J_imag' in self.data:
            self.J = torch.complex(self.data['J_real'], self.data['J_imag'])
     
        # the shape of E_s_real and E_s_imag is (num_groups, N_rec, N_inc)
        E_s_real = self.data['E_s_real']
        E_s_imag = self.data['E_s_imag']

        # add noise to each image
        N_rec = self.params['N_rec']
        N_inc = self.params['N_inc']
        for i in range(self.num_groups):
            energe = torch.sqrt(torch.mean((E_s_real[i] ** 2 + E_s_imag[i] ** 2))) * (1 / torch.sqrt(torch.tensor([2])))
            E_s_real[i] = E_s_real[i] + energe * noise_ratio * np.random.randn(*E_s_real[i].shape)
            E_s_imag[i] = E_s_imag[i] + energe * noise_ratio * np.random.randn(*E_s_imag[i].shape)
        
        E_s_real_imag = torch.cat(
            (E_s_real, E_s_imag), 
            dim=1
        )  # shape: (num_groups, 2*N_rec, N_inc)
        # reshape epsilon_gt. Since numpy used order='F', we need to be careful with the alignment
        # assuming original shape: (num_groups, Mx, Mx)
        # transpose first then reshape to simulate 'F' (column-major) order
        eps_gt = self.data['epsilon_gt']  # shape: (num_groups, Mx, Mx)
        # if eps_gt was originally (num_groups, Mx*Mx), no need to reshape
        # or we can simulate 'F' order flattening:
        eps_gt = eps_gt.permute(0, 2, 1)   # swap the second and third dimensions
        eps_gt = eps_gt.reshape(self.num_groups, -1)  # (num_groups, Mx*Mx)
        if self.channels == -1:
            eps_gt = torch.repeat_interleave(eps_gt, repeats=N_inc, dim=0)
        self.epsilon = eps_gt

        # ---------- 2. decide inputs and ouputs according to config ----------

        # the second dimension represents N_inc inputs of incident waves
        # adjust axis order        
        if self.channels != -1:
            N_inc = 1
            indices = torch.tensor([self.channels])
            J_0 = self.J
            if mode == 'test':
                E_s_real_imag = E_s_real_imag.permute(0, 2, 1)
                E_s_real_imag = E_s_real_imag[:, self.channels, :]

                self.J = self.J.permute(0, 2, 1)
                J_0 = self.J[:, self.channels, :]
        else:
            N_inc = self.params['N_inc']
            indices = torch.arange(N_inc)
            E_s_real_imag = E_s_real_imag.permute(0, 2, 1)
            E_s_real_imag = E_s_real_imag.reshape((self.num_groups * N_inc, -1))

            self.J = self.J.permute(0, 2, 1)
            J_0 = self.J.reshape((self.num_groups * N_inc, -1))

        self.inc_indices = indices.repeat(self.num_groups)
        self.input_data = E_s_real_imag
        self.num_groups *= N_inc
        self.J_separate = torch.cat([J_0.real, J_0.imag], dim=-1)
        # process input and labels according to config
        # if (x, y) coordinates are to be taken as input
        # need to concatenate E_s with (x, y)
        # and the number of data groups changes from num_groups to num_groups * Mx * Mx
        if self.baseline == 'Es_xy_to_J' and (not self.image_input):
            self.num_groups = self.num_groups * self.Mx * self.Mx
            x_dom = self.data['x_dom']  # shape: (Mx, Mx) 或 (Mx*Mx,)
            y_dom = self.data['y_dom']
            self.input_data, self.labels = get_coords_and_pixels(
                self.input_data, self.J_separate, x_dom, y_dom, self.Mx, self.pos_encoding, self.d
            )
        self.transform = transform

    def __len__(self):
        return self.num_groups
    
    def __getitem__(self, idx):
        """
        return (input_data, labels)
        if inputs and labels have been constructed in __init__,
        here we just need the index.
        """
        x = [self.input_data[idx], self.inc_indices[idx]] 

        if self.baseline == 'Es_xy_to_J': 
            if not self.image_input:
                y = self.labels[idx]
            else:
                y = [self.J_separate[idx], self.epsilon[idx]]
        else:
            y = [self.J_separate[idx], self.epsilon[idx]]

        return x, y
    
class MoEDataset(Dataset):
    def __init__(self, root, config, noise_ratio, mode, transform=None):
        """
        root: path of the pickle file
        config: includes these keys:
                - xy_as_input: bool, whether take (x, y) as the input of the model or not
                - image_input: bool, whether input the data as image or as points
                - pos_encoding: bool, whether use positional encoding or not
                - J_as_label:  bool, whether take J as the output label
                - epsilon_as_label: bool, whether take epsilon as the output label
                - dimension: int, the dimension of positional encoding
        transform: additional transform (not necessarily needed)
        """

        self.baseline = config['experiment']['baseline']
        self.image_input = config['experiment']['image_input']
        self.supervise = config['experiment']['supervise']
        self.channels = config['experiment']['channels']
        self.pos_encoding = config['advanced']['pos_encoding']
        self.d = config['advanced']['dimesions']

        super().__init__()
        # ---------- 1. load data and transform to tensor ----------
        with open(root, 'rb') as f:
            data_dict = pickle.load(f)

        # basic parameters
        self.params = data_dict['params']  # usually some scalars or hyperparameters, not necessarily need to be converted to Tensor
        self.data = data_dict['data']      # dictionary storing the actual data
        self.num_groups = self.params['num_groups'] 
        self.Mx = self.params['Mx']        # the grid size when generating the data

        # transform numpy arrays in data to torch.Tensor
        # Note: if the data is already Tensor, no need to transform
        for k, v in self.data.items():
            if isinstance(v, (list, tuple)):
                self.data[k] = torch.tensor(v, dtype=torch.float32)
            elif isinstance(v, (torch.Tensor,)):
                # already Tensor, no need to transform
                continue
            else:
                # assume v is numpy.ndarray or other types that can be transformed to Tensor
                self.data[k] = torch.from_numpy(v).float()

        # get J (complex) from its real and imaginary parts
        if ('J_real' in self.data and 'J_imag' in self.data):
            self.J = torch.complex(self.data['J_real'], self.data['J_imag'])
        else:
            self.J = torch.zeros((self.num_groups, self.Mx * self.Mx, self.params['N_inc']), dtype=torch.complex64)
        # the shape of E_s_real and E_s_imag is (num_groups, N_rec, N_inc)

        E_s_real = self.data['E_s_real']
        E_s_imag = self.data['E_s_imag']

        # add noise to each image
        N_rec = self.params['N_rec']
        N_inc = self.params['N_inc']
        for i in range(self.num_groups):
            energe = torch.sqrt(torch.mean((E_s_real[i] ** 2 + E_s_imag[i] ** 2))) * (1 / torch.sqrt(torch.tensor([2])))
            E_s_real[i] = E_s_real[i] + energe * noise_ratio * np.random.randn(*E_s_real[i].shape)
            E_s_imag[i] = E_s_imag[i] + energe * noise_ratio * np.random.randn(*E_s_imag[i].shape)
        
        E_s_real_imag = torch.cat(
            (E_s_real, E_s_imag), 
            dim=1  # corresponds to original axis=1
        )  # shape: (num_groups, 2*N_rec, N_inc)

        # reshape epsilon_gt. Since numpy used order='F', we need to be careful with the alignment
        # assuming original shape: (num_groups, Mx, Mx)
        # transpose first then reshape to simulate 'F' (column-major) order
        eps_gt = self.data['epsilon_gt']  # shape: (num_groups, Mx, Mx)
        # if eps_gt was originally (num_groups, Mx*Mx), no need to reshape
        # or we can simulate 'F' order flattening:
        eps_gt = eps_gt.permute(0, 2, 1)   # swap the second and third dimensions
        eps_gt = eps_gt.reshape(self.num_groups, -1)  # (num_groups, Mx*Mx)
        # if self.channels == -1:
        #     eps_gt = torch.repeat_interleave(eps_gt, repeats=N_inc, dim=0)
        self.epsilon = eps_gt

        # ---------- 2. decide inputs and ouputs according to config ----------

        # the second dimension represents N_inc inputs of incident waves
        # [:, :, 0] represents the amplitude of the first incident wave
        # for MNIST dataset, only the first one can be taken
        # adjust axis order
        if mode == 'train':
            pass
        elif mode == 'test':
            self.input_data = E_s_real_imag[:, :, self.channels]
            self.J_separate = torch.cat([self.J.real, self.J.imag], dim=1)[:, :, self.channels]

    def __len__(self):
        return self.num_groups
    
    def __getitem__(self, idx):
        """
        return (input_data, labels)
        if inputs and labels have been constructed in __init__,
        here we just need the index.
        """
        x = self.input_data[idx]
        y = [self.J_separate[idx], self.epsilon[idx]]

        return x, y

class ElecDataset_3D(Dataset):
    def __init__(self, root, config, noise_ratio, mode, transform=None):
        """
        root: path of the directory containing .npy files
        config: includes these keys:
                - xy_as_input: bool, whether take (x, y) as the input of the model or not
                - image_input: bool, whether input the data as image or as points
                - pos_encoding: bool, whether use positional encoding or not
                - J_as_label:  bool, whether take J as the output label
                - epsilon_as_label: bool, whether take epsilon as the output label
                - dimension: int, the dimension of positional encoding
        noise_ratio: ratio of noise to be added to the data
        """
        super().__init__()
        
        # initialize the dataset
        self.data = {}

        # the root directory should contain the required .npy files
        required_files = [
            "E_inc_imag.npy", "E_inc_real.npy", "E_s_imag.npy", "E_s_real.npy",
            "epsilon_gt.npy", "J_imag.npy", "J_real.npy", "Phi_mat_imag.npy",
            "Phi_mat_real.npy", "R_mat_imag.npy", "R_mat_real.npy",
            "x_dom.npy", "y_dom.npy", "z_dom.npy"
        ]

        # check if the required files exist in the root directory
        for file_name in required_files:
            file_path = os.path.join(root, file_name)
            if os.path.exists(file_path):
                try:
                    # load the numpy file and store it in the data dictionary
                    key = os.path.splitext(file_name)[0]  # remove the .npy extension
                    self.data[key] = np.load(file_path)
                except Exception as e:
                    print(f"Cannot load the file {file_name}: {e}")
            else:
                print(f"File not found: {file_path}")

        # transform numpy arrays in data to torch.Tensor
        for key, value in self.data.items():
            self.data[key] = torch.from_numpy(value).float()

        # tranpose and reshape the data
        self.data['epsilon_gt'] = self.data['epsilon_gt'].permute(0, 3, 2, 1)
        self.data['epsilon_gt'] = self.data['epsilon_gt'].reshape(self.data['epsilon_gt'].shape[0], -1)

        # merge the real and imaginary parts of E_inc, E_s, J, Phi_mat, and R
        # channels = config['experiment']['channels']
        # for ch in channels:
        #     self.data['E_inc_imag'] = self.data['E_inc_imag'][:, ch, :]
        #     self.data['E_inc_real'] = self.data['E_inc_real'][:, ch, :]
        #     self.data['E_s_imag'] = self.data['E_s_imag'][:, :, ch]
        #     self.data['E_s_real'] = self.data['E_s_real'][:, :, ch]
        #     self.data['J_imag'] = self.data['J_imag'][:, :, ch, :]
        #     self.data['J_real'] = self.data['J_real'][:, :, ch, :]
        self.data['E_s'] = torch.cat([self.data['E_s_real'], self.data['E_s_imag']], dim=-1)
        # merge the real and imaginary parts of J of 3 dimensions
        # self.data['J_real'] = self.data['J_real'].flatten(start_dim=1)
        # self.data['J_imag'] = self.data['J_imag'].flatten(start_dim=1)

        # other initializations
        self.config = config
        self.noise_ratio = noise_ratio
        self.mode = mode
        self.transform = transform

    def __len__(self):
        # return the number of samples in the dataset
        return self.data["epsilon_gt"].shape[0] if "epsilon_gt" in self.data else 0

    def __getitem__(self, idx):
        """
        returns the input and label for a given index.
        """
        E_s = self.data["E_s"][idx]
        J_real = self.data["J_real"][idx]
        J_imag = self.data["J_imag"][idx]
        epsilon_gt = self.data["epsilon_gt"][idx]
        return E_s, [J_real, J_imag, epsilon_gt]
        