import os
import yaml
import argparse
import numpy as np
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_ssim
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


# util.py
# This file contains a collection of general-purpose utility functions
# that are commonly used across various parts of the project, which includes
# 
# 1. File and Directory Operations:
# 2. Logging and Debugging:
# 3. Model Utilities: save_model, load_model
# 4. Command Line Utilities: parse_args
# 5. Configuration:
# 6. Visualization:

# function name
# function introduction

# Numpy2Tensor(data, device)
# trans numpy data to tensor data
def Numpy2Tensor(data, device='cpu'):
    assert (isinstance(data, np.ndarray)), "Error, Your input data is not a Numpy data."
    tensor = torch.from_numpy(data)
    return tensor.to(device)

# Tensor2Numpy(data)
# trans tensor data to numpy data and turn it to cpu
def Tensor2Numpy(data):
    assert(isinstance(data, torch.Tensor)), "Error, Your input data is not a Tensor data."
    return data.cpu().detach().numpy()

# write_train_result()
def write_train_result(total_loss, epoch, num_data):
    print(f"Train Loss: {(total_loss / num_data)}")
    return

# write_test_result()
def write_test_result(total_loss, num_data):
    print(f"Test Loss: {(total_loss / num_data)}")
    return

# save_model()
def save_model(model, save_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))

# one_hot_encode(labels, num_class)
def one_hot_encode(labels, num_class):
    one_hot = torch.zeros(labels.size(0), num_class)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

# show_date(dataset, id)
# randomly select data and display the relative dielectric constant
# and current distribution. 
def show_data(dataset, id=None):
    num_groups = dataset.num_groups
    if id is None:
        id = np.random.randint(0, num_groups)
    plt.subplot(1, 2, 1)
    epsilon_gt = dataset.data['epsilon_gt'][id, :]
    Mx = dataset.params['Mx']
    plt.imshow(epsilon_gt.reshape((Mx, Mx)))
    plt.colorbar()
    plt.title('dielectric')
    plt.subplot(1, 2, 2)
    J = dataset.J[id, :, 0]
    plt.imshow(np.real(J.reshape((Mx, Mx))))
    plt.colorbar()
    plt.title('current')
    plt.savefig('./fig/test3/one_sample_of_testdata.png')


def compute_metrics(epsilon_pred, epsilon_gt, Mx):
    """Compute RRMSE, PSNR, and SSIM metrics for the entire batch."""
    # Vectorized computation for RRMSE, PSNR, SSIM
    batch_rrmse = RRMSE(epsilon_pred, epsilon_gt, Mx)
    batch_psnr = PSNR(epsilon_pred, epsilon_gt, Mx)
    batch_ssim = SSIM(epsilon_pred, epsilon_gt, Mx)
    
    # Calculate average metrics for the batch
    rrmse = np.mean(batch_rrmse)
    psnr = np.mean(batch_psnr)
    ssim = np.mean(batch_ssim)
    
    # Return metrics in a structured dictionary
    return {'rrmse': rrmse, 'psnr': psnr, 'ssim': ssim}

def visualize_dielectric(epsilon_pred, epsilon_gt, folder_path, n, Mx, batch_metrics=None):
    """the visualization function for relative permittivity"""
    plt.figure(figsize=(10, 4))
    
    # ================= processing metrics =================
    sample_metrics = compute_metrics(epsilon_pred, epsilon_gt, Mx)
    batch_metrics = batch_metrics or {}
    
    # ================= generate titles =================
    title_lines = []
    # the sample metrics
    elem_line = "Eps | Sample: " + metric_str(sample_metrics)
    title_lines.append(elem_line)
    if batch_metrics:
        batch_line = "Eps | Batch: " + metric_str(batch_metrics)
        title_lines.append(batch_line)
    
    
    # ================= visualization =================
    # the ground truth value
    plot_component(epsilon_gt.reshape(Mx, Mx).T, 
                 subplot=121,
                 title="Ground Truth Eps",
                 vrange=(1, 3.0))
    
    # the predicted value
    plot_component(epsilon_pred.reshape(Mx, Mx).T,
                 subplot=122,
                 title="Predicted Eps",
                 vrange=(1, 3.0))
    
    title_lines = "\n".join(title_lines)
    plt.suptitle(title_lines)
    
    # ================= save the result =================
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, f'eps_{n}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def vis_batch_dielectric(epsilon_pred, epsilon_gt, Mx, fig_path, mode):
    """process the batch of relative permittivity visualization"""
    batch_size = epsilon_pred.shape[0]
    N = epsilon_gt.shape[0]
    
    # compute the batch metrics
    batch_metrics = compute_metrics(epsilon_pred, epsilon_gt, Mx)
    batch_metrics['batch_size'] = batch_size
    
    interval = 80
    
    # create the folder path
    folder_path = os.path.join(fig_path, f"{mode}")
    os.makedirs(folder_path, exist_ok=True)
    
    # visualize each sample in the batch
    for n in range(N):
        if n % interval == 0:
            visualize_dielectric(
                epsilon_pred=epsilon_pred[n][np.newaxis, :],
                epsilon_gt=epsilon_gt[n][np.newaxis, :],
                folder_path=folder_path,
                n=n,
                Mx=Mx,
                batch_metrics=batch_metrics
            )
    
    return batch_metrics

def visualize_current2dielectric(J_pred, J_gt, epsilon_pred, epsilon_gt, Mx, fig_path, mode, n, epoch, batch_metrics=None):
    """Visualize the relationship between current and dielectric constant"""
    plt.figure(figsize=(12, 10))
    
    # ================= compute metrics =================
    # compute sample metrics
    sample_metrics = {
        'J': compute_metrics(J_pred, J_gt, Mx),
        'epsilon': compute_metrics(epsilon_pred, epsilon_gt, Mx)
    }
    
    # batch metrics from arguments
    batch_metrics = batch_metrics or {}
    
    # ================= generate titles =================
    title_lines = []
    
    # the metrics for J
    j_elem_line = "J | Sample: " + metric_str(sample_metrics['J']) 
    title_lines.append(j_elem_line)
    if 'J' in batch_metrics:
        j_batch_line = "J | Batch: " + metric_str(batch_metrics['J'])
        title_lines.append(j_batch_line)
    
    # the metrics for epsilon
    eps_elem_line = "Eps | Sample: " + metric_str(sample_metrics['epsilon'])
    title_lines.append(eps_elem_line)
    if 'epsilon' in batch_metrics:
        eps_batch_line = "Eps | Batch: " + metric_str(batch_metrics['epsilon'])
        title_lines.append(eps_batch_line)

    # batch size information
    if 'batch_size' in batch_metrics:
        title_lines.append(f"Batch Size: {batch_metrics['batch_size']}")
    
    title_lines = "\n".join(title_lines)
    plt.suptitle(title_lines)
    
    # ================= visualization =================
    # visualize the current
    plot_component(J_gt.reshape(Mx, Mx).T, subplot=221, 
                  title="True Current", cmap='viridis')
    plot_component(J_pred.reshape(Mx, Mx).T, subplot=222, 
                  title="Predicted Current", cmap='viridis')
    
    # visualize the relative permittivity
    plot_component(epsilon_gt.reshape(Mx, Mx).T, subplot=223, 
                  title="True Dielectric", vrange=[1, 3.0])
    plot_component(epsilon_pred.reshape(Mx, Mx).T, subplot=224, 
                  title="Predicted Dielectric", vrange=[1, 3.0])
    
    # ================= save the result =================
    save_path = os.path.join(fig_path, f"epoch_{epoch}_fig_{n}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def vis_batch_current2dielectric(J_pred, J_gt, epsilon_pred, epsilon_gt, Mx, fig_path, mode, epoch):
    """Visualize a batch of current and dielectric constant data"""
    batch_size = J_pred.shape[0]
    N = J_gt.shape[0]  # assuming J_gt and epsilon_gt have the same first dimension
    
    # compute metrics for the batch
    batch_metrics = {
        'J': compute_metrics(J_pred, J_gt, Mx),
        'epsilon': compute_metrics(epsilon_pred, epsilon_gt, Mx),
        'batch_size': batch_size
    }
    
    # set the interval for visualization
    interval = 80  # keep the same as the visualization of relative permittivity
    if mode == 'BUG_BATCH':
        interval = 1

    # create the folder path for saving figures
    folder_path = os.path.join(fig_path, f"{mode}_joint")
    os.makedirs(folder_path, exist_ok=True)
    
    # visualize each sample in the batch
    for n in range(N):
        if n % interval == 0:
            visualize_current2dielectric(
                J_pred=J_pred[n][np.newaxis, :],
                J_gt=J_gt[n][np.newaxis, :],
                epsilon_pred=epsilon_pred[n][np.newaxis, :],
                epsilon_gt=epsilon_gt[n][np.newaxis, :],
                Mx=Mx,
                fig_path=folder_path,
                mode=mode,
                n=n,
                epoch=epoch,
                batch_metrics=batch_metrics
            )
    
    return batch_metrics



# helper functions
def metric_str(metrics):
    """transform metrics dictionary to a string for display"""
    if metrics is None:
        return ""
    return " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])

def plot_component(data, subplot, title, cmap='viridis', vrange=None):
    """Plot a 2D component of the data with a colorbar."""
    plt.subplot(subplot)
    im = plt.imshow(data, cmap=cmap)
    if vrange: 
        im.set_clim(vrange[0], vrange[1])
    plt.colorbar(shrink=1.0)
    plt.title(title)
    plt.axis('off')







def get_metric(epsilon_pred, epsilon_gt, Mx):
    N = epsilon_gt.shape[0]
    
    rrmse = RRMSE(epsilon_pred, epsilon_gt, Mx)
    psnr = PSNR(epsilon_pred, epsilon_gt, Mx)
    ssim = SSIM(epsilon_pred, epsilon_gt, Mx)
    
    return np.mean(rrmse), np.mean(psnr), np.mean(ssim)

def RRMSE(epsilon_pred, epsilon_gt, Mx):
    mse = (epsilon_pred - epsilon_gt) / epsilon_gt  # Element-wise division
    return np.sqrt(np.sum(np.power(mse, 2), axis=-1) / (Mx * Mx))

def PSNR(epsilon_pred, epsilon_gt, Mx):
    maxi = np.max(epsilon_gt, axis=-1)
    mse = np.mean((epsilon_pred - epsilon_gt) ** 2, axis=-1)  # Mean squared error for each sample
    return -10. * np.log(mse / (maxi ** 2)) / np.log(10.)

def SSIM(epsilon_pred, epsilon_gt, Mx, K1=0.01, K2=0.03, L=1):
    N = epsilon_gt.shape[0]
    # normalize the input data
    maxi = np.max(epsilon_gt, axis=-1)
    epsilon_gt = epsilon_gt / maxi[:, np.newaxis]
    maxi2 = np.max(epsilon_pred, axis=-1)
    epsilon_pred = epsilon_pred / maxi2[:, np.newaxis]

    # calculate SSIM
    epsilon_gt = torch.tensor(epsilon_gt.reshape(N, 1, Mx, Mx))
    epsilon_pred = torch.tensor(epsilon_pred.reshape(N, 1, Mx, Mx))
    return pytorch_ssim.ssim(epsilon_pred, epsilon_gt).cpu().numpy()

def single_SSIM(epsilon_pred, epsilon_gt, Mx, K1=0.01, K2=0.03, L=1):
    maxi = np.max(epsilon_gt)
    epsilon_gt = epsilon_gt / maxi
    maxi2 = np.max(epsilon_pred)
    epsilon_pred = epsilon_pred / maxi2
    epsilon_gt = torch.tensor(epsilon_gt)
    epsilon_pred = torch.tensor(epsilon_pred)
    epsilon_gt = epsilon_gt.unsqueeze(0).unsqueeze(0)
    epsilon_pred = epsilon_pred.unsqueeze(0).unsqueeze(0)
    return pytorch_ssim.ssim(epsilon_pred, epsilon_gt).cpu().numpy()

# dealing yaml document
def read_yaml(yaml_path): 
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()

    y = yaml.load(file_data, Loader=yaml.FullLoader)
    return y

# command line parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Specify the configuration file to load")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', '3Dtrain', '3Dtest'],
        default='train',
        help="Set the run mode: 'train', 'test', '3Dtrain', '3Dtest' (default: 'train')")

    args = parser.parse_args()

    dir = os.path.dirname(__file__)
    CONF = read_yaml(os.path.join(dir, args.config))
    exp_name = CONF["global"]["experiment_name"]
    CONF["global"]["model_save_path"] = CONF["global"]["model_save_path"].format(experiment_name=exp_name)
    CONF["global"]["fig_path"] = CONF["global"]["fig_path"].format(experiment_name=exp_name)
    
    mode = args.mode
    return CONF, mode


# transform the coordinates and pixels to images
def get_images(J, Mx):
    """
    J: (num_groups * Mx * Mx, 2)
    """
    J_real = J[:, 0].reshape(-1, Mx*Mx)
    J_imag = J[:, 1].reshape(-1, Mx*Mx)
    image_J = torch.cat((J_real, J_imag), dim=1)
    return image_J

# transform the images to coordinates and pixels
def get_coords_and_pixels(E_s, J, x_dom, y_dom, Mx, pos_encoding, d):
    """
    transform the baseline_a's input to b's input
    E_s: (num_groups, 2 * N_rec)
    J: (num_groups, 2*Mx*Mx)
    """
    J_real = J[:, :Mx * Mx].reshape(-1)
    J_imag = J[:, Mx * Mx:].reshape(-1)
    labels = torch.stack((J_real, J_imag), dim=-1)

    x0 = x_dom.t().reshape(-1)  # simulate 'F' flatten
    y0 = y_dom.t().reshape(-1)
    xy0 = torch.stack((x0, y0), dim=-1)  # shape: (Mx*Mx, 2)

    # if we want to do position encoding
    if pos_encoding:
        xy0 = position_encoding(xy0, d)

    # E_s copy each Mx*Mx times
    repeat_E_s = E_s.repeat_interleave(xy0.shape[0], dim=0)
    # xy0 copy each num_groups times
    repeat_xy0 = xy0.repeat(J.shape[0], 1)
    input_datas = torch.cat((repeat_E_s, repeat_xy0), dim=1)
    return input_datas, labels

def position_encoding(xy, d):
    """
    for each (x, y) coordinate pair, compute sine and cosine position encoding
    xy: (N, 2), for each row is (x, y)
    d:  dimension of position encoding
    return: (N, 4*d) position encoding
            [sin(2^i * x), cos(2^i * x), sin(2^i * y), cos(2^i * y), ...]
    """
    N = xy.shape[0]
    device = xy.device
    # create a (N, 4*d) zero tensor
    encoding = torch.zeros((N, 4 * d), dtype=xy.dtype).to(device)
    for i in range(d):
        # 2^i
        scale = 2.0 ** i
        # x part
        encoding[:, 4*i]   = torch.sin(scale * xy[:, 0])
        encoding[:, 4*i+1] = torch.cos(scale * xy[:, 0])
        # y part
        encoding[:, 4*i+2] = torch.sin(scale * xy[:, 1])
        encoding[:, 4*i+3] = torch.cos(scale * xy[:, 1])
        # # use x + y and x - y to posencoding
        # encoding[:, 4*i]   = torch.sin(scale * (xy[:, 0] + xy[:, 1]))
        # encoding[:, 4*i+1] = torch.cos(scale * (xy[:, 0] + xy[:, 1]))
        # # y part
        # encoding[:, 4*i+2] = torch.sin(scale * (xy[:, 0] - xy[:, 1]))
        # encoding[:, 4*i+3] = torch.cos(scale * (xy[:, 0] - xy[:, 1]))

    return encoding

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# transform the coordinates and pixels to images in 3D
def get_coords_and_pixels_3d(E_s, J_real, J_imag, x_dom, y_dom, z_dom, Mx, pos_encoding, d):
    """
    transform the baseline_a's input to b's input
    E_s: (num_groups, 2 * N_rec)
    J: (num_groups, 3*Mx*Mx*Mx)
    """
    N = J_real.shape[0]
    J_real = J_real.reshape(N * Mx * Mx * Mx, 3)
    J_imag = J_imag.reshape(N * Mx * Mx * Mx, 3)
    labels = torch.cat((J_real, J_imag), dim=1)

    x0 = x_dom.permute(2, 1, 0).contiguous().view(-1)  # 模拟 'F' flatten
    y0 = y_dom.permute(2, 1, 0).contiguous().view(-1)
    z0 = z_dom.permute(2, 1, 0).contiguous().view(-1)
    xyz0 = torch.stack((x0, y0, z0), dim=-1)  # shape: (Mx*Mx*Mx, 3)
    # if we want to do position encoding
    if pos_encoding:
        xyz0 = position_encoding_3d(xyz0, d)

    # E_s copy each Mx*Mx*Mx times
    E_s = 1e5 * E_s
    repeat_E_s = E_s.repeat_interleave(xyz0.shape[0], dim=0)
    # xy0 copy each num_groups times
    repeat_xyz0 = xyz0.repeat(N, 1)
    input_datas = torch.cat((repeat_E_s, repeat_xyz0), dim=1)
    return input_datas, labels

def position_encoding_3d(xyz, d):
    """
    calculate the position encoding for 3D coordinates
    xyz: (N, 3), for each row is (x, y, z)
    d: dimension of position encoding
    return: (N, 6*d) position encoding
            [sin(2^i * x), cos(2^i * x), sin(2^i * y), cos(2^i * y), ...]
    """
    N = xyz.shape[0]
    device = xyz.device
    # create a (N, 6*d) zero tensor
    encoding = torch.zeros((N, 6 * d), dtype=xyz.dtype).to(device)
    for i in range(d):
        # 2^i
        scale = 2.0 ** i
        # x part
        encoding[:, 4*i]   = torch.sin(scale * xyz[:, 0])
        encoding[:, 4*i+1] = torch.cos(scale * xyz[:, 0])
        # y part
        encoding[:, 4*i+2] = torch.sin(scale * xyz[:, 1])
        encoding[:, 4*i+3] = torch.cos(scale * xyz[:, 1])
        # z part
        encoding[:, 4*i+4] = torch.sin(scale * xyz[:, 2])
        encoding[:, 4*i+5] = torch.cos(scale * xyz[:, 2])
        # use x + y and x - y to posencoding
        # encoding[:, 4*i]   = torch.sin(scale * (xyz[:, 0] + xyz[:, 1]))
        # encoding[:, 4*i+1] = torch.cos(scale * (xyz[:, 0] + xyz[:, 1]))
        # # y part
        # encoding[:, 4*i+2] = torch.sin(scale * (xyz[:, 0] - xyz[:, 1]))
        # encoding[:, 4*i+3] = torch.cos(scale * (xyz[:, 0] - xyz[:, 1]))

    return encoding

# transform the images to coordinates and pixels in 3D
def get_images_3d(J, Mx):
    """
    J: (num_groups * Mx * Mx, 6)
    """
    Jx_real = J[:, 0].reshape(-1, Mx*Mx*Mx)
    Jy_real = J[:, 1].reshape(-1, Mx*Mx*Mx)
    Jz_real = J[:, 2].reshape(-1, Mx*Mx*Mx)
    Jx_imag = J[:, 3].reshape(-1, Mx*Mx*Mx)
    Jy_imag = J[:, 4].reshape(-1, Mx*Mx*Mx)
    Jz_imag = J[:, 5].reshape(-1, Mx*Mx*Mx)
    image_Jx = torch.cat((Jx_real, Jx_imag), dim=1)
    image_Jy = torch.cat((Jy_real, Jy_imag), dim=1)
    image_Jz = torch.cat((Jz_real, Jz_imag), dim=1)
    image_J = torch.stack((image_Jx, image_Jy, image_Jz), dim=-1)
    return image_J

# def visualize_3d_reconstruction(epsil_original, epsil_reconstructed, J_original, J_reconstructed, Mx, save_path, epoch):
#     """
#     Visualize the 3D reconstruction of ε and J distributions.
#     """
#     # reshape the data to 3D arrays
#     epsil_recon_3d = epsil_reconstructed.reshape(Mx, Mx, Mx).transpose(2, 1, 0)
#     epsil_original = epsil_original.reshape(Mx, Mx, Mx).transpose(2, 1, 0)
#     J_recon_3d = J_reconstructed.reshape(Mx, Mx, Mx).transpose(2, 1, 0)
#     J_original_3d = J_original.reshape(Mx, Mx, Mx).transpose(2, 1, 0)

#     # create a figure for 3D visualization
#     fig = plt.figure(figsize=(16, 12))
    
#     # create a custom colormap for J
#     cmap_J = plt.cm.get_cmap('RdBu_r')  # red for positive, blue for negative
#     cmap_J.set_bad(color='white', alpha=0)  # set NaN values to be transparent

#     # set the layout of the figure
#     left = 0.05
#     bottom = 0.05
#     width = 0.4
#     height = 0.4
#     cbar_width = 0.02
#     cbar_pad = 0.02

#     # original epsilon visualization
#     ax1 = fig.add_axes([left, bottom+height+0.1, width, height], projection='3d')
#     threshold_eps = 1.5  # optimal threshold for ε visualization
#     # 1. def the mask for original ε data
#     voxels_orig_mask = (epsil_original != 1) & (epsil_original > threshold_eps)

#     # 2. calculate the color for original ε data 
#     # 2.1 directly use the original ε values for coloring
#     epsil_color = epsil_original.copy()  # copy to avoid modifying original data

#     # 2.2 set the color for ε==1 to NaN (transparent)
#     epsil_color[epsil_original == 1] = np.nan  # set ε==1 to NaN for transparency

#     # 3. use a colormap for original ε data
#     cmap = plt.cm.viridis
#     # 3.1 get the valid range of ε values
#     valid_epsil = epsil_original[epsil_original != 1]
#     if len(valid_epsil) > 0:
#         epsil_min, epsil_max = valid_epsil.min(), valid_epsil.max()
#     else:
#         epsil_min, epsil_max = 0, 1  # default range if no valid values

#     # 3.2 normalize the ε values for colormap
#     epsil_normalized = (epsil_color - epsil_min) / (epsil_max - epsil_min + 1e-10)  # avoid division by zero
#     # 4. set transparency for ε==1
#     ax1.voxels(
#         voxels_orig_mask,
#         facecolors=cmap(epsil_normalized),  # directly use normalized values for coloring
#         edgecolor=None,
#         alpha=0.3
#     )
#     ax1.set_title("Original Epsilon")

#     # ε的colorbar
#     cax1 = fig.add_axes([left+width+cbar_pad, bottom+height+0.1, cbar_width, height])
#     plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=epsil_min, vmax=epsil_max)), cax=cax1)
#     cax1.set_ylabel('Epsilon Value', rotation=90)

#     # reconstructed epsilon visualization(the same as original)
#     ax2 = fig.add_axes([left+width+0.1+cbar_width+cbar_pad, bottom+height+0.1, width, height], projection='3d')
#     voxels_recon_mask = (epsil_recon_3d != 1) & (epsil_recon_3d > threshold_eps)

#     epsil_recon_color = epsil_recon_3d.copy()
#     epsil_recon_color[epsil_recon_3d == 1] = np.nan  # set ε==1 to NaN for transparency
#     # use a colormap for reconstructed ε data
#     valid_epsil_recon = epsil_recon_3d[epsil_recon_3d != 1]
#     if len(valid_epsil_recon) > 0:
#         epsil_recon_min, epsil_recon_max = valid_epsil_recon.min(), valid_epsil_recon.max()
#     else:
#         epsil_recon_min, epsil_recon_max = 0, 1

#     epsil_recon_normalized = (epsil_recon_color - epsil_recon_min) / (epsil_recon_max - epsil_min + 1e-10)

#     ax2.voxels(
#         voxels_recon_mask,
#         facecolors=cmap(epsil_recon_normalized),
#         edgecolor=None,
#         alpha=0.3
#     )
#     ax2.set_title("Reconstructed Epsilon")

def visualize_3d_reconstruction(epsil_original, epsil_reconstructed, J_original, J_reconstructed, Mx, save_path, step, mode):
    """simple 3D visualization of ε and J distributions"""
    
    # reshape the data to 3D arrays
    def reshape_3d(array):
        return array.reshape(Mx, Mx, Mx).transpose(2, 1, 0)
    
    epsil_orig_3d = reshape_3d(epsil_original)
    epsil_recon_3d = reshape_3d(epsil_reconstructed)
    J_orig_3d = reshape_3d(J_original)
    J_recon_3d = reshape_3d(J_reconstructed)
    # create a figure for 3D visualization
    fig = plt.figure(figsize=(16, 12))
    cmap_J = plt.cm.RdBu_r  # red for positive, blue for negative
    cmap_eps = plt.cm.viridis  # colormap for ε
    # subplot parameters
    subplot_params = {
        'left': 0.05,
        'width': 0.4,
        'height': 0.4,
        'cbar_width': 0.02,
        'vertical_gap': 0.1
    }
    
    # visualization functions
    def plot_epsilon(ax, data, title, cbar_pos):
        mask = (data != 1) & (data > 1.5)  # mask for ε > 1.5
        vmin, vmax = 1, 2.5
        
        # generate colors based on ε values
        colors = cmap_eps((data - vmin)/(vmax - vmin + 1e-10))
        colors[data == 1] = [0, 0, 0, 0]  # transparent for ε == 1
        # plot the voxels
        ax.voxels(mask, facecolors=colors, edgecolor=None, alpha=0.9)
        ax.set_title(title, fontsize=10)
        
        # remove axis labels and ticks
        ax.axis('off')  # hide axes
        # add colorbar
        cax = fig.add_axes([cbar_pos[0], cbar_pos[1], subplot_params['cbar_width'], subplot_params['height']])
        
        # customize colorbar for ε
        cmap = cmap_eps
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        # set the colorbar with extend='min' to show the minimum value
        cbar = plt.colorbar(sm, cax=cax, extend='min')  # 'min' to show the minimum value
        cbar.set_label('ε', rotation=0, labelpad=10)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))  # customize ticks

    # visualization function for current
    def plot_current(ax, data, title, cbar_pos):
        j_max = np.abs(data).max()
        colors = cmap_J((data/j_max + 1)/2)  # normalize to [0, 1] range
        alpha = np.clip(np.abs(data)/(j_max + 1e-10), 0.1, 1)  # dynamic alpha based on current magnitude
        colors[..., 3] = alpha
        
        mask = np.abs(data) > 1e-6  # only plot non-zero values
        ax.voxels(mask, facecolors=colors, edgecolor=None)
        ax.set_title(title)
        
        # add colorbar for current
        cax = fig.add_axes([cbar_pos[0], cbar_pos[1], subplot_params['cbar_width'], subplot_params['height']])
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_J, norm=plt.Normalize(vmin=-j_max, vmax=j_max)), cax=cax)
    # set the bottom position for the first row
    epsilon_bottom = subplot_params['vertical_gap']
    
    # the first row: original and reconstructed ε
    ax1 = fig.add_axes([subplot_params['left'], epsilon_bottom + subplot_params['height'], 
                    subplot_params['width'], subplot_params['height']], projection='3d')
    plot_epsilon(ax1, epsil_orig_3d, "Original ε", 
                [subplot_params['left'] + subplot_params['width'] + 0.02, epsilon_bottom + subplot_params['height']])
    
    ax2 = fig.add_axes([subplot_params['left'] + subplot_params['width'] + 0.15, epsilon_bottom + subplot_params['height'], 
                    subplot_params['width'], subplot_params['height']], projection='3d')
    plot_epsilon(ax2, epsil_recon_3d, "Reconstructed ε", 
                [subplot_params['left'] + 2*subplot_params['width'] + 0.17, epsilon_bottom + subplot_params['height']])
    # the second row: original and reconstructed J
    ax3 = fig.add_axes([subplot_params['left'], subplot_params['vertical_gap']/2, 
                    subplot_params['width'], subplot_params['height']], projection='3d')
    plot_current(ax3, J_orig_3d, "Original J", 
                [subplot_params['left'] + subplot_params['width'] + 0.02, subplot_params['vertical_gap']/2])
    
    ax4 = fig.add_axes([subplot_params['left'] + subplot_params['width'] + 0.15, subplot_params['vertical_gap']/2, 
                    subplot_params['width'], subplot_params['height']], projection='3d')
    plot_current(ax4, J_recon_3d, "Reconstructed J", 
                [subplot_params['left'] + 2*subplot_params['width'] + 0.17, subplot_params['vertical_gap']/2])
    # set the view angle for all axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    # save the figure
    os.makedirs(f"{save_path}/{mode}", exist_ok=True)
    plt.savefig(f"{save_path}/{mode}/step_{step}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # save interactive 3D plot using Plotly (optional)
    # try:
    #     import plotly.graph_objects as go
        
    #     # create a 3D isosurface plot for the original ε
    #     fig = go.Figure(data=[
    #         go.Isosurface(
    #             x=x.flatten(),
    #             y=y.flatten(),
    #             z=z.flatten(),
    #             value=epsil_original.flatten(),
    #             isomin=threshold,
    #             isomax=epsil_original.max(),
    #             opacity=0.3,
    #             surface_count=3,
    #             colorscale='Viridis',
    #             name='Reconstructed'
    #         )
    #     ])
        
    #     fig.add_trace(
    #         go.Isosurface(
    #             x=x.flatten(),
    #             y=y.flatten(),
    #             z=z.flatten(),
    #     ))

    #     fig.show()
    # except ImportError:
    #     print("Plotly is not installed. Please install it to generate interactive 3D plots.")

def view_debug(Es, cal_Es):
    Es_reshaped = Es.view(16, 20)
    true_Es_reshaped = cal_Es.view(16, 20)
    diff_Es = (Es_reshaped - true_Es_reshaped).abs()  # calculate the absolute difference

    plt.figure(figsize=(10, 5))

    # show Es
    plt.subplot(1, 3, 1)
    plt.imshow(Es_reshaped.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Es (16x20)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # show true_Es
    plt.subplot(1, 3, 2)
    plt.imshow(true_Es_reshaped.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Cal Es (16x20)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # show the difference between Es and true_Es
    plt.subplot(1, 3, 3)
    plt.imshow(diff_Es.cpu().numpy(), cmap='viridis', aspect='auto')  # use 'viridis' colormap
    plt.title("Difference (|Es - Cal Es|)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")



    # save the figure
    plt.tight_layout()
    plt.savefig('./test.png')

def view_test(data_3d, path):
    fig = plt.figure(figsize=(10, 10))  # set figure size
    for i in range(25):
        ax = fig.add_subplot(5, 5, i+1)  # 5x5 grid of subplots
        ax.imshow(data_3d[:, :, i], cmap='viridis')  # plot each layer
        ax.axis('off')  # hide axes
        ax.set_title(f'Layer {i}')
        plt.tight_layout()  # adjust layout to prevent overlap
        plt.savefig(path)

def view_zoom(epsilon, figname):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(epsilon)

    # define the zoom area (x0, x1, y0, y1)
    zoom_area = (10, 20, 30, 40)

    # plot the zoom area rectangle
    rect = plt.Rectangle((zoom_area[0], zoom_area[2]), 
                        zoom_area[1] - zoom_area[0], 
                        zoom_area[3] - zoom_area[2],
                        edgecolor='red', fill=False, linewidth=2, linestyle='--')
    ax.add_patch(rect)

    # create an inset for zoomed area
    axins = zoomed_inset_axes(ax, zoom=2, loc='lower right')  # create inset axes
    axins.imshow(epsilon)
    axins.set_xlim(zoom_area[0], zoom_area[1])
    axins.set_ylim(zoom_area[3], zoom_area[2])  # note the inverted y-axis

    # hide the ticks in the inset
    axins.set_xticks([])
    axins.set_yticks([])

    # add title and save the figure
    ax.set_title("Image with Zoomed Inset", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./fig/Au/{figname}.png")

def get_metric_3d(epsil_pred, epsil_true):
    M = epsil_pred.shape[0]
    relative_error_sq = np.square((epsil_pred - epsil_true) / epsil_true)
    return np.sqrt(np.mean(relative_error_sq)) / M