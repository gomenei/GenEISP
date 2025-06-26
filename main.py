import torch
import torch.nn as nn
import torch.optim as optim
from model import NeRF
from dataset import MoEDataset, ElecDataset_3D
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from util import *
from loss import *
from main import *
import time

# -------------------------------
# current2dielectric()  
# Converts the model's predicted current values (J) to dielectric constant epsilon  
# Assumes input J shape: (batch_size, 2*Mx*Mx),  
# where the first half is the real part and the second half is the imaginary part
# -------------------------------
def current2dielectric(J, Mx, E_inc, Phi_mat, omega, eps_0, cell_area,
                        limit_output_range, output_upper, output_lower):
    """
    Input:
        J: Tensor of shape (batch_size, 2 * Mx * Mx) (real and imaginary parts concatenated),
        Mx: Grid size
        Other parameters: Electromagnetic parameters (E_inc, Phi_mat, omega, eps_0, cell_area)
    Output:
        epsilon: Real-valued Tensor of shape (batch_size, Mx * Mx)
    """
    # Convert to complex form
    # If shape=(B, 2*Mx*Mx), real part is [0:Mx*Mx], imaginary part is [:, Mx*Mx:2*Mx*Mx]
    # Modify indices if your shape is different
    batch_size = J.shape[0]
    J_real = J[:, : Mx * Mx]
    J_imag = J[:, Mx * Mx :]
    J_complex = torch.complex(J_real, J_imag)  # (B, Mx*Mx)

    epsilon = torch.zeros(batch_size, Mx * Mx, dtype=torch.float32, device=J.device)
    E_tot = E_inc.T + J_complex @ Phi_mat.T
    ratio = J_complex / E_tot
    epsilon = -torch.imag(ratio) / (omega * eps_0 * cell_area) + 1.0
    # use your clamp option here.
    if limit_output_range:
        epsilon = torch.clamp(epsilon, min=output_lower, max=output_upper)
    return epsilon

def current2dielectric_3d(J, Mx, E_inc, Phi_mat, omega, eps_0, cell_volume,
                        limit_output_range, output_upper, output_lower):
    batch_size = J.shape[0]
    J_real = J[:, :Mx*Mx*Mx, :]
    J_imag = J[:, Mx*Mx*Mx:, :]
    J_complex = torch.complex(J_real, J_imag)  # (B, Mx*Mx*Mx, 3)

    epsilon = torch.zeros(batch_size, Mx * Mx * Mx, dtype=torch.float32, device=J.device)
    for i in range(3):
        E_tot = E_inc[:, i].T + J_complex[:, :, i] @ Phi_mat.T
        ratio = J_complex[:, :, i] / E_tot
        epsilon += -torch.imag(ratio) / (omega * eps_0 * cell_volume) + 1.0
    epsilon /= 3

    if limit_output_range:
        epsilon = torch.clamp(epsilon, min=output_lower, max=output_upper)
    return epsilon

def current2Es(J, Mx, R_mat):
    J_real = J[:, : Mx * Mx]
    J_imag = J[:, Mx * Mx :]
    J_complex = torch.complex(J_real, J_imag)
    Es = J_complex @ R_mat.T
    Es_separate = torch.cat([Es.real, Es.imag], dim=-1)
    return Es_separate

def current2Es_3d(J, Mx, R_mat):
    J_real = J[:, :Mx * Mx * Mx, :]
    J_imag = J[:, Mx * Mx * Mx:, :]
    N = J.shape[0]
    # (batch_size, Mx*Mx*Mx, 3)
    R_expanded = R_mat.unsqueeze(0)
    R_expanded = R_expanded.repeat(N, 1, 1, 1)
    J_complex = torch.complex(J_real, J_imag)
    J_expanded = J_complex.unsqueeze(1)
    Es = torch.sum(R_expanded * J_expanded, dim=(2, 3))
    Es_separate = torch.cat((Es.real, Es.imag), dim=-1)
    return Es_separate       

def test(models, test_dataloader, device, N_inc, channels, Mx, pos_encoding, d,
         E_inc, Phi_mat, omega, eps_0, cell_area, limit_output_range, 
         output_upper, output_lower, fig_save_interval):
    """
    Test the model performance on test data.
    """
    # Initialize metrics
    tot_eps_rrmse = 0
    tot_eps_psnr = 0
    tot_eps_ssim = 0
    epoch = 0
    record = {}
    avg_psnr, avg_rrmse, avg_ssim = 0, 0, 0
    step = 0

    for X, y in tqdm(test_dataloader, desc=f"Test :", leave=False):
        with torch.no_grad():
            Es = X.to(device)
            X = Es
            J_separate = y[0].to(device)
            epsilon = y[1].to(device)
            J_preds = torch.zeros_like(J_separate)
            epsil_preds = []

            for i in range(N_inc):
                x, y = get_coords_and_pixels(X[:, :, i], J_separate[:, :, i],
                                                x_dom, y_dom,
                                                Mx, pos_encoding, d)
                y_pred = models[i](x)
                y_pred = get_images(y_pred, Mx)
                J_preds[:, :, i] = y_pred
                epsil_pred = current2dielectric(y_pred, Mx, E_inc[:, [channels[i]]], Phi_mat, omega, eps_0, cell_area, limit_output_range, output_upper, output_lower)
                epsil_preds.append(epsil_pred)
                rrmse, psnr_val, ssim_val = get_metric(epsil_pred.clone().detach().cpu().numpy(),
                                                    epsilon.clone().detach().cpu().numpy(), Mx)
                num_case = epsil_pred.shape[0]
                avg_psnr += psnr_val * num_case
                avg_rrmse += rrmse * num_case
                avg_ssim += ssim_val * num_case

            epsil_preds = torch.stack(epsil_preds, dim=-1)
            avg_epsilon = torch.mean(epsil_preds, dim=-1)

            M = avg_epsilon.shape[0]
            for j in range(M):
                rrmse, psnr_val, ssim_val = get_metric(avg_epsilon[j:j+1].clone().detach().cpu().numpy(),
                                                        epsilon[j:j+1].clone().detach().cpu().numpy(), Mx)
                record[epoch] = psnr_val
                tot_eps_rrmse += rrmse
                tot_eps_psnr += psnr_val
                tot_eps_ssim += ssim_val
                # plot figure
                if step % fig_save_interval == 0:
                    plt.figure(figsize=(15, 5))

                    # the first subplot
                    plt.subplot(1, 2, 1)
                    im = plt.imshow(epsilon[j].detach().cpu().numpy().reshape(Mx, Mx).T)
                    im.set_clim(output_lower, output_upper)
                    plt.title('Ground Truth', fontsize=14)
                    plt.axis('off')

                    # the seconde
                    plt.subplot(1, 2, 2)
                    im = plt.imshow(avg_epsilon[j].detach().cpu().numpy().reshape(Mx, Mx).T)
                    im.set_clim(output_lower, output_upper)
                    cbar = plt.colorbar(im, shrink=0.8, pad=0.01)  # set the colorbar
                    cbar.set_label('Value', fontsize=12)  # add label to colorbar
                    plt.title(f"epsilon\nrrmse: {rrmse:.4f}\npsnr: {psnr_val:.4f}\nssim: {ssim_val:.4f}", fontsize=14)
                    plt.axis('off')
                    plt.tight_layout()
                    
                    folder_path = os.path.join('./fig', f"{experiment_name}/test")
                    os.makedirs(folder_path, exist_ok=True)
                    plt.savefig(os.path.join(folder_path, f'multiple_{step}.png'))

                step += 1
    tot_eps_rrmse /= step
    tot_eps_psnr /= step
    tot_eps_ssim /= step
    avg_rrmse /= step
    avg_psnr /= step
    avg_ssim /= step
    print(f"multiple fusion metrics RRMSE: {tot_eps_rrmse}, PSNR: {tot_eps_psnr}, SSIM: {tot_eps_ssim}")
    print(f"averge singal metrics RRMSE: {avg_rrmse/N_inc}, PSNR: {avg_psnr/N_inc}, SSIM: {avg_ssim/N_inc}")

def test_3d(model, dataloader, device, config,
         E_inc, Phi_mat, omega, eps_0, cell_volume, epoch, mode):
    model.eval()
    
    experiment_name = config['global']['experiment_name']
    image_input = config['experiment']['image_input']
    limit_output_range = config['advanced']['limit_output_range']
    output_upper = config['advanced']['output_upper_limit']
    output_lower = config['advanced']['output_lower_limit']
    fig_save_interval = config['advanced']['fig_save_interval']
    step = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc=f"Test Epoch {epoch}:", leave=False):
            Es = X.to(device)
            X = Es
            J_real = y[0].to(device)
            J_imag = y[1].to(device)
            J_separate = torch.cat((J_real, J_imag), dim=1)
            epsilon = y[2].to(device)
            if image_input:
                X, y = get_coords_and_pixels_3d(X, J_real, J_imag,
                                                x_dom, y_dom, z_dom,
                                                Mx, pos_encoding, d)
            y_pred = model(X)
            if image_input:
                y_pred = get_images_3d(y_pred, Mx)
                y = get_images_3d(y, Mx)
            epsil_pred = current2dielectric_3d(y_pred, Mx, E_inc, Phi_mat, omega, eps_0, cell_volume, limit_output_range, output_upper, output_lower)
            M = epsil_pred.shape[0]
            for j in range(M):
                if step % fig_save_interval == 0:
                    visualize_3d_reconstruction(epsilon[j, :].detach().cpu().numpy(),
                                                epsil_pred[j, :].detach().cpu().numpy(),
                                                J_separate[j, :Mx*Mx*Mx, 0].detach().cpu().numpy(),
                                                y_pred[j, :Mx*Mx*Mx, 0].detach().cpu().numpy(),
                                                Mx, f'3Dfig/{experiment_name}', step, mode)  
                step += 1
                # rrmse = get_metric_3d(epsil_pred[j, :].detach().cpu().numpy(), epsilon[j, :].detach().cpu().numpy())
                # print(epsil_pred[j, :].shape)
                # record[step] = rrmse

if __name__ == "__main__":
    set_random_seed(1000)

    CONF, mode = get_args()
    print(f"config loaded: {CONF}")
    print(f"mode: {mode}")

    experiment_name = CONF['global']['experiment_name']
    train_dataset_path = CONF['global']['train_dataset_path']
    test_dataset_path = CONF['global']['test_dataset_path']

    BATCH_SIZE = CONF['training']['BATCH_SIZE']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "train":
        pass
    elif mode == "test":
        test_noise_ratio = CONF['experiment']['test_noise_ratio']
        test_dataset = MoEDataset(test_dataset_path, CONF, test_noise_ratio, 'test')

        # load some physical parameters or hyperparameters
        data = test_dataset.data
        Mx = test_dataset.params['Mx']
        N_rec = test_dataset.params['N_rec']
        lam_0 = test_dataset.params['lam_0']
        N_inc = test_dataset.params['N_inc']
        k_0 = 2.0 * torch.pi / lam_0
        omega = k_0 * 3e8
        eps_0 = 8.85e-12
        MAX = test_dataset.params['MAX']
        step_size = 2.0 * MAX / (Mx - 1)
        cell_area = step_size ** 2       

        E_inc = torch.complex(data['E_inc_real'], data['E_inc_imag']).to(device)
        Phi_mat = torch.complex(data['Phi_mat_real'], data['Phi_mat_imag']).to(device)
        R_mat = torch.complex(data['R_mat_real'], data['R_mat_imag']).to(device)
        x_dom = data['x_dom'].to(device)
        y_dom = data['y_dom'].to(device)

        test_dataloader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                drop_last=False,
            )

        # load model
        D = CONF['model']['D']
        W = CONF['model']['W']
        skips = CONF['model']['skips']
        d = CONF['advanced']['dimesions']
        baseline = CONF['experiment']['baseline']
        pos_encoding = CONF['advanced']['pos_encoding']
        channels = CONF['experiment']['channels']
        N_inc = len(channels)
        if baseline == "Es_xy_to_J":
            if pos_encoding:
                input_ch = 2 * N_rec + 4 * d
            else:
                input_ch = 2 * N_rec + 2
            output_ch = 2
        else:
            input_ch = 2 * N_rec
            output_ch = 2 * Mx * Mx

        models = [NeRF(D, W, input_ch, output_ch, skips, CONF).to(device) for i in range(N_inc)]
        experiment_name = CONF['global']['experiment_name']
        for i in range(N_inc):
            models[i].load_state_dict(torch.load(f'./model/{experiment_name}/{experiment_name}_{channels[i]}.pth', weights_only=True))
            models[i].eval()

        limit_output_range = CONF['advanced']['limit_output_range']
        output_upper = CONF['advanced']['output_upper_limit']
        output_lower = CONF['advanced']['output_lower_limit']
        image_input = CONF['experiment']['image_input']
        fig_save_interval = CONF['advanced']['fig_save_interval']

        test(models, test_dataloader, device, N_inc, channels, Mx, pos_encoding, d, 
        E_inc, Phi_mat, omega, eps_0, cell_area, limit_output_range, 
        output_upper, output_lower, fig_save_interval)

    elif mode == "3Dtrain":
        pass
    elif mode == "3Dtest":
        test_noise_ratio = CONF['experiment']['test_noise_ratio']
        test_dataset = ElecDataset_3D(test_dataset_path, CONF, test_noise_ratio, 'test')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )

        # load some physical parameters or hyperparameters
        Mx = 25
        N_rec = 160
        c = 3e8
        freq = 0.4
        lam_0 = c / (freq * 1e9)
        k_0 = 2.0 * torch.pi / lam_0
        omega = k_0 * 3e8
        eps_0 = 8.85e-12
        MAX = 1
        step_size = 2.0 * MAX / (Mx - 1)
        cell_volume = step_size ** 3       

        data = test_dataset.data        
        E_inc = torch.complex(data['E_inc_real'], data['E_inc_imag']).to(device)
        Phi_mat = torch.complex(data['Phi_mat_real'], data['Phi_mat_imag']).to(device)
        R_mat = torch.complex(data['R_mat_real'], data['R_mat_imag']).to(device)
        x_dom = data['x_dom'].to(device)
        y_dom = data['y_dom'].to(device)
        z_dom = data['z_dom'].to(device)

        # load model
        D = CONF['model']['D']
        W = CONF['model']['W']
        skips = CONF['model']['skips']
        pos_encoding = CONF['advanced']['pos_encoding']
        d = CONF['advanced']['dimesions']
        channels = CONF['experiment']['channels']
        if pos_encoding:
            input_ch = 2 * N_rec + 6 * d
        else:
            input_ch = 2 * N_rec + 30
        output_ch = 6
        model = NeRF(D, W, input_ch, output_ch, skips, CONF, Mx).to(device)
        ch = channels[0]
        model.load_state_dict(torch.load(f'./model/{experiment_name}/{experiment_name}_{ch}.pth', weights_only=True))

        EPOCH = CONF['training']['num_epochs']
        step = 1
        prev_loss = 8e-5
        wandb_metric = {}
        epoch = 0
        test_3d(model, test_dataloader, device, CONF,
                    E_inc, Phi_mat, omega, eps_0, cell_volume, epoch, mode)