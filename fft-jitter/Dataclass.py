from torch.utils.data import Dataset
import torch
import config
import utils
import matplotlib.pyplot as plt

def generate_ground_truth(ft_psf,):
    white_noise = torch.randn(*ft_psf.shape)
    ground_truth = torch.fft.ifft2(ft_psf * torch.fft.fft2(white_noise))  
    return torch.real(ground_truth), white_noise

def generate_shifts(max_jitter, image_hight):
    return torch.randn(image_hight-1)*max_jitter


def test():

    ft_psf = torch.fft.fft2(config.PSF)
    ground_truth, unconvoluted = generate_ground_truth(ft_psf)

    shifts = generate_shifts(config.MAX_JITTER, config.IMAGE_SIZE)
    print(shifts.shape)

    # fig, (ax1, ax2) = plt.subplots(1, 2)

    plt.imshow(ground_truth)
    plt.colorbar()
    # ax2.imshow(unconvoluted)
    # plt.colorbar()
    plt.show()
    

if __name__ == "__main__":
    test()
