import ipywidgets as widgets
import numpy as np
from argparse import ArgumentParser
import os, stat
from subprocess import call
from utils import get_model, sample_fid, calculate_fid
import urllib, tarfile
import matplotlib.pyplot as plt
import torch 

download_dict = {
    "cat_lsgan.tar.gz":   "https://www.dropbox.com/s/ajfx8e8obktfplr/cat_lsgan.tar.gz?dl=1",
    "cat_rasgan-gp.tar.gz":   "https://www.dropbox.com/s/6zfhksq6zi5yl8d/cat_rasgan-gp.tar.gz?dl=1",
    "cat_rsgan-gp.tar.gz":   "https://www.dropbox.com/s/f49fx5c4v7f5cct/cat_rsgan-gp.tar.gz?dl=1",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1xHb7kz1fmHwk0gXsArtcbNa4oLTLz1sN&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1xfKBB_wz94am9fDWrkhYcTsc8bR3Tz3U&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1y8V8SFoeRErI_oDAaxK4sqROklTPO1B8&export=download",
    "cat_ralsgan.tar.gz":   "https://www.dropbox.com/s/yuepqv27hq8g8e1/cat_ralsgan.tar.gz?dl=1",
    "gen_cifar10_hingegan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1za2MEwbN4GsxjLTV0sHf0EfNxRRB8gs-&export=download",
    "gen_cifar10_hingegan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1om3KtU1Q-5k7SXq-jKYgC9z9K1RSPyz4&export=download",
    "gen_cifar10_rsgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1pd8SwAXU7vGWiG_Z3gmN3c-O-krSL8E-&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1qMsPo-LSrH7ta2wnjo6itCpGQqlTtl_M&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1szHr_vaLGK3_pr5YOfKNsCxyx40wzSaW&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1tffHN2nL-PXXpSs6SYAL1LxrrFKi9QnE&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1uq8rhKYpv08gwm8iJSYeV34WL_Fo3irD&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1kQfcD3F5L1RUftCfGQNy6SOFdwMmiCK5&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1mTwPFOp9bSxmzNA4XfVYqd7boVblVQvL&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1nDX-k5AUUdfSEEniifrlwWUQqm8hQcw3&export=download",
    "gen_cifar10_rsgan-gp_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1oRHzNh3rOCyEExpEwViI_P8rn_a1xmL4&export=download",
    "gen_cifar10_rasgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1oYj0vTyBA96BxI7A7sqH8AnVrYo5Fbu7&export=download",
    "gen_cifar10_rahingegan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1fuKaVvLiGiAWpnGNVOYf9dbgPFLXkFOe&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1g-nZs7fMGkfz5nl7Czn0Aph7Od1W0wzR&export=download",
    "cat_rahingegan.tar.gz":   "https://www.dropbox.com/s/9vrrpp0cqqg2q57/cat_rahingegan.tar.gz?dl=1",
    "gen_cifar10_sgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1j696N2dZ4YX8xqlhAnoqeN238fs3EMrb&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1jae9XPDL9jWYji73IVfjbd-ip4CoMJc9&export=download",
    "cat_sgan.tar.gz":   "https://www.dropbox.com/s/dzc1bmptpkomba1/cat_sgan.tar.gz?dl=1",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1aGf31oW76pWXOgGDspjjoYToFT0QC1MH&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1ax_iOQj1RVddTPg1_DboAQturRxhvZ1W&export=download",
    "gen_cifar10_wgan-gp_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1c0KnrK1KXa0qOlPls0eT-7f27NLoHzzg&export=download",
    "gen_cifar10_lsgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1dJsKOwqVU00CdYvgBpq7CLo2aXahQvLp&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1dm_V6T-B7XLPCHgmEOpehi_ZzcVT0E9i&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1XtJ_mayaOPHHe3KCuMe-Zaa-M_iJjQ9K&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1ZMPOFuaAXUib-aePVpCYfPvhkKkv_onJ&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1Zqtoz0TRyjpxi5POgVG6TXvH1gnFKk4o&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1_DzU04_k9FsilMkTTXauw949z4Igz2M6&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1_fXTWBGpDKL-undp3kmQSWzdrJk9f_0m&export=download",
    "cat_hingegan.tar.gz":   "https://www.dropbox.com/s/3rvxjo8bo7zmzi7/cat_hingegan.tar.gz?dl=1",
    "cat_rasgan.tar.gz":   "https://www.dropbox.com/s/uh4037eds4vpec4/cat_rasgan.tar.gz?dl=1",
    "gen_cifar10_ralsgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1S3_Katg1Gr60wPWiPb2adPsUp8vquDve&export=download",
    "gen_cifar10_rasgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1SnSBVb4Lwd_4p7F9ulHAyVAxwmet6FwV&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1XBsckZfpLquu-CCexvPPmQReGwvNYNlA&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1KTdmBlfIn3KUTHqCPkx95c5cqrIq6E3P&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1KqzU8Ooblzu8FzycDnemOrjL-6sgu_qt&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1L5iuO4EM7nGiZuDFFKcfX40Z0Y5d0zSy&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1LY7G_Puv7EvPh66EOudjrVLSBokC1n8n&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1MO0Yjkr_56NPBWpIgfd7LO2F-SSJ1nJL&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1FaXYHfePO-1kw2hwhAjeqKhT194VIAP7&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1H2_Eb6PwfZZ4Hb7UQ-J2dGEBN2AkUmpC&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1HFGJxWZo_myyRnxtV6jfQ295O1GxC-Xh&export=download",
    "gen_cifar10_rsgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1JGpjujPMH0Ip-f604dJdcZqhGtduYel5&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1Jn9ScH1DHWH-RrpJ1j98QRVmYeAVbIgQ&export=download",
    "cat_rsgan.tar.gz":   "https://www.dropbox.com/s/ca1mffjhy3bbovg/cat_rsgan.tar.gz?dl=1",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=18Z808e5CqZq2jXNMQutKvSVpwUWpX2t8&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1AQA1lMsAeX_0hre5GnzMHZL6-Ju3I5yW&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1BM5NulMk1NTsKiSBwJkyZ1OJHXt1gCln&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1E7qeqVMADcITOU_BW0EDE_26idkxDs8_&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1EQi5kqPW23frH1uIoBLEZ8SR4woo5_GY&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=12RF5y9LYMA7kPXwGl1GZy_gDjbdxzWDJ&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=14nITdHjaalbmiHQIRF4_kDp6xtoJyfLm&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=15-ymRj0Y8xf8D2FNDr8VN56SSD1NfhXm&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=15D1IwouiORzsXSl7xeF1w4q5p0doGk7Z&export=download"

}

def def_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--d_iter', type=int, default=1, help='the number of iterations to train the discriminator before training the generator')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type, "cifar10" or "cat"')
    parser.add_argument('--model', type=str, default='standard_cnn', help='model architecture, "standard_cnn" or "dcgan_64"')
    parser.add_argument('--loss_type', type=str, default='sgan', help='loss type, "sgan", "rsgan", "rasgan", "lsgan", "ralsgan", "hingegan", "rahingegan", "wgan-gp", "rsgan-gp" or "rasgan-gp"')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for the discriminator and the generator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 value of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 value of adam')
    parser.add_argument('--spec_norm', type=bool, default=False, help = 'spectral normalization for the discriminator')
    parser.add_argument('--no_BN', type=bool, default=False, help = 'do not use batchnorm for any of the models')
    parser.add_argument('--all_tanh', type=bool, default=False, help = 'use tanh for all activations of the models')
    parser.add_argument('--lambd', type=int, default=10, help='coefficient for gradient penalty')
    return parser.parse_args([])
   
def reproduce_results(experiment, part, use_pretrained,seed, cuda, delete_models, delete_samples):

    """
    Reproduces results according to the interactive ipywidget.

        experiment 0 = cifar10 experiments
        experiment 1 = cat 64x64 experiments
        experiment 2 = unstable experiments

        Calculates the FIDs and writes them to fid.txt
    """

    cat_ids = np.array(["All", "SGAN Experiment", "RSGAN Experiment", "RaSGAN Experiment", "LSGAN Experiment", "RaLSGAN Experiment", "HingeGAN Experiment", "RaHingeGAN Experiment", "RSGAN-GP Experiment", "RaSGAN-GP Experiment"])

    cifar10_ids = np.array(["All", "SGAN Experiment 1", "SGAN Experiment 2", "RSGAN Experiment 1", "RSGAN Experiment 2", "RaSGAN Experiment 1", "RaSGAN Experiment 2", "LSGAN Experiment 1", "LSGAN Experiment 2", "RaLSGAN Experiment 1", "RaLSGAN Experiment 2", "HingeGAN Experiment 1", "HingeGAN Experiment 2", "RaHingeGAN Experiment 1", "RaHingeGAN Experiment 2", "WGAN-GP Experiment 1", "WGAN-GP Experiment 2", "RSGAN-GP Experiment 1", "RSGAN-GP Experiment 2", "RaSGAN-GP Experiment 1", "RaSGAN-GP Experiment 2"])

    unstable_ids = np.array(["All", "SGAN lr = 0.001","SGAN Beta = (0.9, 0.9)","SGAN Remove BatchNorms","SGAN All Activations Tanh","RSGAN lr = 0.001","RSGAN Beta = (0.9, 0.9)","RSGAN Remove BatchNorms","RSGAN All Activations Tanh","RaSGAN lr = 0.001","RaSGAN Beta = (0.9, 0.9)","RaSGAN Remove BatchNorms","RaSGAN All Activations Tanh","LSGAN lr = 0.001","LSGAN Beta = (0.9, 0.9)","LSGAN Remove BatchNorms","LSGAN All Activations Tanh","RaLSGAN lr = 0.001","RaLSGAN Beta = (0.9, 0.9)","RaLSGAN Remove BatchNorms","RaLSGAN All Activations Tanh","HingeGAN lr = 0.001","HingeGAN Beta = (0.9, 0.9)","HingeGAN Remove BatchNorms","HingeGAN All Activations Tanh","RaHingeGAN lr = 0.001","RaHingeGAN Beta = (0.9, 0.9)","RaHingeGAN Remove BatchNorms","RaHingeGAN All Activations Tanh","WGAN-GP lr = 0.001","WGAN-GP Beta = (0.9, 0.9)","WGAN-GP Remove BatchNorms","WGAN-GP All Activations Tanh"])

    part = np.array(part)

    if(experiment == 0): # cifar

        if(part[0] == 0):

            print("Going to reproduce every CIFAR10 experiment.")

            num_exp = 19

            part = np.arange(1,20)

        else:

            print("Going to reproduce ", end = '')

            reproduced = cifar10_ids[part]

            for i in range(len(reproduced)):
                print( (f"{reproduced[i]}, " if i+1 != len(reproduced) else f"{reproduced[i]} "), end='')

            print("for CIFAR10.")

            num_exp = len(reproduced)


        if(use_pretrained):

            args = def_args()
            args.spec_norm = True
            args.batch_size = 64
            args.fid_sample = 50000
            args.cuda = cuda
            if args.cuda and torch.cuda.is_available():
                args.device = torch.device('cuda')

            elif args.cuda:
                print("GPU training is selected but it is not available, CPU will be used instead.")
                args.device = torch.device('cpu')
                args.cuda = False

            else:
                args.device = torch.device('cpu')

            os.makedirs("samples", exist_ok = True)

            print(f"Downloading {num_exp}" + (" models "  if num_exp!=1 else " model ") + "to models folder...")

            for exp in cifar10_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = int(exp_split[2])

                the_key = ""

                if(exp_type == 1):
                    args.d_iter = 1
                    args.beta2 = 0.999
                    args.lr = 0.0002

                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_64"

                else:
                    args.d_iter = 5
                    args.beta2 = 0.9
                    args.lr = 0.0001
                    model_text = "_n_d_5_b1_0.5_b2_0.9_b_size_64"

                for key in download_dict.keys():

                    if(key.startswith(f"gen_cifar10_{loss.lower()}{model_text}")):

                        url = download_dict[key]
                        the_key = key
                        break

                if(os.path.isfile(os.path.join("models", f"{the_key}"))):
                    print("Model file has already been downloaded. Loading the file")

                else:
                    print(f"Downloading model for {exp}")
                    urllib.request.urlretrieve(url, os.path.join("models", f"{the_key}"))

                args.loss_type = loss.lower()

                Generator, _ = get_model(args)

                Generator.load_state_dict(torch.load(os.path.join("models", f"{the_key}"), map_location=args.device))

                print(f"Generating samples for {exp}")

                sample = the_key[4:-3]+"npz"

                sample_fid(Generator, 99999, args)

                print(f"Calculating the FID between the generated samples and the real samples...")

                fid_val = calculate_fid(os.path.join("datasets", "stats_cifar.npz"), os.path.join("samples", sample), batch_size=50, use_cuda = cuda, verbose=True)

                print(f"Calculated the FID = {fid_val}, re-run 'read_calculations()' to see the result in the table.")

                if(delete_models):
                    os.remove(os.path.join("models", f"{the_key}"))
                if(delete_samples):
                    os.remove(os.path.join("samples", f"{sample}"))


        else:

            print(f"Creating training script for {num_exp}" + (" models"  if num_exp!=1 else " model") + ".")

            training_script = open("reproduce_cifar10.sh", "w+")
            os.chmod("reproduce_cifar10.sh", stat.S_IRWXU)

            training_script.write("#!/usr/bin/env bash\n")

            samples = []

            for exp in cifar10_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = int(exp_split[2])

                if(exp_type == 1):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.npz")
                    training_script.write(f"python main.py --loss_type {loss.lower()} --batch_size 64 --spec_norm True --seed {seed} " + (" --cuda False\n" if not cuda else "\n"))

                else:

                    samples.append(f"cifar10_{loss.lower()}_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.npz")
                    training_script.write(f"python main.py --loss_type {loss.lower()} --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5 --seed {seed} " + (" --cuda False\n" if not cuda else "\n"))

                
            print(f"Starting training the models.")

            training_script.close()

            run("./reproduce_cifar10.sh")

            print("Training is completed, now FIDs will be calculated.")

            for sample in samples:
                fid_val = calculate_fid(os.path.join("datasets", "stats_cifar.npz"), os.path.join("samples", sample), batch_size=50, use_cuda = cuda, verbose=True)

                print(f"Calculated the FID = {fid_val}, re-run 'read_calculations()' to see the result in the table.")


            if(delete_models):
                for sample in samples:
                    os.remove(os.path.join("models", "gen_"+sample[:-3]+"pth"))
            if(delete_samples):
                for sample in samples:
                    os.remove(os.path.join("samples", f"{sample}"))




    elif(experiment== 1): # cat

        if(part[0] == 0):

            print("Going to reproduce every CAT 64x64 experiment.")

            num_exp = 9

            part = np.arange(1,10)

        else:

            print("Going to reproduce ", end = '')

            reproduced = cat_ids[part]

            for i in range(len(reproduced)):
                print( (f"{reproduced[i]}, " if i+1 != len(reproduced) else f"{reproduced[i]} "), end='')

            print("for CAT 64x64.")

            num_exp = len(reproduced)


        if(use_pretrained):

            args = def_args()
            args.model = "dcgan_64"
            args.dataset = "cat"
            args.batch_size = 64
            args.fid_sample = 9303
            args.cuda = cuda
            if args.cuda and torch.cuda.is_available():
                args.device = torch.device('cuda')

            elif args.cuda:
                print("GPU training is selected but it is not available, CPU will be used instead.")
                args.device = torch.device('cpu')
                args.cuda = False

            else:
                args.device = torch.device('cpu')

            print(f"Downloading {num_exp}" + (" models "  if num_exp!=1 else " model ") + "to models folder...")

            for exp in cat_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                the_key = f"cat_{loss.lower()}.tar.gz"

                url = download_dict[the_key]

                if(os.path.isfile(os.path.join("models", f"{the_key}"))):
                    print("Model files has already been downloaded. Loading the file")

                else:
                    print(f"Downloading model for {exp}")
                    urllib.request.urlretrieve(url, os.path.join("models", f"{the_key}"))

                    print("Extracting the models...")
                    cats_tar = tarfile.open(os.path.join("models", f"{the_key}"))
                    cats_tar.extractall("models") 
                    cats_tar.close()
                    print("Extraction is completed.")
                    if(delete_models):
                        os.remove(os.path.join("models", f"{the_key}"))

                Generator, _ = get_model(args)

                args.loss_type = loss.lower()

                for it in range(20000,100001,10000):

                    Generator.load_state_dict(torch.load(os.path.join("models", f"cat_{loss.lower()}", f"gen_cat_{loss.lower()}_n_d_1_b_size_64_lr_0.0002_{it}.pth"), map_location=args.device))

                    print(f"Generating samples for {exp} {it}/100000")

                    sample = f"cat_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_{it}.npz"

                    sample_fid(Generator, it-1, args)

                    print(f"Calculating the FID between the generated samples and the real samples...")

                    fid_val = calculate_fid(os.path.join("datasets", "stats_cat.npz"), os.path.join("samples", sample), batch_size=10, use_cuda = cuda, verbose=True)

                    print(f"Calculated the FID = {fid_val}, re-run 'read_calculations()' to see the result in the table.")

                    if(delete_models):
                        os.remove(os.path.join("models", f"gen_cat_{loss.lower()}_n_d_1_b_size_64_lr_0.0002_{it}.pth"))
                    if(delete_samples):
                        os.remove(os.path.join("samples", f"{sample}"))

        else:

            print(f"Creating training script for {num_exp}" + (" models"  if num_exp!=1 else " model") + ".")

            training_script = open("reproduce_cat.sh", "w+")
            os.chmod("reproduce_cat.sh", stat.S_IRWXU)

            training_script.write("#!/usr/bin/env bash\n")

            samples = []

            for exp in cat_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                for it in range(20000,100001,10000):

                    samples.append(f"cat_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_{it}.npz")

                training_script.write(f"python main.py --loss_type {loss.lower()} --batch_size 64 --dataset cat --model dcgan_64 --fid_iter 10000 --save_model 10000 --seed {seed}" + (" --cuda False\n" if not cuda else "\n") )
            
            print(f"Starting training the models.")

            training_script.close()

            run("./reproduce_cat.sh")

            print("Training is completed, now FIDs will be calculated.")

            for sample in samples:
                fid_val = calculate_fid(os.path.join("datasets", "stats_cat.npz"), os.path.join("samples", sample), batch_size=10, use_cuda = cuda, verbose=True)

                print(f"Calculated the FID = {fid_val}, re-run 'read_calculations()' to see the result in the table.")
                

            if(delete_models):
                for sample in samples:
                    os.remove(os.path.join("models", "gen_"+sample[:-3]+"pth"))
            if(delete_samples):
                for sample in samples:
                    os.remove(os.path.join("samples", f"{sample}"))




    else: # unstable

        if(part[0] == 0):

            print("Going to reproduce every unstable experiment.")

            num_exp = 32

            part = np.arange(1,33)

        else:

            print("Going to reproduce ", end = '')

            reproduced = unstable_ids[part]

            for i in range(len(reproduced)):
                print( (f"{reproduced[i]}, " if i+1 != len(reproduced) else f"{reproduced[i]} "), end='')

            print("for unstable experiments.")

            num_exp = len(reproduced)


        if(use_pretrained):

            args = def_args()
            args.fid_sample = 50000
            args.cuda = cuda
            if args.cuda and torch.cuda.is_available():
                args.device = torch.device('cuda')

            elif args.cuda:
                print("GPU training is selected but it is not available, CPU will be used instead.")
                args.device = torch.device('cpu')
                args.cuda = False

            else:
                args.device = torch.device('cpu')

            print(f"Downloading {num_exp}" + (" models "  if num_exp!=1 else " model ") + "to models folder...")

            for exp in unstable_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = exp_split[1]

                the_key = ""

                if(exp_type == 'lr'):
                    args.no_BN = False
                    args.all_tanh = False
                    args.beta1=0.5
                    args.beta2=0.999
                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001"
                    args.lr = 0.001

                elif(exp_type == 'Beta'):
                    args.no_BN = False
                    args.all_tanh = False
                    args.lr = 0.0002
                    model_text = "_n_d_1_b1_0.9_b2_0.9_b_size_32_"
                    args.beta1=0.9
                    args.beta2=0.9
                    
                elif(exp_type == 'Remove'):
                    args.all_tanh = False
                    args.lr = 0.0002
                    args.beta1=0.5
                    args.beta2=0.999
                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN"
                    args.no_BN = True

                elif(exp_type == 'All'):
                    args.no_BN = False
                    args.lr = 0.0002
                    args.beta1=0.5
                    args.beta2=0.999
                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh"
                    args.all_tanh = True

                for key in download_dict.keys():                    

                    if(key.startswith(f"gen_cifar10_{loss.lower()}{model_text}")):

                        url = download_dict[key]
                        the_key = key
                        break

                if(os.path.isfile(os.path.join("models", f"{the_key}"))):
                    print("Model file has already been downloaded. Loading the file")

                else:
                    print(f"Downloading model for {exp}")
                    urllib.request.urlretrieve(url, os.path.join("models", f"{the_key}"))

                args.loss_type = loss.lower()

                Generator, _ = get_model(args)

                Generator.load_state_dict(torch.load(os.path.join("models", f"{the_key}"), map_location=args.device))

                print(f"Generating samples for {exp}")

                sample = the_key[4:-3]+"npz"

                sample_fid(Generator, 99999, args)

                print(f"Calculating the FID between the generated samples and the real samples...")

                fid_val = calculate_fid(os.path.join("datasets", "stats_cifar.npz"), os.path.join("samples", sample), batch_size=50, use_cuda = cuda, verbose=True)

                print(f"Calculated the FID = {fid_val}, re-run 'read_calculations()' to see the result in the table.")

                if(delete_models):
                    os.remove(os.path.join("models", f"{the_key}"))
                if(delete_samples):
                    os.remove(os.path.join("samples", f"{sample}"))

        else:

            print(f"Creating training script for {num_exp}" + (" models"  if num_exp!=1 else " model") + ".")

            training_script = open("reproduce_unstable.sh", "w+")
            os.chmod("reproduce_unstable.sh", stat.S_IRWXU)

            training_script.write("#!/usr/bin/env bash\n")

            samples = []

            for exp in unstable_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = exp_split[1]

                training_script.write(f"python main.py --loss_type {loss.lower()} --seed {seed} " + ("--cuda False " if not cuda else "") )

                if(exp_type == 'lr'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.npz")

                    training_script.write("--lr 0.001\n")

                elif(exp_type == 'Beta'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.npz")

                    training_script.write("--beta1 0.9 --beta2 0.9\n")
                    
                elif(exp_type == 'Remove'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.npz")

                    training_script.write("--no_BN True\n")

                elif(exp_type == 'All'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.npz")

                    training_script.write("--all_tanh True\n")

            print(f"Starting training the models.")

            training_script.close()

            run("./reproduce_unstable.sh")

            print("Training is completed, now FIDs will be calculated.")

            for sample in samples:
                fid_val = calculate_fid(os.path.join("datasets", "stats_cifar.npz"), os.path.join("samples", sample), batch_size=50, use_cuda = cuda, verbose=True)

                print(f"Calculated the FID = {fid_val}, re-run 'read_calculations()' to see the result in the table.")

            if(delete_models):
                for sample in samples:
                    os.remove(os.path.join("models", "gen_"+sample[:-3]+"pth"))
            if(delete_samples):
                for sample in samples:
                    os.remove(os.path.join("samples", f"{sample}"))


def reproduce_cat_samples(row,column,cuda):

    args = def_args()
    args.cuda = cuda
    args.model = "dcgan_64"
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    elif args.cuda:
        print("GPU training is selected but it is not available, CPU will be used instead.")
        args.device = torch.device('cpu')
        args.cuda = False

    else:
        args.device = torch.device('cpu')

    print("Loading the model...")

    Generator, _ = get_model(args)

    Generator.load_state_dict(torch.load(os.path.join("models", "cat_sample_model.pth"), map_location=args.device))

    print("Generating samples...")
    Generator.eval()
    with torch.no_grad():
        out = Generator(torch.randn(size=(row*column,128,1,1), device = args.device))        
        fig=plt.figure(figsize=(10,10))
        plt.subplots_adjust(wspace=0.05, hspace=0)
        for i in range(1, row*column+1):
            img = (out[i-1].permute(1,2,0).cpu()+1)*0.5
            fig.add_subplot(row, column, i)
            plt.axis('off')
            plt.imshow(img)
        plt.show(block=False)



def read_results():

    fido = open("fids.txt", "r")
    cifar_dict = {}
    cat_dict = {}
    unstable_dict = {}
    while(True):
        line = fido.readline()
        
        if line == "":
            break
            
            
        line_space = line.split(" ")

        name = line_space[0]

        fid = float(line_space[2])

        name_sub = name.split("_")
        
        dataset = name_sub[0]
        
        loss = name_sub[1].upper().replace("RA", "Ra").replace("HINGE", "Hinge")      
        
        if(line.startswith("cifar10") and name.endswith("b_size_64_lr_0.0002_100000.npz")):
            if loss not in cifar_dict:
                cifar_dict[loss] = [fid, None]
            else:
                cifar_dict[loss][0] = fid
            continue
            
        if(line.startswith("cifar10") and name.endswith("b_size_64_lr_0.0001_100000.npz")):
            if loss not in cifar_dict:
                cifar_dict[loss] = [None, fid]
            else:
                cifar_dict[loss][1] = fid
            continue  
            
        if(line.startswith("cifar10") and name.endswith("b_size_32_lr_0.001_100000.npz")):
            if loss not in unstable_dict:
                unstable_dict[loss] = {'lr' : fid}
            else:
                unstable_dict[loss]['lr'] = fid
            continue 
            
        if(line.startswith("cifar10") and name.endswith("b_size_32_lr_0.0002_100000_noBN.npz")):
            if loss not in unstable_dict:
                unstable_dict[loss] = {'bn' : fid}
            else:
                unstable_dict[loss]['bn'] = fid
            continue
        
        if(line.startswith("cifar10") and name.endswith("b_size_32_lr_0.0002_100000_alltanh.npz")):
            if loss not in unstable_dict:
                unstable_dict[loss] = {'tanh' : fid}
            else:
                unstable_dict[loss]['tanh'] = fid
            continue
            
        if(line.startswith("cifar10") and name.endswith("0.9_b_size_32_lr_0.0002_100000.npz")):
            if loss not in unstable_dict:
                unstable_dict[loss] = {'beta' : fid}
            else:
                unstable_dict[loss]['beta'] = fid
            continue 
            
        if(line.startswith('cat')):
            
            if loss not in cat_dict:
                cat_dict[loss] = np.empty(9)
                
            itero = name[-10:-4] # change this
            if(itero[0]=='_'):
                itero = itero[1:]
            itero = int(itero)
            
            if(itero == 10000):
                continue
            
            cat_dict[loss][itero//10000-2] = fid
            
            continue

    unstable_dict = {'SGAN': unstable_dict['SGAN'], 'RSGAN': unstable_dict['RSGAN'],'RaSGAN': unstable_dict['RaSGAN'],'LSGAN': unstable_dict['LSGAN'],'RaLSGAN': unstable_dict['RaLSGAN'],'HingeGAN': unstable_dict['HingeGAN'],'RaHingeGAN': unstable_dict['RaHingeGAN'],'WGAN-GP': unstable_dict['WGAN-GP']}
    cat_dict= {'SGAN': cat_dict['SGAN'], 'RSGAN': cat_dict['RSGAN'],'RaSGAN': cat_dict['RaSGAN'],'LSGAN': cat_dict['LSGAN'],'RaLSGAN': cat_dict['RaLSGAN'],'HingeGAN': cat_dict['HingeGAN'],'RaHingeGAN': cat_dict['RaHingeGAN'],'RSGAN-GP': cat_dict['RSGAN-GP'],'RaSGAN-GP': cat_dict['RaSGAN-GP']}
    cifar_dict = {'SGAN': cifar_dict['SGAN'], 'RSGAN': cifar_dict['RSGAN'],'RaSGAN': cifar_dict['RaSGAN'],'LSGAN': cifar_dict['LSGAN'],'RaLSGAN': cifar_dict['RaLSGAN'],'HingeGAN': cifar_dict['HingeGAN'],'RaHingeGAN': cifar_dict['RaHingeGAN'],'WGAN-GP': cifar_dict['WGAN-GP'],'RSGAN-GP': cifar_dict['RSGAN-GP'],'RaSGAN-GP': cifar_dict['RaSGAN-GP']}

    return cifar_dict, cat_dict, unstable_dict



def create_markdown_cifar(fid_dict, precision=3):
    markdown = r'|<span style="font-weight:normal">Loss Type</span>|$lr$<span style="font-weight:normal">= .0002</span><br>$\beta$ <span style="font-weight:normal">= (0.5,0.999)</span><br>$n_D$ <span style="font-weight:normal">= 1</span>|$lr$<span style="font-weight:normal">= .0001</span><br>$\beta$ <span style="font-weight:normal">= (0.5,0.9)</span><br>$n_D$ <span style="font-weight:normal">= 5</span>|' + "\n"
    markdown += r"|:-:|:-:|:-:|" + "\n"
    mins_0, mins_1 = [], []
    for key in fid_dict.keys():
        mins_0.append(fid_dict[key][0])
        if(key!="RaSGAN-GP"):
            mins_1.append(fid_dict[key][1])
    mins_0, mins_1 = min(mins_0), min(mins_1)
    for key in fid_dict.keys():
        markdown += f"|{key}" + \
                    (f"|{fid_dict[key][0]:.{precision}f}|" if(fid_dict[key][0]!=mins_0) else f"|**{fid_dict[key][0]:.{precision}f}**|") + \
                    ((f"{fid_dict[key][1]:.{precision}f}" if(fid_dict[key][1]!=mins_1) else f"**{fid_dict[key][1]:.{precision}f}**") if key!="RaSGAN-GP" else "") + "|\n"
    return markdown

def create_markdown_cat(fid_dict, precision=3):
    markdown = r'|<span style="font-weight:normal">Loss Type</span>|<span style="font-weight:normal">Minimum FID</span>|<span style="font-weight:normal">Maximum FID</span>|<span style="font-weight:normal">Mean of FIDs</span>|<span style="font-weight:normal">StDev of FIDs</span>|' + "\n"
    markdown += r"|:-:|:-:|:-:|:-:|:-:|" + "\n"
    mins, maxs, devs, means = [], [], [], []
    for key in fid_dict.keys():
        mins.append(min(fid_dict[key]))
        maxs.append(max(fid_dict[key]))
        devs.append(np.std(fid_dict[key]))
        means.append(fid_dict[key].mean())
    mins, maxs, devs, means = min(mins), min(maxs), min(devs), min(means)
    for key in fid_dict.keys():
        markdown += f"|{key}" +  \
                    (f"|{min(fid_dict[key]):.{precision}f}" if(min(fid_dict[key])!=mins) else f"|**{min(fid_dict[key]):.{precision}f}**") + \
                    (f"|{max(fid_dict[key]):.{precision}f}" if(max(fid_dict[key])!=maxs) else f"|**{max(fid_dict[key]):.{precision}f}**") + \
                    (f"|{fid_dict[key].mean():.{precision}f}"  if(fid_dict[key].mean()!=means) else f"|**{fid_dict[key].mean():.{precision}f}**") + \
                    (f"|{np.std(fid_dict[key]):.{precision}f}|\n" if(np.std(fid_dict[key])!=devs) else f"|**{np.std(fid_dict[key]):.{precision}f}**|\n")
    return markdown

def create_markdown_unstable(fid_dict, precision=3):
    markdown = r'|<span style="font-weight:normal">Loss Type</span>|$lr$<span style="font-weight:normal">= .001</span>|$\beta$<span style="font-weight:normal">=(0.9,0.9)</span>|<span style="font-weight:normal">No BN</span>|<span style="font-weight:normal">Tanh Activations</span>|' + "\n"
    markdown += r"|:-:|:-:|:-:|:-:|:-:|" + "\n"
    mins_0, mins_1, mins_2, mins_3 = [], [], [], []
    for key in fid_dict.keys():
        mins_0.append(fid_dict[key]['lr'])
        mins_1.append(fid_dict[key]['beta'])
        mins_2.append(fid_dict[key]['bn'])
        mins_3.append(fid_dict[key]['tanh'])
    mins_0, mins_1, mins_2, mins_3 = min(mins_0), min(mins_1), min(mins_2), min(mins_3)
    for key in fid_dict.keys():
        markdown += f"|{key}" +  \
                    (f"|{fid_dict[key]['lr']:.{precision}f}" if(fid_dict[key]['lr']!=mins_0) else f"|**{fid_dict[key]['lr']:.{precision}f}**") + \
                    (f"|{fid_dict[key]['beta']:.{precision}f}" if(fid_dict[key]['beta']!=mins_1) else f"|**{fid_dict[key]['beta']:.{precision}f}**") + \
                    (f"|{fid_dict[key]['bn']:.{precision}f}"  if(fid_dict[key]['bn']!=mins_2) else f"|**{fid_dict[key]['bn']:.{precision}f}**") + \
                    (f"|{fid_dict[key]['tanh']:.{precision}f}|\n" if(fid_dict[key]['tanh']!=mins_3) else f"|**{fid_dict[key]['tanh']:.{precision}f}**|\n")
    return markdown

def interaction():
    layout = widgets.Layout(width='auto')
    style = {'description_width': 'initial'}

    seed = widgets.IntText(value=1,description='Use pre-determined seed for training (set 0 for random seed):',disabled=True,style=style, layout=layout)

    delete_models = widgets.Checkbox(value=False,indent = False, style=style, layout=layout, description='Delete trained or downloaded models after calculating FID ')

    delete_samples = widgets.Checkbox(value=False,indent = False, style=style, layout=layout, description='Delete generated samples after calculating FID ')

    use_pretrained = widgets.Checkbox(value=True,indent = False, style=style, layout=layout, description='Use pre-trained models to calculate FIDs (Will re-train if not checked)')

    experiment = widgets.ToggleButtons(options=[('Cifar 10 Experiments', 0), ('Cat 64x64 Experiments', 1), ('Unstable Experiments', 2)], style=style, layout=layout,value=0,description='Experiment:')

    part =  widgets.SelectMultiple(options=[("All", 0),("SGAN Experiment 1",1),("SGAN Experiment 2",2),("RSGAN Experiment 1",3),("RSGAN Experiment 2",4),("RaSGAN Experiment 1",5),("RaSGAN Experiment 2",6),("LSGAN Experiment 1",7),("LSGAN Experiment 2",8),("RaLSGAN Experiment 1",9),("RaLSGAN Experiment 2",10),("HingeGAN Experiment 1",11),("HingeGAN Experiment 2",12),("RaHingeGAN Experiment 1",13),("RaHingeGAN Experiment 2",14),("WGAN-GP Experiment 1",15),("WGAN-GP Experiment 2",16),("RSGAN-GP Experiment 1",17),("RSGAN-GP Experiment 2",18),("RaSGAN-GP Experiment 1",19)],
                            value=[0],rows=8,disabled=False, style = {'description_width': 'initial'},layout = layout,description='Reproduce Selected Parts (Ctrl+Click for Multiple Selection):')

    def update_parts(*args):
        parts = [[("All", 0),("SGAN Experiment 1",1),("SGAN Experiment 2",2),("RSGAN Experiment 1",3),("RSGAN Experiment 2",4),("RaSGAN Experiment 1",5),("RaSGAN Experiment 2",6),("LSGAN Experiment 1",7),("LSGAN Experiment 2",8),("RaLSGAN Experiment 1",9),("RaLSGAN Experiment 2",10),("HingeGAN Experiment 1",11),("HingeGAN Experiment 2",12),("RaHingeGAN Experiment 1",13),("RaHingeGAN Experiment 2",14),("WGAN-GP Experiment 1",15),("WGAN-GP Experiment 2",16),("RSGAN-GP Experiment 1",17),("RSGAN-GP Experiment 2",18),("RaSGAN-GP Experiment 1",19)], [("All", 0),("SGAN Experiment"        ,1),("RSGAN Experiment"        ,2),("RaSGAN Experiment"        ,3),("LSGAN Experiment"        ,4),("RaLSGAN Experiment"      ,5),("HingeGAN Experiment"      ,6),("RaHingeGAN Experiment"   ,7),("RSGAN-GP Experiment"      ,8),("RaSGAN-GP Experiment"     ,9)], [("All", 0),("SGAN lr = 0.001",1),("SGAN Beta = (0.9, 0.9)",2),("SGAN Remove BatchNorms",3),("SGAN All Activations Tanh",4),("RSGAN lr = 0.001",5),("RSGAN Beta = (0.9, 0.9)",6),("RSGAN Remove BatchNorms",7),("RSGAN All Activations Tanh",8),("RaSGAN lr = 0.001",9),("RaSGAN Beta = (0.9, 0.9)",10),("RaSGAN Remove BatchNorms",11),("RaSGAN All Activations Tanh",12),("LSGAN lr = 0.001",13),("LSGAN Beta = (0.9, 0.9)",14),("LSGAN Remove BatchNorms",15),("LSGAN All Activations Tanh",16),("RaLSGAN lr = 0.001",17),("RaLSGAN Beta = (0.9, 0.9)",18),("RaLSGAN Remove BatchNorms",19),("RaLSGAN All Activations Tanh",20),("HingeGAN lr = 0.001",21),("HingeGAN Beta = (0.9, 0.9)",22),("HingeGAN Remove BatchNorms",23),("HingeGAN All Activations Tanh",24),("RaHingeGAN lr = 0.001",25),("RaHingeGAN Beta = (0.9, 0.9)",26),("RaHingeGAN Remove BatchNorms",27),("RaHingeGAN All Activations Tanh",28),("WGAN-GP lr = 0.001",29),("WGAN-GP Beta = (0.9, 0.9)",30),("WGAN-GP Remove BatchNorms",31),("WGAN-GP All Activations Tanh",32)]]
        part.options = parts[experiment.value]
    experiment.observe(update_parts, 'value')

    def update_seed(*args):
        seed.disabled = use_pretrained.value
    use_pretrained.observe(update_seed, 'value')

    device = widgets.ToggleButtons(options=[('CPU', 0), ('CUDA', 1)], style=style, layout=layout,value=1,description='Device to be used for sampling and training:')

    info = widgets.Text(value='Select experiments to reproduce and click "Run Interact" button',placeholder='',description='',disabled=True, layout=layout)

    selecter = widgets.interactive(reproduce_results, {'manual': True}, experiment=experiment, part = part, use_pretrained = use_pretrained, seed =seed, cuda=device, delete_models = delete_models, delete_samples= delete_samples)

    return info, selecter


def interaction_cat():

    layout = widgets.Layout(width='auto')
    style = {'description_width': 'initial'}
    row = widgets.BoundedIntText(value=4,min=1,max=20,step=1,description='Rows:',disabled=False)
    column  = widgets.BoundedIntText(value=5,min=1,max=20,step=1,description='Columns:',disabled=False)
    device = widgets.ToggleButtons(options=[('CPU', 0), ('CUDA', 1)], style=style, layout=layout,value=1,description='Device to be used for sampling:')
    selecter = widgets.interactive(reproduce_cat_samples, {'manual': True}, row=row, column = column, cuda=device)
    info = widgets.Text(value='Select options and click "Run Interact" button to reproduce cat samples',placeholder='',description='',disabled=True, layout=layout)
    return info, selecter