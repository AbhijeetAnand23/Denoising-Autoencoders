import torch
import os
import math
from EncDec import *
import torch.nn.functional as F
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torchvision.io as io
from torchvision import transforms
from torchvision.transforms import Lambda
import torchvision.transforms.functional as TF
from torchvision.datasets.folder import default_loader
from torchvision.datasets import VisionDataset


class AlteredMNIST(Dataset):
    def __init__(self, root="Data"):
        super(AlteredMNIST, self).__init__()
        self.root = root

        self.aug_dir = os.path.join(root, 'aug')
        self.clean_dir = os.path.join(root, 'clean')

        self.aug_images = sorted(os.listdir(self.aug_dir))
        self.clean_images = sorted(os.listdir(self.clean_dir))

        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.aug_images)

    def __getitem__(self, idx):
        aug_img_name = self.aug_images[idx]
        aug_img_path = os.path.join(self.aug_dir, aug_img_name)
        aug_image = io.read_image(aug_img_path).float().mean(dim=0, keepdim=True)

        aug_image = self.transform(aug_image)

        # Find corresponding clean image
        label = int(aug_img_name.split('_')[-1].split('.')[0])
        clean_img_name = None
        for img in self.clean_images:
            if img.endswith(f"_{label}.png"):
                clean_img_name = img
                break

        clean_img_path = os.path.join(self.clean_dir, clean_img_name)
        clean_image = io.read_image(clean_img_path).float().mean(dim=0, keepdim=True)

        clean_image = self.transform(clean_image)

        return aug_image, clean_image


class ResidualBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = None
        if stride != 1 or in_channels != out_channels:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.type = ""
        self.in_channels = 1
        
        # Autoencoder
        self.AEconv1 = nn.Conv2d(self.in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.AEbn1 = nn.BatchNorm2d(16)
        self.AErelu = nn.ReLU(inplace=True)
        self.AEmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.AElayer1 = self._make_layer(16, 32, 2, stride=1)
        self.AElayer2 = self._make_layer(32, 64, 2, stride=2)
        self.AElayer3 = self._make_layer(64, 128, 2, stride=2)

        # Variational Autoencoder   
        self.VAElatent_dim = 4

        self.VAEconv1 = nn.Conv2d(self.in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.VAEbn1 = nn.BatchNorm2d(16)
        self.VAErelu = nn.ReLU(inplace=True)
        self.VAEmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.VAElayer1 = self._make_layer(16, 32, 2, stride=1)
        self.VAElayer2 = self._make_layer(32, 64, 2, stride=2)
        self.VAElayer3 = self._make_layer(64, 128, 2, stride=2)

        self.VAEfc_mu = nn.Linear(128 * 2 * 2, self.VAElatent_dim)
        self.VAEfc_logvar = nn.Linear(128 * 2 * 2, self.VAElatent_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlockEncoder(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlockEncoder(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if(self.type == "AE"):
            x = self.AEconv1(x)
            x = self.AEbn1(x)
            x = self.AErelu(x)
            x = self.AEmaxpool(x)
            x = self.AElayer1(x)
            x = self.AElayer2(x)
            x = self.AElayer3(x)

            return x


        if(self.type == "VAE"):
            x = self.VAEconv1(x)
            x = self.VAEbn1(x)
            x = self.VAErelu(x)
            x = self.VAEmaxpool(x)
            x = self.VAElayer1(x)
            x = self.VAElayer2(x)
            x = self.VAElayer3(x)
            x = x.view(-1, 128 * 2 * 2)

            mu = self.VAEfc_mu(x)
            logvar = self.VAEfc_logvar(x)

            return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.type = ""
        self.out_channels = 1

        # Autoencoder  
        self.AEconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.AEbn1 = nn.BatchNorm2d(128)
        self.AErelu = nn.ReLU(inplace=True)
        self.AElayer1 = self._make_layer(128, 64, 2, stride=2)
        self.AElayer2 = self._make_layer(64, 32, 2, stride=2)
        self.AElayer3 = self._make_layer(32, 16, 2, stride=1)

        self.AEconv_out = nn.ConvTranspose2d(16, self.out_channels, kernel_size=6, stride=2, padding=1, bias=False)
        self.AEsigmoid = nn.Sigmoid()

        # Variational Autoencoder  
        self.VAElatent_dim = 4

        self.VAEfc = nn.Linear(self.VAElatent_dim, 128 * 2 * 2)
        self.VAErelu = nn.ReLU(inplace=True)

        self.VAEconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.VAEbn1 = nn.BatchNorm2d(128)
        self.VAErelu = nn.ReLU(inplace=True)

        self.VAElayer1 = self._make_layer(128, 64, 2, stride=2)
        self.VAElayer2 = self._make_layer(64, 32, 2, stride=2)
        self.VAElayer3 = self._make_layer(32, 16, 2, stride=1)

        self.VAEconv_out = nn.ConvTranspose2d(16, self.out_channels, kernel_size=6, stride=2, padding=1, bias=False)
        self.VAEsigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlockDecoder(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlockDecoder(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if(self.type == "AE"):
            x = self.AEconv1(x)
            x = self.AEbn1(x)
            x = self.AErelu(x)
            x = self.AElayer1(x)
            x = self.AElayer2(x)
            x = self.AElayer3(x)
            x = self.AEconv_out(x)
            x = self.AEsigmoid(x)

            return x

        if(self.type == "VAE"):
            x = self.VAEfc(x)
            x = self.VAErelu(x)
            x = x.view(-1, 128, 2, 2)

            x = self.VAEconv1(x)
            x = self.VAEbn1(x)
            x = self.VAErelu(x)
            x = self.VAElayer1(x)
            x = self.VAElayer2(x)
            x = self.VAElayer3(x)
            x = self.VAEconv_out(x)
            x = self.VAEsigmoid(x)

            return x


class AELossFn(nn.Module):
    def __init__(self):
        super(AELossFn, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        loss = self.mse_loss(output, target)
        return loss


class VAELossFn(nn.Module):
    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss


def ParameterSelector(E, D):
    params_to_train = list(E.parameters()) + list(D.parameters())
    return params_to_train


class AETrainer:
    def __init__(self, data_loader, encoder, decoder, criterion, optimizer, gpu):
        self.data_loader = data_loader
        self.encoder = encoder
        encoder.type = "AE"
        self.decoder = decoder
        decoder.type = "AE"
        self.criterion = criterion
        self.optimizer = optimizer
        if gpu == "T":
            self.gpu = "cuda"
        else:
            self.gpu = "cpu"

        self.train()

    def train(self):
        self.encoder.to(self.gpu)
        self.decoder.to(self.gpu)

        for epoch in range(1, EPOCH + 1):
            total_loss = 0.0
            total_similarity = 0.0
            num_batches = len(self.data_loader)

            for minibatch, (inputs, targets) in enumerate(self.data_loader, 1):
                inputs, targets = inputs.to(self.gpu), targets.to(self.gpu)

                # Forward pass
                encoded = self.encoder(inputs)
                outputs = self.decoder(encoded)

                # Compute loss
                loss = self.criterion(outputs, targets)


                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Print every 10th minibatch
                if minibatch % 10 == 0:
                    similarity = self.calculate_SSIM(inputs, outputs)
                    # similarity = self.calculate_PSNR(inputs, outputs)
                    total_similarity += similarity
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss.item(), similarity))

            avg_similarity = total_similarity / (num_batches / 10)
            # Print average loss for the epoch
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, total_loss / num_batches, avg_similarity))

            # Plot TSNE every 5 epochs
            if epoch % 5 == 0:
                self.plot_tsne(epoch)

            torch.save(self.encoder.state_dict(), 'AE_encoder.pth')
            torch.save(self.decoder.state_dict(), 'AE_decoder.pth')

    def calculate_SSIM(self, inputs, outputs):
        batch_size = outputs.size(0)
        similarities = []
        for i in range(batch_size):
            input_i = inputs[i:i+1]  # Select one image from inputs
            output_i = outputs[i:i+1]  # Select corresponding image from outputs
            
            # Reshape input and output to match the expected shape [1, H, W]
            input_i = input_i.squeeze(0)  # Squeeze out the batch dimension
            output_i = output_i.squeeze(0)  # Squeeze out the batch dimension

            with torch.no_grad():
                similarity = structure_similarity_index(output_i, input_i)
                if math.isnan(similarity):  # Check for NaN
                    similarity = 0.0  # Replace NaN with 0
                similarities.append(similarity)

        return sum(similarities) / len(similarities)


    def calculate_PSNR(self, inputs, outputs):
        batch_size = outputs.size(0)
        noises = []
        for i in range(batch_size):
            input_i = inputs[i:i+1]  # Select one image from inputs
            output_i = outputs[i:i+1]  # Select corresponding image from outputs
            
            # Reshape input and output to match the expected shape [1, H, W]
            input_i = input_i.squeeze(0)  # Squeeze out the batch dimension
            output_i = output_i.squeeze(0)  # Squeeze out the batch dimension

            with torch.no_grad():
                noise = peak_signal_to_noise_ratio(output_i, input_i)
                if math.isnan(noise):  # Check for NaN
                    noise = 0.0  # Replace NaN with 0
                noises.append(noise)

        return sum(noises) / len(noises)


    def plot_tsne(self, epoch):
        # Get logits for whole dataset
        all_logits = []
        with torch.no_grad():
            for inputs, _ in self.data_loader:
                inputs = inputs.to(self.gpu)
                encoded = self.encoder(inputs)
                all_logits.append(encoded.cpu().numpy())

        # Convert the list of numpy arrays to a single numpy array
        all_logits = np.concatenate(all_logits, axis=0)

        # Reshape the array to have 2 dimensions
        all_logits_reshaped = all_logits.reshape(all_logits.shape[0], -1)

        # Perform TSNE
        tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(all_logits_reshaped)

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], alpha=0.5)
        ax.set_title('TSNE Plot of AE (Epoch {})'.format(epoch))
        plt.savefig('AE_epoch_{}.png'.format(epoch))
        plt.close()


class VAETrainer:
    def __init__(self, data_loader, encoder, decoder, criterion, optimizer, gpu):
        self.data_loader = data_loader
        self.encoder = encoder
        self.encoder.type = "VAE"
        self.decoder = decoder
        self.decoder.type = "VAE"
        self.criterion = criterion
        self.optimizer = optimizer
        if gpu == "T":
            self.gpu = "cuda"
        else:
            self.gpu = "cpu"

        self.train()


    def train(self):
        self.encoder.to(self.gpu)
        self.decoder.to(self.gpu)
        
        for epoch in range(1, EPOCH + 1):
            total_loss = 0.0
            for i, (input_data, _) in enumerate(self.data_loader):
                input_data = input_data.to(self.gpu)
                self.optimizer.zero_grad()
                
                mu, logvar = self.encoder(input_data)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                reconstructed_data = self.decoder(z)
                loss = self.criterion(reconstructed_data, input_data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    with torch.no_grad():
                        similarity = self.calculate_SSIM(input_data, reconstructed_data)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, i + 1, loss.item(), similarity))

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, total_loss / len(self.data_loader), similarity))
            
            if epoch % 5 == 0:
                self.plot_tsne(epoch)

            torch.save(self.encoder.state_dict(), 'VAE_encoder.pth')
            torch.save(self.decoder.state_dict(), 'VAE_decoder.pth')



    def calculate_SSIM(self, inputs, outputs):
        batch_size = outputs.size(0)
        similarities = []
        for i in range(batch_size):
            input_i = inputs[i:i+1]  # Select one image from inputs
            output_i = outputs[i:i+1]  # Select corresponding image from outputs
            
            # Reshape input and output to match the expected shape [1, H, W]
            input_i = input_i.squeeze(0)  # Squeeze out the batch dimension
            output_i = output_i.squeeze(0)  # Squeeze out the batch dimension

            with torch.no_grad():
                similarity = structure_similarity_index(output_i, input_i)
                if math.isnan(similarity):  # Check for NaN
                    similarity = 0.0  # Replace NaN with 0
                similarities.append(similarity)

        return sum(similarities) / len(similarities)
    

    def calculate_PSNR(self, inputs, outputs):
        batch_size = outputs.size(0)
        noises = []
        for i in range(batch_size):
            input_i = inputs[i:i+1]  # Select one image from inputs
            output_i = outputs[i:i+1]  # Select corresponding image from outputs
            
            # Reshape input and output to match the expected shape [1, H, W]
            input_i = input_i.squeeze(0)  # Squeeze out the batch dimension
            output_i = output_i.squeeze(0)  # Squeeze out the batch dimension

            with torch.no_grad():
                noise = peak_signal_to_noise_ratio(output_i, input_i)
                if math.isnan(noise):  # Check for NaN
                    noise = 0.0  # Replace NaN with 0
                noises.append(noise)

        return sum(noises) / len(noises)
    

    def plot_tsne(self, epoch):
        # Get logits for whole dataset
        all_logits = []
        with torch.no_grad():
            for inputs, _ in self.data_loader:
                inputs = inputs.to(self.gpu)
                mu, _ = self.encoder(inputs)
                all_logits.append(mu.cpu().numpy())  # Only using the mean tensor

        # Convert the list of numpy arrays to a single numpy array
        all_logits = np.concatenate(all_logits, axis=0)

        # Reshape the array to have 2 dimensions
        all_logits_reshaped = all_logits.reshape(all_logits.shape[0], -1)

        # Perform TSNE
        tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(all_logits_reshaped)

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], alpha=0.5)
        ax.set_title('TSNE Plot of VAE (Epoch {})'.format(epoch))
        plt.savefig('VAE_epoch_{}.png'.format(epoch))
        plt.close()

class AE_TRAINED:
    def __init__(self, encoder_path='AE_encoder.pth', decoder_path='AE_decoder.pth', gpu=False):
        # Initialize encoder and decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.type = "AE"
        self.decoder.type = "AE"
        
        # Load trained weights
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        
        # Set device
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def from_path(self, sample_path, original_path, type):

        # Load sample and original images
        sample_image = io.read_image(sample_path).float().mean(dim=0, keepdim=True).unsqueeze(0)
        original_image = io.read_image(original_path).float().mean(dim=0, keepdim=True).unsqueeze(0)

        # Move images to device
        sample_image = sample_image.to(self.device)
        original_image = original_image.to(self.device)

        # Forward pass through encoder-decoder
        with torch.no_grad():
            encoded = self.encoder(sample_image)
            reconstructed = self.decoder(encoded)

        # Compute similarity score
        if type == "SSIM":
            similarity = self.calculate_SSIM(reconstructed, original_image)
        elif type == "PSNR":
            similarity = self.calculate_PSNR(reconstructed, original_image)

        return similarity
    
    def calculate_SSIM(self, input_img, output_img):
        with torch.no_grad():
            similarity = structure_similarity_index(output_img, input_img)
            if math.isnan(similarity):  # Check for NaN
                return 0.0
            
        return similarity
    
    
    def calculate_PSNR(self, input_img, output_img):
        with torch.no_grad():
            noise = peak_signal_to_noise_ratio(output_img, input_img)
            if math.isnan(noise):  # Check for NaN
                return 0.0

        return noise


class VAE_TRAINED:
    def __init__(self, encoder_path='VAE_encoder.pth', decoder_path='VAE_decoder.pth', gpu=False):
        # Initialize encoder and decoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.type = "VAE"
        self.decoder.type = "VAE"

        # Load trained weights
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        
        # Set device
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def from_path(self, sample_path, original_path, type):
        # Load sample and original images
        sample_image = io.read_image(sample_path).float().mean(dim=0, keepdim=True).unsqueeze(0)
        original_image = io.read_image(original_path).float().mean(dim=0, keepdim=True).unsqueeze(0)

        # Move images to device
        sample_image = sample_image.to(self.device)
        original_image = original_image.to(self.device)

        # Forward pass through encoder-decoder
        with torch.no_grad():
            mu, _ = self.encoder(sample_image)
            reconstructed = self.decoder(mu)

        # Compute similarity score
        if type == "SSIM":
            similarity = self.calculate_SSIM(reconstructed, original_image)
        elif type == "PSNR":
            similarity = self.calculate_PSNR(reconstructed, original_image)

        return similarity
    
    def calculate_SSIM(self, input_img, output_img):
        with torch.no_grad():
            similarity = structure_similarity_index(output_img, input_img)
            if math.isnan(similarity):  # Check for NaN
                return 0.0
            
        return similarity
    
    
    def calculate_PSNR(self, input_img, output_img):
        with torch.no_grad():
            noise = peak_signal_to_noise_ratio(output_img, input_img)
            if math.isnan(noise):  # Check for NaN
                return 0.0

        return noise

class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11, device=img1.device)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()