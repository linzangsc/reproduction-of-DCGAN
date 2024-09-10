import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import CustomizedDataset, visualize_binary_result, sigmoid, visualize_float_result, visualize_latent_space
from model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.image_size = config['image_size']
        self.logger = SummaryWriter(self.config['log_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dataset = CustomizedDataset()
        self.train_dataset = self.dataset.train_dataset
        self.test_dataset = self.dataset.test_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                       batch_size=self.batch_size, shuffle=False)
        self.input_channel = config['input_channel']
        self.output_channel = config['output_channel']
        self.generator = Generator(self.input_channel, self.output_channel).to(self.device)
        self.discriminator = Discriminator(self.output_channel).to(self.device)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=config['learning_rate'])
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

    def train(self):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(self.train_loader):
                # translate to binary images
                images = images.to(self.device)
                real_labels = torch.ones((images.shape[0])).to(self.device)
                fake_labels = torch.zeros((images.shape[0])).to(self.device)
                # train D for one step
                self.discriminator_optimizer.zero_grad()
                z = torch.randn(images.shape[0], self.input_channel).to(self.device)
                fake_x = self.generator(z)
                pred_fake_x = self.discriminator(fake_x.detach())
                pred_real_x = self.discriminator(images)
                discriminator_loss = self.loss(pred_real_x, real_labels) + self.loss(pred_fake_x, fake_labels)
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()
                # train G for one step
                self.generator_optimizer.zero_grad()
                pred_fake_x = self.discriminator(fake_x)
                generator_loss = self.loss(pred_fake_x, real_labels)
                generator_loss.backward()
                self.generator_optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], \
                            G loss: {generator_loss.item():.4f}, D loss: {discriminator_loss.item():.4f}')
                self.logger.add_scalar('G loss/train', generator_loss.item(), i + epoch * len(self.train_loader))
                self.logger.add_scalar('D loss/train', discriminator_loss.item(), i + epoch * len(self.train_loader))
            self.save_model(self.config['ckpt_path'])
            with torch.no_grad():
                z = torch.randn(16, self.input_channel).to(self.device)
                sample_image = self.generator(z)
                self.visualize_samples(sample_image, epoch)

    def save_model(self, output_path):
        if not os.path.exists(output_path): os.mkdir(output_path)
        torch.save(self.generator.state_dict(), os.path.join(output_path, f"generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(output_path, f"discriminator.pth"))

    def visualize_samples(self, sample_images, epoch):
        sample_images = sample_images.reshape(sample_images.shape[0], self.image_size, self.image_size).to('cpu')
        npy_sampled_theta = np.array(sample_images)
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = visualize_float_result(npy_sampled_theta, axs)
        self.logger.add_figure(f"sample results", plt.gcf(), epoch)
        plt.close(fig)

    def visualize_latent_space(self, epoch):
        fig, ax = plt.subplots()
        for (images, labels) in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, _, _, latents = self.model(images)
            ax = visualize_latent_space(latents, labels, ax)
        plt.colorbar(ax.collections[0], ax=ax)
        self.logger.add_figure(f"latent space", plt.gcf(), epoch)
        plt.close(fig)

if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config=config)
    trainer.train()
