# CNN.py
# Générateur CNN pour produire 3 vues RGB d'une surface quadratique à partir de 7 coefficients

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Réseau convolutionnel simple pour générer les 3 vues [top, front, side]
    Sortie : [B, 9, H, W] (3 vues x 3 canaux RGB)
    """

    def __init__(self, img_size=128, latent_dim=7):
        super().__init__()
        self.img_size = img_size

        # projection initiale des coefficients dans un tenseur 128x8x8
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)

        # couches de convolution 
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 8->16
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 16->32
        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)   # 32->64
        self.deconv4 = nn.ConvTranspose2d(16, 9, 4, stride=2, padding=1)    # 64->128, 9 canaux (3 vues RGB)

    def forward(self, x):
        """
        x : tenseur [B,7] des coefficients
        Retour : tenseur [B,9,H,W] images générées
        """
        x = F.relu(self.fc(x))           # projection linéaire + ReLU
        x = x.view(-1, 128, 8, 8)       # reshape pour convolution
        x = F.relu(self.deconv1(x))     # upsampling 8->16
        x = F.relu(self.deconv2(x))     # upsampling 16->32
        x = F.relu(self.deconv3(x))     # upsampling 32->64
        x = torch.sigmoid(self.deconv4(x))  # upsampling final 64->128, normalisation pixels [0,1]
        return x
