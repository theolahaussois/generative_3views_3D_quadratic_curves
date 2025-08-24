import torch
import torch.nn as nn

# --- GENERATEUR MLP ---
class MLPGenerator(nn.Module):
    """
    Réseau MLP pour générer les 3 vues [top, front, side]
    Sortie : [B, 9, H, W] (3 vues x 3 canaux RGB)
     """
    def __init__(self, input_dim=7, img_h=128, img_w=128):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.output_dim = 9 * img_h * img_w # 9 canaux (3 vues x 3 couleurs) aplatis
        
# définition du MLP : 3 couches fully connected avec ReLU + Sigmoid en sortie
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), # entrée 7 coefficients -> 256
            nn.ReLU(),
            nn.Linear(256, 1024), # couche cachée
            nn.ReLU(),
            nn.Linear(1024, self.output_dim), # sortie aplatie [9*H*W]
            nn.Sigmoid() # activation pour normaliser pixels entre 0 et 1
        )

    def forward(self, x):
        """
        x : tenseur [B,7] des coefficients
        Retour : tenseur [B,9,H,W] images générées
        """
        out = self.fc(x) # passage MLP -> [B, 9*H*W]
        out = out.view(-1, 9, self.img_h, self.img_w) # reshape en [B,9,H,W]
        return out
