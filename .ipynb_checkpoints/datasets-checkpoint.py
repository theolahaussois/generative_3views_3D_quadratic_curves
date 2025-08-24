# datasets.py
# Dataset PyTorch pour les surfaces quadratiques et fonction de split train/val/test

import os, csv
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import torchvision.transforms as T  # pour resize et conversion en tenseur

# --- DATASET ---
class QuadraticSurfaceDataset(Dataset):
    """
    Dataset PyTorch pour charger les images 3 vues d'une surface quadratique
    - Chaque exemple contient trois images : top, front, side
    - Chaque exemple contient les 7 coefficients correspondants
    """

    def __init__(self, root_dir: str, csv_name: str = "dataset.csv", img_size: int = 128):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, csv_name)
        self.samples = []

        # lecture du CSV pour récupérer les noms et coefficients
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prefix = row["prefix"]  # nom unique pour chaque exemple
                coeffs = [float(row[k]) for k in ["a","b","c","d","e","f","g"]]  # récupération coefficients
                self.samples.append((prefix, coeffs))  # ajout à la liste des exemples

        # transformations PyTorch : resize + conversion en tenseur
        self.to_tensor = T.Compose([
            T.Resize((img_size, img_size)),  # redimensionnement
            T.ToTensor()  # conversion en [C,H,W] et normalisation [0,1]
        ])

    def __len__(self):
        return len(self.samples)  # nombre total d'exemples

    def __getitem__(self, idx):
        """
        Retourne un exemple :
        - img_tensor : concaténation des 3 vues [9,H,W] (3 vues x 3 canaux)
        - coeffs : tenseur float des 7 coefficients
        """
        prefix, coeffs = self.samples[idx]
        views = []
        for v in ["top", "front", "side"]:
            img_path = os.path.join(self.root_dir, f"{prefix}_{v}.png")
            img = Image.open(img_path).convert("RGB")  # ouverture en RGB
            img = self.to_tensor(img)  # resize + conversion en tenseur
            views.append(img)
        img_tensor = torch.cat(views, dim=0)  # concatène canaux : [9,H,W]
        coeffs = torch.tensor(coeffs, dtype=torch.float32)
        return img_tensor, coeffs

# --- SPLIT ---
def split_dataset(ds: Dataset, train=0.8, val=0.1, test=0.1, seed=0):
    """
    Sépare un dataset PyTorch en train / validation / test
    - ds : dataset complet
    - train, val, test : fractions
    - seed : reproductibilité
    """
    n = len(ds)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val  # reste pour test
    g = torch.Generator().manual_seed(seed)  # generator pour random_split reproductible
    return random_split(ds, [n_train, n_val, n_test], generator=g)
