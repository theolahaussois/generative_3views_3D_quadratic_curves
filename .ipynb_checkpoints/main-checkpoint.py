# main.py
# 1. Entraînement : génère dataset, entraîne MLP / CNN
# 2. Inférence : génère images à partir de coefficients donnés

import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import QuadraticSurfaceDataset, split_dataset
from MLP import MLPGenerator
from CNN import *
from imageutils import generate_dataset

# --- ENTRAINEMENT ET EVALUATION ---

def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()  # mode entraînement : active dropout, batchnorm
    total_loss = 0
    for imgs, coeffs in loader:                       
        imgs, coeffs = imgs.to(device), coeffs.to(device)  # transfert sur GPU ou CPU
        preds = model(coeffs)  # passage des coefficients
        loss = loss_fn(preds, imgs)  # calcul MSE entre prédiction et image réelle
        optim.zero_grad()  # reset gradients avant backprop
        loss.backward()    # backpropagation
        optim.step()       # mise à jour des poids
        total_loss += loss.item() * imgs.size(0)  # somme pondérée par batch/groupe d'entraînement
    return total_loss / len(loader.dataset)  # moyenne sur tout le dataset

def evaluate(model, loader, loss_fn, device):
    model.eval()  # mode évaluation : désactive dropout, batchnorm
    total_loss = 0
    with torch.no_grad():  # désactive calcul des gradients
        for imgs, coeffs in loader:                   
            imgs, coeffs = imgs.to(device), coeffs.to(device)
            preds = model(coeffs)
            loss = loss_fn(preds, imgs)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# --- FONCTION ENTRAINEMENT ---
def main_training(argv=None):
    # Tous les arguments pour l'entraînement : fichier d'enregistrement, nombre d'images, taille, modèle : MLP/CNN, nombre d'epochs, taille des batch, learning rate, seed
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_dir", type=str, default="images_entraînement")
    p.add_argument("--generate", action="store_true")  # génère dataset si True
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--model", type=str, choices=["mlp","cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    args = p.parse_args(argv) if argv is not None else p.parse_args()

    if args.generate:
        print("[INFO] Génération du dataset…")
        generate_dataset(args.data_dir, n_samples=args.n_samples, img_size=args.img_size, seed=args.seed)
        # génère des images : 3 vues par surface quadratique

    ds = QuadraticSurfaceDataset(args.data_dir, img_size=args.img_size)
    train_ds, val_ds, test_ds = split_dataset(ds, seed=args.seed)  # split train/val/test
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        MLPGenerator(img_h=args.img_size, img_w=args.img_size)
        if args.model == "mlp"
        else SimpleCNN(args.img_size)
    )
    model.to(device)  # envoie le modèle sur GPU si disponible

    loss_fn = nn.MSELoss()  # erreur quadratique moyenne
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)  # optimiseur Adam

    best_val = float("inf")  # initialisation meilleur score validation
    for epoch in range(1, args.epochs+1): #boucle epochs
        tr = train_one_epoch(model, train_loader, optim, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f}")
        # sauvegarde du meilleur modèle
        if va < best_val:
            best_val = va
            torch.save({
                "model": args.model,
                "img_size": args.img_size,
                "state_dict": model.state_dict()
            }, args.checkpoint)
            print(f"[INFO] Meilleur modèle sauvegardé -> {args.checkpoint}")

    te = evaluate(model, test_loader, loss_fn, device)
    print(f"Test MSE: {te:.4f}")

# --- FONCTION POUR SAUVEGARDER LES VUES ---
def save_views(batch_output, folder="images_gen", prefix="sample"):
    views = ["top", "front", "side"]
    os.makedirs(folder, exist_ok=True)  # crée le dossier si inexistant

    for i in range(batch_output.size(0)):  
        for v, name in enumerate(views):
            img = batch_output[i, v*3:(v+1)*3]  
            # découpe des canaux RGB pour chaque vue [0:3], [3:6], [6:9]
            path = os.path.join(folder, f"{prefix}_{i}_{name}.png")
            save_image(img, path)  # sauvegarde PNG

# --- FONCTION D'INFERENCE ---
def main_inference(argv=None):
    # Arguments pour l'inférence
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--coeffs", type=float, nargs=7, help="a b c d e f g")
    p.add_argument("--out", type=str, default="prediction.png")
    args = p.parse_args(argv) if argv is not None else p.parse_args()

    # chargement du modèle et de ses poids
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_type = ckpt["model"]
    img_size = ckpt["img_size"]
    model = MLPGenerator(img_h=img_size, img_w=img_size) if model_type == "mlp" else SimpleCNN(img_size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # mode évaluation

    coeffs = args.coeffs if args.coeffs is not None else [0.3,0.2,0.0,0.1,-0.1,0.0,0.0]

    with torch.no_grad():  # désactive gradient
        c = torch.tensor(coeffs, dtype=torch.float32).unsqueeze(0)
        # transforme coefficients [7] -> [1,7] pour modèle
        pred = model(c).clamp(0,1)  # clamp : pixels entre 0 et 1
        print("DEBUG sortie prédiction :", pred.min().item(), pred.max().item(), pred.mean().item())

    save_views(pred, folder="images_gen", prefix="pred")
    print(f"[INFO] Images prédite sauvegardées dans images_gen/")

# --- MAIN ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="cmd", required=True)  # sous-commandes train / infer

    # parser pour l'entraînement
    tr = subs.add_parser("train")
    tr.add_argument("--data_dir", type=str, default="images_entraînement")
    tr.add_argument("--generate", action="store_true")
    tr.add_argument("--n_samples", type=int, default=2000)
    tr.add_argument("--img_size", type=int, default=128)
    tr.add_argument("--model", type=str, choices=["mlp","cnn"], default="cnn")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--checkpoint", type=str, default="best_model.pt")

    # parser pour l'inférence
    inf = subs.add_parser("infer")
    inf.add_argument("--checkpoint", type=str, required=True)
    inf.add_argument("--coeffs", type=float, nargs=7)
    inf.add_argument("--out", type=str, default="prediction.png")

    # parsing des arguments principaux
    args = ap.parse_args()
    if args.cmd == "train":
        main_training([
            "--data_dir", args.data_dir,
            *(["--generate"] if args.generate else []),
            "--n_samples", str(args.n_samples),
            "--img_size", str(args.img_size),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--checkpoint", args.checkpoint,
        ])
    else:  # infer
        infer_args = ["--checkpoint", args.checkpoint, "--out", args.out]
        if args.coeffs is not None:
            infer_args += ["--coeffs", *map(str, args.coeffs)]
        main_inference(infer_args)
