# Génération de surfaces quadratiques en 3 vues

Ce projet implémente des modèles **MLP** et **CNN** en PyTorch pour générer **3 vues RGB** (`top`, `front`, `side`) d'une surface quadratique définie par 7 coefficients.

---

## Structure du repo

├── CNN.py # Générateur CNN
├── MLP.py # Générateur MLP
├── datasets.py # Dataset PyTorch (images + coefficients)
├── imageutils.py # Génération du dataset synthétique
├── main.py # Script principal (train / infer)
├── images_entraînement/ # Dataset généré (images + dataset.csv)
├── images_gen/ # Résultats des inférences
└── best_model.pt # Modèle sauvegardé (Git LFS si >100Mo)

### Le fichier `best_model.pt` actuel correspond à un **MLP entraîné sur 2000 surfaces** avec des images de taille **64x64**.

---

## Installation

```bash
git clone https://github.com/theolahaussois/generative_3views_3D_quadratic_curves.git
cd REPO
pip install -r requirements.txt
```

## Entraînement 

```bash

python main.py train \
    --generate \
    --n_samples 2000 \
    --img_size 64 \
    --model cnn \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --checkpoint best_model.pt
```


## Test

```bash
python main.py infer \
    --checkpoint best_model.pt \
    --coeffs 0.4 0.2 -0.1 0.1 -0.3 0.0 0.5 \
    --out prediction.png
```
