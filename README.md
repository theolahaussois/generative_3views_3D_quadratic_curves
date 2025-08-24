# Génération de surfaces quadratiques en 3 vues

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6106502-14b9-42a4-94d4-aa1707f43508" alt="top view" width="128" />
  <img src="https://github.com/user-attachments/assets/3083ee7a-bcb4-46cb-8bce-8cb0b6faafca" alt="front view" width="128" />
  <img src="https://github.com/user-attachments/assets/55cfbf40-36c9-4473-aff1-839351b23175" alt="side view" width="128" />
</p>

<p align="center">
  <strong>Top</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Front</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Side</strong>
</p>





Ce projet implémente des modèles **MLP** et **CNN** en **PyTorch** pour générer **3 vues RGB** (`top`, `front`, `side`) d'une surface quadratique définie par **7 coefficients**.

---

## Structure du dépôt

| Fichier / Dossier      | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `MLP.py`               | Définition du **MLP** pour générer les 3 vues                                        |
| `CNN.py`               | Définition du **CNN** pour générer les 3 vues                                        |
| `datasets.py`          | Dataset PyTorch pour les surfaces quadratiques et fonction de split train/val/test |
| `imageutils.py`        | Fonctions pour générer les surfaces et créer le dataset                              |
| `main.py`              | Script **CLI** pour lancer l’entraînement ou l’inférence                             |
| `images_entrainement/` | Images utilisées pour l’entraînement                                               |
| `images_gen/`          | Images générées par le modèle à partir de coefficients                             |
| `best_model.pt`        | Modèle **MLP** entraîné avec **2000 images** de taille 64×64                       |
| `requirements.txt`     | Dépendances Python nécessaires au projet                                           |
| `checkpoints/`         | Checkpoints d’entraînement intermédiaires                                          |
| `__pycache__/`         | Cache Python (à ignorer dans Git)                                                  |

Le fichier `best_model.pt` correspond à un **MLP entraîné sur 2000 surfaces** avec des images de taille **64x64**.

---

## Installation

```bash
git clone https://github.com/theolahaussois/generative_3views_3D_quadratic_curves.git
cd generative_3views_3D_quadratic_curves
pip install -r requirements.txt
```

---

## Entraînement

Exemple avec un MLP :

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

---

## Test 

```bash
python main.py infer \
    --checkpoint best_model.pt \
    --coeffs 0.4 0.2 -0.1 0.1 -0.3 0.0 0.5 \
    --out prediction.png
```

---
