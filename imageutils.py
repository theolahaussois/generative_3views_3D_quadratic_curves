# imageutils.py
# Fonctions pour générer des surfaces quadratiques et créer un dataset d'images 3 vues

import os, csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt

# --- SURFACE ---
def eval_surface(X, Y, coeffs):
    a,b,c,d,e,f,g = coeffs
    Z = a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f + g
    return Z

# --- VUE 3D ---
def render_view(coeffs, elev, azim, img_size=128, grid_lim=2.0):
    """
    Génère une vue 3D d'une surface quadratique à partir des coefficients.
    """
    # --- maillage 3D ---
    n = 64  # résolution
    x = np.linspace(-grid_lim, grid_lim, n)
    y = np.linspace(-grid_lim, grid_lim, n)
    X, Y = np.meshgrid(x, y)
    
    # --- calcul Z ---
    a,b,c,d,e,f,g = coeffs
    Z = a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f + g

    # --- figure matplotlib ---
    fig = plt.figure(figsize=(1,1), dpi=img_size)  # figsize*dpi = img_size pixels
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    lim = (-grid_lim, grid_lim)
    ax.set_xlim(lim); ax.set_ylim(lim)
    zpad = 0.1 * (Z.max() - Z.min() + 1e-6)
    ax.set_zlim(Z.min()-zpad, Z.max()+zpad)

    fig.canvas.draw()  # rasterisation
    # lecture buffer rendu exact
    buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # RGB
    plt.close(fig)

    # --- redimensionner pour garantir la taille exacte ---
    img = Image.fromarray(buf)
    img = img.resize((img_size, img_size), Image.BICUBIC)

    return img


# --- GENERATION DU DATASET ---
def generate_dataset(out_dir, n_samples=1000, img_size=128, seed=0):
    os.makedirs(out_dir, exist_ok=True)  # crée dossier si nécessaire
    csv_path = os.path.join(out_dir, "dataset.csv")  # fichier CSV pour enregistrer coefficients
    rng = np.random.default_rng(seed)  # générateur aléatoire

    def sample_coeffs():
        a = rng.uniform(-1.0, 1.0)
        b = rng.uniform(-1.0, 1.0)
        c = rng.uniform(-0.5, 0.5)
        d = rng.uniform(-1.0, 1.0)
        e = rng.uniform(-1.0, 1.0)
        f = rng.uniform(-0.5, 0.5)
        g = rng.uniform(-0.5, 0.5)
        return (a,b,c,d,e,f,g)

    # ouverture CSV pour écrire les coefficients
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix","a","b","c","d","e","f","g"])  # header CSV

        for i in range(1, n_samples+1):
            coeffs = sample_coeffs()  # échantillonnage coefficients
            prefix = f"sample_{i:06d}"  # nom unique pour chaque exemple

            # génération des trois vues
            top   = render_view(coeffs, 90, -90, img_size)   # dessus
            front = render_view(coeffs, 0, -90, img_size)    # devant
            side  = render_view(coeffs, 0,   0, img_size)    # côté

            # sauvegarde images PNG
            top.save(os.path.join(out_dir, prefix+"_top.png"))
            front.save(os.path.join(out_dir, prefix+"_front.png"))
            side.save(os.path.join(out_dir, prefix+"_side.png"))

            writer.writerow([prefix, *coeffs])  # enregistrement CSV

    return csv_path  # retourne chemin CSV généré
