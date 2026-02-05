# foil_dataset.py
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from .dataset import Dataset

import matplotlib

import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_PATH_IN = "../Dataset"  # folder containing manifest.json and the "Dataset/" dir


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_pts_xyz(path: str, xyz: np.ndarray) -> None:
    with open(path, "w") as f:
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_seg_labels(path: str, labels: np.ndarray) -> None:
    with open(path, "w") as f:
        for v in labels.astype(int):
            f.write(f"{int(v)}\n")


def make_png_scatter(path: str, xy: np.ndarray) -> None:
    fig = plt.figure(figsize=(3, 3), dpi=200)
    ax = plt.gca()
    ax.scatter(xy[:, 0], xy[:, 1], s=1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # Re-save via PIL to strip metadata and ensure PNG is clean
    Image.open(path).save(path)


def to_fixed_size(points_xy: np.ndarray, npoints: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    N = len(points_xy)
    if N == npoints:
        return points_xy
    if N > npoints:
        idx = rng.choice(N, size=npoints, replace=False)
    else:
        extra = rng.choice(N, size=npoints - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
    return points_xy[idx]


def tensor_to_numpy(t):
    if t is None:
        return None
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

def convert_to_shapenet_like(
    npoints: int = 100,
    path_in: str = DEFAULT_PATH_IN,
    root_out: str = "shapenet_like_out2",
    category_name: str = "default",
    category_id: str = "00000000",
    manifest_keys = ["full_test","full_train"],
):
    """
    Convert items listed in manifest[manifest_keys[1],manifest_keys[2],...] into a ShapeNet-like layout.
    Only the chosen manifest keys is used to build the *test* split.
    """
    # -------- inputs --------
    path_in = osp.abspath(path_in)                     # .../3D/Dataset
    if not osp.isdir(path_in):
        raise FileNotFoundError(f"Dossier Dataset introuvable: {path_in}")
    manifest_path = osp.join(path_in, "manifest.json")
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json introuvable: {manifest_path}")

    # parent containing "Dataset/"
    dataset_root = osp.dirname(path_in)                # .../3D

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    all_sources = []
    for key in manifest_keys:
        if key not in manifest:
            raise KeyError(f"Clé '{key}' absente du manifest {manifest_path}")
        file_list = manifest[key]
        if not isinstance(file_list, (list, tuple)) or not file_list:
            raise ValueError(f"manifest['{key}'] est vide ou invalide.")
        all_sources.extend(file_list)

    # -------- outputs (ABSOLUTE paths) --------
    root_out_abs = osp.abspath(root_out)
    cat_dir     = ensure_dir(osp.join(root_out_abs, category_id))
    points_dir  = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))
    split_dir   = ensure_dir(osp.join(root_out_abs, "train_test_split"))

    with open(osp.join(root_out_abs, "synsetoffset2category.txt"), "w") as f:
        f.write(f"{category_name} {category_id}\n")

    test_tokens: list[str] = []

    # -------- conversion --------
    cwd_before = os.getcwd()
    try:
        # Change CWD so Dataset resolves "Dataset/<src>/..."
        os.chdir(dataset_root)

        for src in tqdm(all_sources, desc=f"Converting to ShapeNet-like"):
            data_list = Dataset([src], n_boot = npoints)
            if not (data_list and isinstance(data_list[0], Data)):
                raise RuntimeError(f"{src}: Dataset n'a pas retourné un Data valide.")
            data: Data = data_list[0]

            # Get (x, y)
            if hasattr(data, "pos") and data.pos is not None and data.pos.numel() > 0:
                xy = tensor_to_numpy(data.pos)
            else:
                x_tensor = getattr(data, "x", None)
                if x_tensor is None:
                    raise ValueError(f"{src}: Data ne contient pas 'pos' ni 'x'.")
                xy = tensor_to_numpy(x_tensor)[:, :2]

            xy  = to_fixed_size(xy, npoints)
            xyz = np.concatenate([xy, np.zeros((len(xy), 1), dtype=xy.dtype)], axis=1)
            seg = np.zeros((len(xy),), dtype=np.int64)

            # raw_token = osp.splitext(osp.basename(src))[0]
            raw_token = osp.basename(src.rstrip("/\\"))
            token = raw_token
            k = 1
            while osp.exists(osp.join(points_dir, f"{token}.pts")):
                token = f"{raw_token}_{k}"
                k += 1

            save_pts_xyz(osp.join(points_dir, f"{token}.pts"), xyz)
            save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)
            make_png_scatter(osp.join(segimg_dir, f"{token}.png"), xy)

            test_tokens.append(f"{category_id}/{token}")

    finally:
        os.chdir(cwd_before)

    # -------- splits --------
    with open(osp.join(split_dir, "shuffled_test_file_list.json"), "w") as f:
        json.dump(test_tokens, f, indent=2)
    # ensure empty train/val lists exist
    for empty in ("shuffled_train_file_list.json", "shuffled_val_file_list.json"):
        p = osp.join(split_dir, empty)
        if not osp.exists(p):
            with open(p, "w") as f:
                json.dump([], f)

    print(f"[OK] Test set prêt dans: {root_out_abs}")
    print(f"  Catégorie: {category_name} -> {category_id}")
    print(f"  Exemples (objets): {len(test_tokens)} | npoints/objet: {npoints}")

    return {"root_out": root_out_abs, "tokens": test_tokens}





def transform_xy(xy: np.ndarray, mode: str = "raw") -> np.ndarray:
    """
    Applique une transformation 2D à un nuage de points xy (N,2).
    """
    if mode == "raw":
        return xy

    elif mode == "centered": #shapenet dataset is in the middle
        return xy - xy.mean(axis=0)

    elif mode == "normalized":
        return (xy - xy.mean(axis=0)) / (xy.std(axis=0) + 1e-8)

    elif mode == "minmax":
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        return (xy - mn) / (mx - mn + 1e-8)

    elif mode == "unitsphere":
        r = np.max(np.linalg.norm(xy, axis=1))
        return xy / (r + 1e-8)

    else:
        raise ValueError(f"Mode inconnu : '{mode}'")

def create_extra_category_from_existing(
    root_out: str,
    from_category_id: str = "00000000",
    new_category_id: str = "11111111",
    new_category_name: str = "Normalized",
    transform_mode: str = "normalized",
):
    """
    Crée une nouvelle catégorie ShapeNet-like à partir d'une catégorie déjà existante,
    en appliquant une transformation définie par transform_mode sur XY.
    """
    root_out = osp.abspath(root_out)

    # Dossiers existants (source)
    src_points = osp.join(root_out, from_category_id, "points")
    src_labels = osp.join(root_out, from_category_id, "points_label")

    if not osp.isdir(src_points):
        raise FileNotFoundError(f"Catégorie source introuvable : {src_points}")

    # Dossiers nouvelle catégorie
    cat_dir     = ensure_dir(osp.join(root_out, new_category_id))
    points_dir  = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))

    print(f"\n=== Création catégorie {new_category_name} ({new_category_id}) ===")

    # Liste des objets
    tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])

    new_tokens = []

    for token in tqdm(tokens, desc=f"Transforming {new_category_name}"):
        # ---- lecture du .pts ----
        xyz = np.loadtxt(osp.join(src_points, token + ".pts"))
        xy = xyz[:, :2]

        # ---- transformer XY ----
        xy_new = transform_xy(xy, mode=transform_mode)

        # ---- reconstruire XYZ ----
        xyz_new = np.concatenate([xy_new, np.zeros((len(xy_new), 1))], axis=1)

        # ---- labels ----
        seg = np.loadtxt(osp.join(src_labels, token + ".seg")).astype(int)

        # ---- sauver ----
        save_pts_xyz(osp.join(points_dir, f"{token}.pts"), xyz_new)
        save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)
        make_png_scatter(osp.join(segimg_dir, f"{token}.png"), xy_new)

        new_tokens.append(f"{new_category_id}/{token}")

    # Ajouter dans synsetoffset2category
    with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
        f.write(f"{new_category_name} {new_category_id}\n")

    # ---------------------------
    # 🎯 SOLUTION 1 : écrire toujours une LISTE plate
    # ---------------------------

    split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")

    # Charger ancien fichier
    if osp.isfile(split_path):
        with open(split_path, "r") as f:
            loaded = json.load(f)
    else:
        loaded = []

    # Convertir en liste plate SI dictionnaire
    flat = []

    if isinstance(loaded, dict):
        # flatten dict values
        for lst in loaded.values():
            flat.extend(lst)
    elif isinstance(loaded, list):
        flat = loaded
    else:
        raise ValueError("Format JSON inattendu dans shuffled_test_file_list.json")

    # Ajouter les nouveaux tokens
    flat.extend(new_tokens)

    # Réécrire proprement : une seule liste
    with open(split_path, "w") as f:
        json.dump(flat, f, indent=2)

    print(f"[OK] Ajouté {len(new_tokens)} tokens dans test split.")


def extrude_xy(xy: np.ndarray, thickness: float = 1.0, k: int = 3) -> np.ndarray:
    """
    Extrude a 2D airfoil into 3D by duplicating XY along the Z axis.
    thickness=1.0 → z ∈ [-0.5, +0.5]
    k : number of layers
    """
    half = thickness / 2
    zs = np.linspace(-half, half, k)

    return np.vstack([np.c_[xy, np.full(len(xy), z)] for z in zs])

def create_extra_category_extruded(
    root_out: str,
    from_category_id: str = "00000000",
    new_category_id: str = "11111111",
    new_category_name: str = "Extruded",
    thickness: float = 1.0,
    k_layers: int = 3,
):
    """
    Create a new extruded 3D category from an existing 2D category.
    Airfoil remains unchanged in XY but gets duplicated along Z.
    """
    root_out = osp.abspath(root_out)

    # Source folders
    src_points = osp.join(root_out, from_category_id, "points")
    src_labels = osp.join(root_out, from_category_id, "points_label")

    if not osp.isdir(src_points):
        raise FileNotFoundError(f"Catégorie source introuvable : {src_points}")

    # New folders
    cat_dir     = ensure_dir(osp.join(root_out, new_category_id))
    points_dir  = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))

    print(f"\n=== Création catégorie extrudée {new_category_name} ({new_category_id}) ===")

    # List of objects
    tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])
    new_tokens = []

    for token in tqdm(tokens, desc=f"Extruding {new_category_name}"):

        # Load XY
        xyz = np.loadtxt(osp.join(src_points, token + ".pts"))
        xy  = xyz[:, :2]

        # Extrude ONLY
        xyz_new = extrude_xy(xy, thickness=thickness, k=k_layers)

        # Duplicate segmentation labels
        seg = np.loadtxt(osp.join(src_labels, token + ".seg")).astype(int)
        seg = np.repeat(seg, k_layers)

        # Save new files
        save_pts_xyz(osp.join(points_dir, f"{token}.pts"), xyz_new)
        save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)

        # Still generate 2D scatter plot (XY view)
        make_png_scatter(osp.join(segimg_dir, f"{token}.png"), xy)

        new_tokens.append(f"{new_category_id}/{token}")

    # Append to mapping
    with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
        f.write(f"{new_category_name} {new_category_id}\n")

    # Update test split
    split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")

    if osp.isfile(split_path):
        with open(split_path, "r") as f:
            loaded = json.load(f)
    else:
        loaded = []

    flat = loaded if isinstance(loaded, list) else sum(loaded.values(), [])
    flat.extend(new_tokens)

    with open(split_path, "w") as f:
        json.dump(flat, f, indent=2)

    print(f"[OK] Ajouté {len(new_tokens)} objets extrudés au test split.")

# def create_extra_category_torus(
#     root_out,
#     from_category_id="00000000",
#     new_category_id="22222222",
#     new_category_name="Torus",
#     R_major=3.0,
#     n_angles=100,
#     visualize=False,
#     visualize_n=10,
# ):
#     import numpy as np, json, random
#     import matplotlib.pyplot as plt
#     import os, os.path as osp
#     from tqdm import tqdm

#     # paths
#     src_points = osp.join(root_out, from_category_id, "points")
#     dst_cat = ensure_dir(osp.join(root_out, new_category_id))
#     dst_p = ensure_dir(osp.join(dst_cat, "points"))
#     dst_l = ensure_dir(osp.join(dst_cat, "points_label"))
#     dst_i = ensure_dir(osp.join(dst_cat, "seg_img"))

#     tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])
#     phis = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

#     saved_pts_paths = []
#     new_tokens = []

#     for token in tqdm(tokens, desc="Generating torus"):

#         # load cloud XY (foil in XY plane)
#         xy = np.loadtxt(osp.join(src_points, token + ".pts"))[:, :2]
#         x = xy[:, 0]   # épaisseur → rayon
#         y = xy[:, 1]   # corde → reste inchangée
#         x_shifted = x + R_major
#         # --- IMPORTANT ---
#         # décalage correct du foil pour obtenir le rayon majeur

#         pts_list = []
#         for phi in phis:
#             c = np.cos(phi)
#             s = np.sin(phi)

#             # Rotation autour de l'axe X :
#             # (x, y_shift, 0) → (x, y_shift * cos(phi), y_shift * sin(phi))
#             X = x_shifted * c 
#             Z = x_shifted * s 
#             Y = y

#             pts_list.append(np.stack([X, Y, Z], axis=1))

#         # concat copies
#         pts = np.concatenate(pts_list, axis=0)

#         # labels
#         seg = np.zeros(len(pts), int)

#         # save
#         pts_path = osp.join(dst_p, f"{token}.pts")
#         save_pts_xyz(pts_path, pts)
#         save_seg_labels(osp.join(dst_l, f"{token}.seg"), seg)
#         make_png_scatter(osp.join(dst_i, f"{token}.png"), xy)

#         saved_pts_paths.append(pts_path)
#         new_tokens.append(f"{new_category_id}/{token}")

#     # update synset + split
#     with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
#         f.write(f"{new_category_name} {new_category_id}\n")

#     split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")
#     loaded = json.load(open(split_path)) if osp.isfile(split_path) else []
#     if not isinstance(loaded, list):
#         loaded = sum(loaded.values(), [])
#     loaded.extend(new_tokens)
#     json.dump(loaded, open(split_path, "w"), indent=2)

#     # visualization
#     if visualize and saved_pts_paths:
#         sample_paths = random.sample(saved_pts_paths, min(visualize_n, len(saved_pts_paths)))
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection="3d")
#         for path in sample_paths:
#             xyz = np.loadtxt(path)
#             if len(xyz) > 10000:
#                 xyz = xyz[np.random.choice(len(xyz), 10000, replace=False)]
#             ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.6)
#         plt.show()

#     print("[OK] Torus generation complete.")
def create_extra_category_torus(
    root_out,
    from_category_id="00000000",
    new_category_id="22222222",
    new_category_name="Torus",
    R_major=1.0,
    n_angles=100,
    visualize=False,
    visualize_n=10,
):
    import numpy as np, json, random
    import matplotlib.pyplot as plt
    import os, os.path as osp
    from tqdm import tqdm

    # paths
    src_points = osp.join(root_out, from_category_id, "points")
    dst_cat = ensure_dir(osp.join(root_out, new_category_id))
    dst_p = ensure_dir(osp.join(dst_cat, "points"))
    dst_l = ensure_dir(osp.join(dst_cat, "points_label"))
    dst_i = ensure_dir(osp.join(dst_cat, "seg_img"))

    tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])

    saved_pts_paths = []
    new_tokens = []

    for token in tqdm(tokens, desc="Generating torus"):

        # load unordered contour
        xy = np.loadtxt(osp.join(src_points, token + ".pts"))[:, :2]

        # total number of surface points
        # n_surface = len(xy) * n_angles
        n_surface = 3000

        # sample contour points
        idx = np.random.choice(len(xy), n_surface, replace=True)
        x = xy[idx, 0]
        y = xy[idx, 1]

        # shift for major radius
        x_shifted = x + R_major

        # random angles (fills surface, no rings)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n_surface)
        c = np.cos(phi)
        s = np.sin(phi)

        # torus surface (hollow)
        X = x_shifted * c
        Z = x_shifted * s
        Y = y

        pts = np.stack([X, Y, Z], axis=1)

        # labels
        seg = np.zeros(len(pts), dtype=int)

        # save
        pts_path = osp.join(dst_p, f"{token}.pts")
        save_pts_xyz(pts_path, pts)
        save_seg_labels(osp.join(dst_l, f"{token}.seg"), seg)
        make_png_scatter(osp.join(dst_i, f"{token}.png"), xy)

        saved_pts_paths.append(pts_path)
        new_tokens.append(f"{new_category_id}/{token}")

    # update synset + split
    with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
        f.write(f"{new_category_name} {new_category_id}\n")

    split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")
    loaded = json.load(open(split_path)) if osp.isfile(split_path) else []
    if not isinstance(loaded, list):
        loaded = sum(loaded.values(), [])
    loaded.extend(new_tokens)
    json.dump(loaded, open(split_path, "w"), indent=2)

    # visualization
    if visualize and saved_pts_paths:
        sample_paths = random.sample(saved_pts_paths, min(visualize_n, len(saved_pts_paths)))
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for path in sample_paths:
            xyz = np.loadtxt(path)
            if len(xyz) > 10000:
                xyz = xyz[np.random.choice(len(xyz), 10000, replace=False)]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.6)
        plt.show()

    print("[OK] Torus surface generation complete.")



# def create_extra_category_OT(
#     root_out: str,
#     from_category_id: str,
#     new_category_id: str,
#     new_category_name: str,
#     gmm_reference_path: str = "gmm_reference_cloud.npy",
#     reg: float = 0.05,   # Sinkhorn regularization
# ):
#     """
#     Crée une nouvelle catégorie par transport optimal DISCRET (Sinkhorn)
#     depuis les foils vers un nuage de référence échantillonné depuis le GMM.
#     """

#     import os, os.path as osp
#     import json
#     import numpy as np
#     from tqdm import tqdm
#     import ot   # POT = Python Optimal Transport

#     root_out = osp.abspath(root_out)

#     # --- Charger le nuage de référence GMM ---
#     Y = np.loadtxt(gmm_reference_path)[:, :3]
#     M = len(Y)

#     # --- Source folders ---
#     src_points = osp.join(root_out, from_category_id, "points")
#     src_labels = osp.join(root_out, from_category_id, "points_label")

#     if not osp.isdir(src_points):
#         raise FileNotFoundError(f"Catégorie source introuvable : {src_points}")

#     # --- Destination folders ---
#     cat_dir     = ensure_dir(osp.join(root_out, new_category_id))
#     points_dir  = ensure_dir(osp.join(cat_dir, "points"))
#     labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
#     segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))

#     print(f"\n=== Création catégorie OT (Discrete Sinkhorn) {new_category_name} ({new_category_id}) ===")

#     # --- Liste des objets ---
#     tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])
#     new_tokens = []

#     for token in tqdm(tokens, desc="OT Sinkhorn"):

#         # --- Load cloud ---
#         X = np.loadtxt(osp.join(src_points, token + ".pts"))[:, :3]
#         seg = np.loadtxt(osp.join(src_labels, token + ".seg")).astype(int)

#         N = len(X)

#         # --- Masses uniformes ---
#         a = np.ones(N) / N
#         b = np.ones(M) / M

#         # --- Cost matrix ---
#         C = ot.dist(X, Y, metric="euclidean") ** 2

#         # --- Sinkhorn OT ---
#         P = ot.sinkhorn(a, b, C, reg)

#         # --- Barycentric projection onto true support ---
#         X_ot = (P @ Y) / (P.sum(axis=1, keepdims=True) + 1e-12)

#         # --- Sauvegarde ---
#         save_pts_xyz(osp.join(points_dir, f"{token}.pts"), X_ot)
#         save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)
#         make_png_scatter(osp.join(segimg_dir, f"{token}.png"), X_ot[:, :2])

#         new_tokens.append(f"{new_category_id}/{token}")

#     # --- Ajout dans synsetoffset2category ---
#     with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
#         f.write(f"{new_category_name} {new_category_id}\n")

#     # --- Update test split ---
#     split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")

#     if osp.isfile(split_path):
#         with open(split_path, "r") as f:
#             loaded = json.load(f)
#     else:
#         loaded = []

#     flat = loaded if isinstance(loaded, list) else sum(loaded.values(), [])
#     flat.extend(new_tokens)

#     with open(split_path, "w") as f:
#         json.dump(flat, f, indent=2)

#     print(f"[OK] Ajouté {len(new_tokens)} objets OT Sinkhorn au test split.")
def create_extra_category_OT(
    root_out: str,
    from_category_id: str,
    new_category_id: str,
    new_category_name: str,
    gmm_reference_path: str = "gmm_reference_cloud.npy",
    reg: float = 0.05,
    n_src: int = 512,     # number of source points used for OT
    n_ref: int = 512,     # number of reference points
    numItermax: int = 300,
    stopThr: float = 1e-4,
    make_png: bool = True,
):
    """
    Create a new category using DISCRETE Sinkhorn OT.
    Optimized version:
      - downsampled X and Y
      - float32
      - fast squared Euclidean cost
      - stabilized Sinkhorn
    """

    import os, os.path as osp, json
    import numpy as np
    from tqdm import tqdm
    import ot

    root_out = osp.abspath(root_out)

    # ======================================================
    # Load & downsample reference cloud ONCE
    # ======================================================
    Y_full = np.loadtxt(gmm_reference_path)[:, :3]

    if len(Y_full) > n_ref:
        idx = np.random.choice(len(Y_full), n_ref, replace=False)
        Y = Y_full[idx]
    else:
        Y = Y_full

    Y = Y.astype(np.float32)
    M = Y.shape[0]
    b = np.full(M, 1.0 / M, dtype=np.float32)

    # ======================================================
    # Paths
    # ======================================================
    src_points = osp.join(root_out, from_category_id, "points")
    src_labels = osp.join(root_out, from_category_id, "points_label")

    if not osp.isdir(src_points):
        raise FileNotFoundError(f"Catégorie source introuvable : {src_points}")

    cat_dir    = ensure_dir(osp.join(root_out, new_category_id))
    points_dir = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir = ensure_dir(osp.join(cat_dir, "seg_img"))

    print(f"\n=== Création catégorie OT FAST {new_category_name} ({new_category_id}) ===")

    tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])
    new_tokens = []

    # ======================================================
    # Helper: fast squared distance
    # ======================================================
    def sqeuclidean_cost(X, Y):
        x2 = (X * X).sum(axis=1, keepdims=True)
        y2 = (Y * Y).sum(axis=1, keepdims=True).T
        C = x2 + y2 - 2.0 * (X @ Y.T)
        np.maximum(C, 0.0, out=C)
        return C

    # ======================================================
    # Main loop
    # ======================================================
    for token in tqdm(tokens, desc="OT Sinkhorn (fast)"):

        X_full = np.loadtxt(osp.join(src_points, token + ".pts"))[:, :3]
        seg    = np.loadtxt(osp.join(src_labels, token + ".seg")).astype(int)

        # ---- downsample source cloud ----
        if len(X_full) > n_src:
            idx = np.random.choice(len(X_full), n_src, replace=False)
            X = X_full[idx]
            seg = seg[idx]
        else:
            X = X_full

        X = X.astype(np.float32)
        N = X.shape[0]
        a = np.full(N, 1.0 / N, dtype=np.float32)

        # ---- cost + Sinkhorn ----
        C = sqeuclidean_cost(X, Y)

        P = ot.sinkhorn(
            a, b, C, reg,
            method="sinkhorn_stabilized",
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=False,
        ).astype(np.float32)

        # ---- barycentric projection ----
        X_ot = (P @ Y) / (P.sum(axis=1, keepdims=True) + 1e-12)

        # ---- save ----
        np.savetxt(
            osp.join(points_dir, f"{token}.pts"),
            X_ot,
            fmt="%.6f",
        )
        save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)

        if make_png:
            make_png_scatter(
                osp.join(segimg_dir, f"{token}.png"),
                X_ot[:, :2],
            )

        new_tokens.append(f"{new_category_id}/{token}")

    # ======================================================
    # Register category + update split
    # ======================================================
    with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
        f.write(f"{new_category_name} {new_category_id}\n")

    split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")

    if osp.isfile(split_path):
        loaded = json.load(open(split_path))
    else:
        loaded = []

    if not isinstance(loaded, list):
        loaded = sum(loaded.values(), [])

    loaded.extend(new_tokens)
    json.dump(loaded, open(split_path, "w"), indent=2)

    print(f"[OK] Ajouté {len(new_tokens)} objets OT (FAST) au test split.")


from matplotlib.path import Path

# def sample_points_inside_polygon(xy, n_samples):
#     """
#     Sample n_samples uniformly inside a 2D polygon.
#     """
#     poly = Path(xy)

#     xmin, ymin = xy.min(axis=0)
#     xmax, ymax = xy.max(axis=0)

#     samples = []
#     while len(samples) < n_samples:
#         pts = np.random.uniform(
#             low=(xmin, ymin),
#             high=(xmax, ymax),
#             size=(n_samples * 2, 2)
#         )
#         mask = poly.contains_points(pts)
#         samples.extend(pts[mask])

#     return np.array(samples[:n_samples])
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import numpy as np

def sample_points_inside_polygon(xy, n_samples):
    """
    Sample points inside an unordered 2D contour
    using its convex hull (always valid).
    """

    hull = ConvexHull(xy)
    hull_xy = xy[hull.vertices]

    poly = Path(hull_xy)

    xmin, ymin = hull_xy.min(axis=0)
    xmax, ymax = hull_xy.max(axis=0)

    samples = []
    while len(samples) < n_samples:
        pts = np.random.uniform(
            low=(xmin, ymin),
            high=(xmax, ymax),
            size=(n_samples * 2, 2)
        )
        mask = poly.contains_points(pts)
        samples.extend(pts[mask])

    return np.array(samples[:n_samples])


# def extrude_xy_solid_uniform(
#     xy: np.ndarray,
#     thickness: float = 0.1,
#     n_volume_points: int = 10000,
# ):
#     """
#     Create a FULL 3D solid volume from a 2D airfoil:
#     - interior filled uniformly
#     - z ~ Uniform(-thickness/2, +thickness/2)
#     """

#     # 1) Sample inside the polygon
#     xy_inside = sample_points_inside_polygon(xy, n_volume_points)

#     # 2) Uniform Z thickness
#     half = thickness / 2.0
#     z = np.random.uniform(-half, half, size=n_volume_points)

#     return np.c_[xy_inside, z]

def extrude_xy_surface_only(
    xy,
    thickness=0.1,
    n_points_total=3000,
    ratio_caps=0.4,
):
    """
    Extrude an UNORDERED 2D contour into a hollow solid.
    GUARANTEES no interior points for any z.
    """

    z_front = +thickness / 2
    z_back  = -thickness / 2

    n_caps_total = int(n_points_total * ratio_caps)
    n_caps = n_caps_total // 2
    n_sides = n_points_total - n_caps_total

    # ======================================================
    # 1) LATERAL SURFACES (SAFE: NO XY CREATION)
    # ======================================================

    idx = np.random.choice(len(xy), n_sides, replace=True)
    z = np.random.uniform(z_back, z_front, n_sides)

    # (x,y) ONLY from contour → cannot be interior
    sides = np.c_[xy[idx], z]

    # ======================================================
    # 2) TOP / BOTTOM FACES (FROM CONTOUR)
    # ======================================================

    cap_xy_top = sample_points_inside_polygon(xy, n_caps)
    cap_xy_bot = sample_points_inside_polygon(xy, n_caps)

    front = np.c_[cap_xy_top, np.full(n_caps, z_front)]
    back  = np.c_[cap_xy_bot, np.full(n_caps, z_back)]

    return np.vstack([sides, front, back])




def create_extra_category_extruded_solid(
    root_out: str,
    from_category_id: str = "00000000",
    new_category_id: str = "22222222",
    new_category_name: str = "ExtrudedSolid",
    thickness: float = 0.1,
    n_volume_points: int = 3000,
):
    """
    Create a FULL solid 3D volumetric category from 2D airfoils
    by filling the interior + uniform thickness.
    """

    root_out = osp.abspath(root_out)

    src_points = osp.join(root_out, from_category_id, "points")
    src_labels = osp.join(root_out, from_category_id, "points_label")

    if not osp.isdir(src_points):
        raise FileNotFoundError(f"Catégorie source introuvable : {src_points}")

    cat_dir     = ensure_dir(osp.join(root_out, new_category_id))
    points_dir  = ensure_dir(osp.join(cat_dir, "points"))
    labels_dir  = ensure_dir(osp.join(cat_dir, "points_label"))
    segimg_dir  = ensure_dir(osp.join(cat_dir, "seg_img"))

    print(f"\n=== Création catégorie VOLUMIQUE {new_category_name} ({new_category_id}) ===")

    tokens = sorted([f[:-4] for f in os.listdir(src_points) if f.endswith(".pts")])
    new_tokens = []

    for token in tqdm(tokens, desc=f"Extruding solid {new_category_name}"):

        xyz = np.loadtxt(osp.join(src_points, token + ".pts"))
        xy  = xyz[:, :2]

        # xyz_new = extrude_xy_solid_uniform(
        #     xy,
        #     thickness=thickness,
        #     n_volume_points=n_volume_points,
        # )
        xyz_new = extrude_xy_surface_only(
            xy,
            thickness=thickness,
            n_points_total=n_volume_points,  
        )
        seg = np.zeros(len(xyz_new), dtype=int)

        save_pts_xyz(osp.join(points_dir, f"{token}.pts"), xyz_new)
        save_seg_labels(osp.join(labels_dir, f"{token}.seg"), seg)
        make_png_scatter(osp.join(segimg_dir, f"{token}.png"), xy)

        new_tokens.append(f"{new_category_id}/{token}")

    with open(osp.join(root_out, "synsetoffset2category.txt"), "a") as f:
        f.write(f"{new_category_name} {new_category_id}\n")

    split_path = osp.join(root_out, "train_test_split", "shuffled_test_file_list.json")

    if osp.isfile(split_path):
        loaded = json.load(open(split_path))
    else:
        loaded = []

    if not isinstance(loaded, list):
        loaded = sum(loaded.values(), [])

    loaded.extend(new_tokens)
    json.dump(loaded, open(split_path, "w"), indent=2)

    print(f"[OK] Ajouté {len(new_tokens)} objets SOLIDES au test split.")
