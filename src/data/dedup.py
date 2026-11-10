"""Signature perceptuelle (pHash) et regroupement des quasi-doublons.

Pas d'identifiant de lame/patient dans le dataset : on approche la règle du cahier des
charges en regroupant les images quasi identiques (même champ, léger décalage, recompression)
pour qu'elles ne soient jamais réparties entre train / val / test.
"""
from __future__ import annotations

import cv2
import numpy as np


def phash(gray: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    """pHash : DCT d'une miniature, bits = coefficients basses fréquences > médiane.

    gray : image niveaux de gris uint8 (H, W). Retourne un vecteur bool de hash_size**2 bits.
    """
    size = hash_size * highfreq
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:hash_size, :hash_size].flatten()
    return low > np.median(low[1:])  # médiane hors composante continue


def bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits).tobytes().hex()


def hex_to_bits(h: str, n_bits: int = 64) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))[:n_bits].astype(bool)


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Distances de Hamming entre toutes les paires. bits : (n, n_bits) bool."""
    b = bits.astype(np.int32)
    same = b @ b.T + (1 - b) @ (1 - b).T
    return bits.shape[1] - same


def group_near_duplicates(bits: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Composantes connexes du graphe « distance de Hamming <= threshold ».

    Retourne (labels, D) : labels[i] = indice de groupe (0..k-1, numérotés selon le plus petit
    indice de membre, donc déterministes si les images sont triées), D = matrice des distances.
    Attention : le chaînage (A~B, B~C => A,B,C) peut fabriquer de gros groupes si le seuil
    est trop haut : toujours regarder la taille des groupes.
    """
    n = len(bits)
    D = hamming_matrix(bits)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(D <= threshold, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels, D


def nearest_neighbor_distance(D: np.ndarray) -> np.ndarray:
    """Distance au plus proche voisin pour chaque image (aide au choix du seuil)."""
    D2 = D.astype(float).copy()
    np.fill_diagonal(D2, np.inf)
    return D2.min(axis=1)