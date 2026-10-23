# Conventions des modules

## Entrée
Image chargée en tableau NumPy (RGB, uint8), ou chemin vers le fichier.

## Sortie du Quality Analyzer : `analyze_image(image) -> dict`
```python
{
    "sharpness": float,
    "brightness": float,
    "contrast": float,
    "overexposure_ratio": float,
    "color_statistics": dict,
}
```

## Sortie de la détection de candidats : liste de dict
```python
{
    "id": int,
    "bbox": (x, y, w, h),
    "area": float,
    "perimeter": float,
    "circularity": float,   # 4*pi*A / P^2
}
```
