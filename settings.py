"""
Configuration settings for YOLO Fire Detector Dataset Generator

Contains three main configuration classes:
- ImageTransformSettings: Parameters for image augmentation and transformations
- DatasetGenerationSettings: Parameters for dataset generation (paths, sizes, splits)
- ViewerSettings: Parameters for dataset visualization
"""


class ImageTransformSettings:
    """Impostazioni per le trasformazioni geometriche e fotometriche."""
    
    # Rotazioni / Geometria
    ROTATION_DEG_MIN = -180
    ROTATION_DEG_MAX = 180
    PERSPECTIVE_SHIFT = 120  # distorsione prospettica
    
    # Luminosità / Contrasto
    BRIGHTNESS_BETA_MIN = -50
    BRIGHTNESS_BETA_MAX = 50
    CONTRAST_ALPHA_MIN = 0.60
    CONTRAST_ALPHA_MAX = 1.40
    
    # Colore
    ENABLE_COLOR_SHIFT = True
    COLOR_SHIFT_PROB = 0.35
    COLOR_SHIFT_HUE_MAX = 20
    
    # Blur
    MOTION_BLUR_PROB = 0.40
    MOTION_BLUR_KERNEL_CHOICES = [5, 7, 9, 11]
    GAUSSIAN_BLUR_PROB = 0.20
    GAUSSIAN_BLUR_KERNEL_CHOICES = [3, 5]
    
    # Rumore
    NOISE_PROB = 0.25
    NOISE_LEVEL_MIN = 5
    NOISE_LEVEL_MAX = 30
    
    # Ombra
    SHADOW_PROB = 0.50
    SHADOW_ALPHA_MIN = 0.20
    SHADOW_ALPHA_MAX = 0.55
    
    # Occlusione
    OCCLUSION_PROB = 0.30
    OCCLUSION_COVERAGE_MIN = 0.10
    OCCLUSION_COVERAGE_MAX = 0.35
    
    # Augmentazione sfondi negativi
    AUGMENT_NEGATIVE_BACKGROUNDS = True


class DatasetGenerationSettings:
    """Impostazioni per la generazione del dataset."""
    
    # Path
    FIRE_IMAGE_PATH = r"fire.png"
    DATASET_ROOT = "dataset"
    
    # Dataset
    NUM_IMAGES = 2000
    TRAIN_SPLIT = 0.8          # 80% train, 20% val
    NEGATIVE_RATIO = 0.35      # percentuale immagini senza fuoco
    
    # Dimensioni
    IMAGE_SIZE = 640
    
    # Dimensione fuoco
    FIRE_SCALE_MIN = 0.05
    FIRE_SCALE_MAX = 0.50
    
    # Demo
    DEMO_MODE = False
    DEMO_WAIT_MS = 150


class ViewerSettings:
    """Impostazioni per il visualizzatore del dataset."""
    
    # Dataset
    DATASET_ROOT = "dataset"
    SPLIT = "train"  # "train" oppure "val"
    
    # Visualizzazione
    NUM_SAMPLES = 9       # quante immagini mostrare
    THUMB_SIZE = 280      # dimensione miniatura
    DRAW_TITLE = True     # scrive il nome file sopra
