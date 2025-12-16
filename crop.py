import cv2
import numpy as np
import os

def clean_alpha_noise(image_path):
    # Load image with Alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # Check for alpha
    if img.shape[2] != 4:
        print(f"Skipping {image_path}: No alpha channel found.")
        return

    print(f"Processing {os.path.basename(image_path)}...")

    # 1. ISOLATE ALPHA CHANNEL
    alpha = img[:, :, 3]
    
    # 2. THRESHOLD THE ALPHA
    # "If a pixel is barely visible (alpha < 30), make it FULLY invisible (0)."
    # "If it is visible (alpha >= 30), keep it exactly as it is."
    # We use 'THRESH_TOZERO' which keeps values above the threshold unchanged
    # unlike 'THRESH_BINARY' which forces them to 255.
    _, clean_alpha = cv2.threshold(alpha, 30, 255, cv2.THRESH_TOZERO)

    # Apply this cleaner alpha back to the image
    img[:, :, 3] = clean_alpha

    # 3. CROP BASED ON NEW ALPHA
    # Find all pixels that are not 0
    points = cv2.findNonZero(clean_alpha)
    
    if points is None:
        print("  WARNING: Image became invisible!")
        return

    # Get tight box
    x, y, w, h = cv2.boundingRect(points)
    
    # Crop
    cropped_img = img[y:y+h, x:x+w]

    # Stats
    original_area = img.shape[0] * img.shape[1]
    new_area = w * h
    reduction = 100 - (new_area / original_area * 100)

    print(f"  Cropped: {img.shape[1]}x{img.shape[0]} -> {w}x{h} (Reduced by {reduction:.1f}%)")
    
    # 4. SAVE
    cv2.imwrite(image_path, cropped_img)
    print("  -> Saved.")

# --- RUN ---
coin_files = [
    r"Coins/no_green_Yellow_coin.png",
    r"Coins/no_green_Blue_coin.png",
    r"Coins/no_green_Red_coin.png"
]

for coin in coin_files:
    clean_alpha_noise(coin)