import cv2
import numpy as np
import random

class Generate_Data:
    def __init__(self):
        # 1. Load Background
        self.bg_path = r"Backround.jpg" # Make sure this matches your file extension (.png/.jpg)
        self._backround = cv2.imread(self.bg_path)
        
        if self._backround is None:
            raise FileNotFoundError(f"Could not load {self.bg_path}")
            
        h, w = self._backround.shape[:2]
        print(f"--- Background Size: {w}x{h} ---")
        
        # This mask tracks where coins already exist (White = Occupied, Black = Empty)
        self._occupied_mask = np.zeros((h, w), dtype=np.uint8)

        # 2. Load Coins (Isolate the coin, but don't damage pixel data)
        self._yellow_coin = self._extract_coin_content(r"Coins/no_green_Yellow_coin.png")
        self._blue_coin = self._extract_coin_content(r"Coins/no_green_Blue_coin.png")
        self._red_coin = self._extract_coin_content(r"Coins/no_green_Red_coin.png")

    def _extract_coin_content(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: Could not load {path}")
            return None

        # === CREATE ALPHA FROM BLACK BACKGROUND ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Anything NOT near-black becomes visible
        _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Clean small noise
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

        # Merge into BGRA
        b, g, r = cv2.split(img)
        img_bgra = cv2.merge((b, g, r, alpha))

        # === FIND TIGHT ROI USING ALPHA ===
        points = cv2.findNonZero(alpha)
        if points is None:
            return img_bgra

        x, y, w, h = cv2.boundingRect(points)
        coin_roi = img_bgra[y:y+h, x:x+w]

        print(f"Loaded {path}: ROI Found -> {w}x{h}")
        return coin_roi


    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 1. Rotation Matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 2. Calculate New Canvas Size (so corners don't get cut off)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 3. Adjust Matrix for Translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # 4. Perform Rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0,0,0,0))
        return rotated

    def add_coin_randomly(self, coin_img_original, max_attempts=200):
        if coin_img_original is None: return False

        bg_h, bg_w = self._backround.shape[:2]

        for attempt in range(max_attempts):
            # 1. Rotate the coin (and its hitbox)
            angle = random.randint(0, 360)
            coin_img = self.rotate_image(coin_img_original, angle)
            
            h, w = coin_img.shape[:2]
            
            # If coin is physically larger than background, we can't place it
            if w >= bg_w or h >= bg_h:
                continue 

            # 2. Generate the "Hitbox" (Mask) for this specific rotation
            # We look at the Alpha channel. White = Solid Coin, Black = Empty.
            if coin_img.shape[2] == 4:
                # Any pixel that is visibly opaque enough to matter
                _, coin_mask = cv2.threshold(coin_img[:, :, 3], 10, 255, cv2.THRESH_BINARY)
            else:
                coin_mask = 255 * np.ones((h, w), dtype=np.uint8)

            # 3. Pick a random spot
            x = random.randint(0, bg_w - w)
            y = random.randint(0, bg_h - h)

            # 4. CHECK COLLISIONS
            # Get the region of the 'Occupied Mask' where we want to put the coin
            bg_mask_region = self._occupied_mask[y:y+h, x:x+w]
            
            # Bitwise AND: Checks if White overlaps White
            intersection = cv2.bitwise_and(bg_mask_region, coin_mask)
            
            if np.any(intersection):
                # We hit something! Try again.
                continue 
            
            # 5. Success! Place the visual image AND update the occupied mask
            self._place_pixels(coin_img, x, y)
            
            # Update the global mask so future coins know this spot is taken
            # Bitwise OR: Adds the new coin's shape to the existing mask
            self._occupied_mask[y:y+h, x:x+w] = cv2.bitwise_or(bg_mask_region, coin_mask)
            
            return True

        print("Failed to place coin (No valid spot found).")
        return False

    def _place_pixels(self, coin_img, x, y):
        h, w = coin_img.shape[:2]
        bg_slice = self._backround[y:y+h, x:x+w]
        coin_bgr = coin_img[:, :, :3]
        
        # Standard Alpha Blending
        if coin_img.shape[2] == 4:
            alpha = coin_img[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            
            for c in range(3):
                bg_slice[:, :, c] = (alpha * coin_bgr[:, :, c] + 
                                     alpha_inv * bg_slice[:, :, c])
        else:
            self._backround[y:y+h, x:x+w] = coin_bgr

    def save_image(self, filename):
        cv2.imwrite(filename, self._backround)
        # Optional: Save the mask to verify overlap logic works
        # cv2.imwrite("Debug_Map.png", self._occupied_mask)
        print(f"Saved {filename}")

if __name__ == "__main__":
    gen = Generate_Data()
    
    # Try to place multiple coins
    coins = [gen._yellow_coin, gen._blue_coin, gen._red_coin, gen._yellow_coin, gen._blue_coin]
    
    count = 0
    for coin in coins:
        if gen.add_coin_randomly(coin):
            count += 1
            
    print(f"Total coins placed: {count}/{len(coins)}")
    gen.save_image("Generated_Image.png")