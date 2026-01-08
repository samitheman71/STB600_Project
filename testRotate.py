import cv2
import numpy as np
import matplotlib.pyplot as plt

MINAREA = 6000

def find_brick_contours(img, mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vis = img.copy()
    bricks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MINAREA:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if not (1.2 < aspect < 4.0):
            continue

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)

        bricks.append({
            "contour": cnt,
            "rect": rect,
            "area": area
        })

    return vis, bricks

def long_edge_vertical_angle(rect):
    (_, _), (w, h), angle = rect

    if w > h:
        return -angle
    else:
        return -(angle - 90)




def rotate_and_crop(img, cnt):
    rect = cv2.minAreaRect(cnt)
    angle = long_edge_vertical_angle(rect)
    center = rect[0]

    center = rect[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, img.shape[1::-1],
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    ones = np.ones((4,1))
    pts = np.hstack([box, ones])
    pts = (M @ pts.T).T.astype(int)

    x, y, w, h = cv2.boundingRect(pts)
    x = max(0, x)
    y = max(0, y)
    x2 = min(rotated.shape[1], x + w)
    y2 = min(rotated.shape[0], y + h)

    return rotated[y:y2, x:x2]

img = cv2.imread("Blue.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

binary_t = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    5
)

edges = cv2.Canny(gray, 80, 160)
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

binary = cv2.bitwise_or(binary_t, edges)

h, w = gray.shape
k = max(5, int(0.02 * w))
if k % 2 == 0:
    k += 1

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


vis, bricks = find_brick_contours(img, binary)

print(f"Bricks found: {len(bricks)}")

plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title("Detected bricks")
plt.axis("off")
plt.show()

plt.figure(figsize=(12,3))
plt.subplot(1,4,1); plt.imshow(gray, cmap='gray'); plt.title("Gray")
plt.subplot(1,4,2); plt.imshow(binary, cmap='gray'); plt.title("Binary")
plt.subplot(1,4,4); plt.imshow(vis[...,::-1]); plt.title("Contours")
plt.show()

for i, brick in enumerate(bricks):
    cnt = brick["contour"]
    rect = brick["rect"]

    rotated_brick = rotate_and_crop(img, cnt)

    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rotated_brick, cv2.COLOR_BGR2RGB))
    plt.title(f"Rotated brick #{i}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Original image with contours")
    plt.axis("off")

    


    plt.show()
