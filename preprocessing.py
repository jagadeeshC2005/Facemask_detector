import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ================= Settings =================
IMG_SIZE = 128
DATASET_DIR = "dataset"   # inside this, keep "with_mask" and "without_mask"

CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

print("🔄 Starting preprocessing...")

# ================= Load Images =================
for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    class_label = CATEGORIES.index(category)  # 0=with_mask, 1=without_mask

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(class_label)
        except Exception as e:
            print("⚠️ Skipping image:", img_name, "| Error:", e)

# ================= Convert to Arrays =================
data = np.array(data) / 255.0   # normalize
labels = np.array(labels)

print(f"✅ Loaded {len(data)} images")

# ================= Train/Test Split =================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# ================= Save =================
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Preprocessing complete")
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")