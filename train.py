import os
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import kagglehub

# Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("tanlikesmath/diabetic-retinopathy-resized")
print("Path to dataset files:", path)

# Define paths
dataset_path = path
cropped_images_path = os.path.join(dataset_path, "resized_train_cropped/resized_train_cropped")
csv_path = os.path.join(dataset_path, "trainLabels_cropped.csv")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(csv_path)
print("CSV Columns:", df.columns.tolist())  # Print column names for debugging

# Drop unnecessary columns
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

# Ensure only required columns are kept
expected_columns = ["image", "level"]
df = df[[col for col in expected_columns if col in df.columns]]

df["image"] = df["image"].astype(str) + ".jpeg"

# Split dataset into train (80%), validation (10%), and test (10%)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['level'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['level'], random_state=42)

# Create directories for train, validation, and test datasets
base_dir = "diabetic_retinopathy_data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

for dir_path in [train_dir, val_dir, test_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    for label in df['level'].unique():
        os.makedirs(os.path.join(dir_path, str(label)))

# Function to move images
def move_images(df_subset, target_dir):
    missing_images = 0
    copied_images = 0
    for _, row in df_subset.iterrows():
        src = os.path.join(cropped_images_path, row['image'])
        dst = os.path.join(target_dir, str(row['level']), row['image'])
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied_images += 1
        else:
            missing_images += 1
    print(f"Copied {copied_images} images to {target_dir}, Missing: {missing_images}")

print("Organizing dataset...")
move_images(train_df, train_dir)
move_images(val_df, val_dir)
move_images(test_df, test_dir)

# Debugging: Check if images exist after copying
print("Train directory contains:", len(os.listdir(train_dir)))
print("Validation directory contains:", len(os.listdir(val_dir)))
print("Test directory contains:", len(os.listdir(test_dir)))

# Define ImageDataGenerator for data augmentation
gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)

train_gen = gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
val_gen = gen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
test_gen = gen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', shuffle=False)

# Load Pretrained Model (EfficientNetB0)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Custom Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(len(df['level'].unique()), activation='softmax')(x)  # Multi-class classification

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
print("Training model...")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save Model
model.save("diabetic_retinopathy_model.h5")
print("Model training complete and saved as diabetic_retinopathy_model.h5")
