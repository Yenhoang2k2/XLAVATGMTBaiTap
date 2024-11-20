import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os

# Thiết lập ImageDataGenerator cho Train và Validation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load ảnh từ thư mục
train_data = train_datagen.flow_from_directory(
    'Data/Train/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'  # One-hot encoding
)

val_data = val_datagen.flow_from_directory(
    'Data/Validation/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)



# Xây dựng ANN
ann_model = Sequential([
    Flatten(input_shape=(128, 128, 3)),  # Dàn phẳng ảnh
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Output 2 lớp
])

# Biên dịch mô hình
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện
ann_model.fit(train_data, epochs=20, validation_data=val_data)


# Xây dựng CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Output 2 lớp
])

# Biên dịch và huấn luyện
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_data, epochs=20, validation_data=val_data)


# Load mô hình Faster R-CNN pretrained
model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=3)  # 3 lớp: Background, Dog, Cat
model.train()

# Dataset và DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.images = [os.path.join(folder, img) for img in os.listdir(folder)]
        self.transform = F.to_tensor

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        target = ...  # Load annotations cho ảnh (bounding boxes + labels)
        return image_tensor, target

    def __len__(self):
        return len(self.images)

train_dataset = CustomDataset('Data/Train/')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Training loop (rút gọn)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(10):
    for images, targets in train_loader:
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


val_loss, val_acc = cnn_model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc}")

# ANN và CNN

img = image.load_img('test_image.jpg', target_size=(128, 128))
img_array = np.expand_dims(image.img_to_array(img) / 255., axis=0)

# Dự đoán
pred = cnn_model.predict(img_array)
print(f"Prediction: {'Dog' if np.argmax(pred) == 0 else 'Cat'}")

# R-CNN
image = Image.open('test_image.jpg').convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0)
predictions = model(image_tensor)
for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > 0.5:
        print(f"Box: {box}, Label: {label}, Score: {score}")

