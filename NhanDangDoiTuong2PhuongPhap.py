import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes là số lượng đối tượng
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Đánh giá
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")


# Tải mô hình Faster R-CNN pre-trained trên COCO dataset
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Dự đoán trên ảnh
image = Image.open('test_image.jpg').convert('RGB')
image_tensor = F.to_tensor(image).unsqueeze(0)

start_time = time.time()
predictions = model(image_tensor)  # Dự đoán bounding boxes và labels
elapsed_time = time.time() - start_time

# In kết quả
for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > 0.5:  # Ngưỡng confidence score
        print(f"Box: {box}, Label: {label}, Score: {score}")

print(f"Prediction Time: {elapsed_time}")


# Đọc annotation và prediction
coco_gt = COCO('ground_truth.json')
coco_dt = coco_gt.loadRes('predictions.json')

# Đánh giá
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

