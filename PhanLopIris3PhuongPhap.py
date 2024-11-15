import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.svm import SVC


# Xây dựng mô hình ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(set(y_train)), activation='softmax')  # Số lớp
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện ANN
start_time = time.time()
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
training_time = time.time() - start_time

# Đánh giá
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"ANN Accuracy: {test_acc}")
print(f"Training Time: {training_time}")

# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=5)
start_time = time.time()
knn.fit(X_train, y_train)
training_time = time.time() - start_time

# Dự đoán
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"KNN Accuracy: {accuracy}")
print(f"Training Time: {training_time}")



# Huấn luyện SVM
svm = SVC(kernel='rbf', C=1, gamma='scale')
start_time = time.time()
svm.fit(X_train, y_train)
training_time = time.time() - start_time

# Dự đoán
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM Accuracy: {accuracy}")
print(f"Training Time: {training_time}")


