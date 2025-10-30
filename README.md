# 🧠 CIFAR-10 Image Classification  

A deep learning project that classifies images from the **CIFAR-10** dataset using a **Multilayer Perceptron (MLP)** implemented with TensorFlow and Keras.  
The model is trained on 60,000 color images (32×32×3) across 10 object categories such as airplanes, cars, birds, and ships.

---

## 📘 Overview  

This project demonstrates how to recognize everyday objects using a fully connected neural network trained on the **CIFAR-10** dataset.  
Although **Convolutional Neural Networks (CNNs)** usually perform better for image data, this implementation shows how even a simple **feed-forward MLP** can learn basic visual patterns.  

It serves as an educational baseline for beginners exploring deep learning and computer vision with TensorFlow.

---

## 📊 Dataset  

The **CIFAR-10** dataset contains small RGB images classified into 10 categories:

| Class | Label |
|:------|:------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

- **Training samples:** 50,000  
- **Testing samples:** 10,000  
- **Image size:** 32×32 pixels, 3 color channels  

The dataset is automatically downloaded from the `tensorflow.keras.datasets` module.

---

## 🧠 Model Architecture  

The model is a **fully connected neural network (MLP)** built using Keras’ Sequential API.  
Each 32×32×3 image is flattened into a 3072-dimensional vector and passed through dense layers for classification.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(3072,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

