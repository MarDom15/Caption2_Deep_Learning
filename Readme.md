# 🌟 Medical Project: Classifying X-Ray Images for Pneumonia Detection

## 📋 Table of Contents
1. [Context](#-context)  
2. [Objective](#-objective)  
3. [Tools and Technologies](#-tools-and-technologies)  
4. [Project Steps](#-project-steps)  
   - [Creating and Managing a Google Cloud Account](#1-creating-and-managing-a-google-cloud-account)  
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)  
   - [Data Collection and Preparation](#3-data-collection-and-preparation)  
   - [Modeling](#4-modeling)  
   - [Model Evaluation](#5-model-evaluation)  
   - [Containerization with Docker](#6-containerization-with-docker)  
   - [Orchestration with Kubernetes](#7-orchestration-with-kubernetes)  
   - [Monitoring and Observability](#8-monitoring-and-observability)  
   - [Integrating CI/CD Pipeline](#9-integrating-cicd-pipeline)  
5. [Requirements File (`requirements.txt`)](#-requirements-file-requirementstxt)  
6. [README File (`README.md`)](#-readme-file-readmemd)  

---

## 🌐 Context
Automatic pneumonia detection from chest X-ray images can assist in faster diagnoses, especially in resource-limited environments. This project aims to build a deep learning model capable of classifying chest X-rays as "Pneumonia" or "Normal."

---

## 🎯 Objective
- Build a performant deep learning model.
- Integrate MLOps to ensure a reproducible and deployable workflow.
- Provide an interactive user interface to upload X-ray images and display results.
- Orchestrate and monitor services using **Kubernetes** and **Google Cloud**.

---

## 🛠️ Tools and Technologies
- **Python** for programming.
- **Keras/TensorFlow** for model development.
- **Docker** for containerization.
- **Kubernetes** for container orchestration.
- **Google Cloud Platform (GCP)** for deployment and monitoring.
- **Flask** or **Streamlit** for user interface.
- **Prometheus** and **Grafana** for monitoring.
- **Jenkins** for CI/CD pipelines.

---

## 📑 Project Steps

### 1️⃣ Creating and Managing a Google Cloud Account
1. Visit [Google Cloud Platform](https://cloud.google.com/).
2. Create a free account by clicking **Get Started for Free**.
3. Set up a project:
   - Access the **GCP Console**.
   - Click **Create Project**.
   - Name your project (e.g., `Pneumonia-Detection`).

4. Enable the following APIs:
   - **Kubernetes Engine API**.
   - **Cloud Storage API** for model storage.
   - **Monitoring API** for observability with Prometheus.

5. Install the Google Cloud SDK:
   ```bash
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

---

### 2️⃣ Exploratory Data Analysis (EDA)
Exploratory Data Analysis helps understand the structure and characteristics of the dataset.

#### Goals:
- Identify class distribution (Normal, Pneumonia - Viral, Pneumonia - Bacterial).
- Visualize sample images.
- Count the total number of images per category.

#### Sample EDA Code:
```python
import os
import matplotlib.pyplot as plt
from collections import Counter

# Define paths
data_dir = "data/train"
classes = os.listdir(data_dir)

# Count images per category
image_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

# Display results
print("Number of images per class:", image_counts)

# Visualize sample images
def plot_sample_images(data_dir, class_name, num_images=4):
    images = os.listdir(os.path.join(data_dir, class_name))[:num_images]
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img_name in enumerate(images):
        img_path = os.path.join(data_dir, class_name, img_name)
        img = plt.imread(img_path)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(class_name)
    plt.show()

# Visualize images for each class
for cls in classes:
    plot_sample_images(data_dir, cls)
```

---

### 3️⃣ Data Collection and Preparation
- Download the dataset from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- Organize files into three folders: `train/`, `test/`, and `validation/`.
- Preprocess images:
  - Resize them to `224x224 pixels`.
  - Normalize pixel values between `0 and 1`.
  - Augment data with transformations like rotation and zoom.

#### Sample Code
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')
```

---

### 4️⃣ Modeling
- Use a pre-trained model like **ResNet50** for Transfer Learning.
- Add custom final layers for binary classification.

#### Sample Code
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

---

## 5️⃣ Model Evaluation

After training the model, it's important to evaluate its performance using various metrics. Below are the key evaluation steps:

## 📊 Evaluation Metrics

1. **Accuracy**:  
   - **Definition**: The proportion of correctly classified instances over the total instances. It indicates how often the model is correct.
   - **Formula**:  
   [Accuracy Formula](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=fr)


2. **Precision**:  
   - **Definition**: The ratio of correctly predicted positive observations to the total predicted positives. It answers the question: *Of all the instances predicted as positive, how many were actually positive?*
   - **Formula**:  
     $$
     \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
     $$

3. **Recall (Sensitivity)**:  
   - **Definition**: The ratio of correctly predicted positive observations to all observations in the actual positive class. It answers the question: *Of all the actual positive instances, how many were correctly identified?*
   - **Formula**:  
     $$
     \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
     $$

4. **F1-Score**:  
   - **Definition**: The weighted average of precision and recall, which balances the two metrics. A high F1-Score indicates both high precision and recall.
   - **Formula**:  
     $$
     \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
     $$

5. **Confusion Matrix**:  
   - **Definition**: A summary of prediction results that helps to identify false positives and false negatives. It provides a clear view of the classifier's performance across all classes.
   - **Components**:
     - True Positives (TP): Correctly predicted positive instances.
     - True Negatives (TN): Correctly predicted negative instances.
     - False Positives (FP): Negative instances incorrectly predicted as positive.
     - False Negatives (FN): Positive instances incorrectly predicted as negative.
   - **Formula**:  
     $$
     \begin{matrix}
     & \text{Predicted Normal} & \text{Predicted Pneumonia} \\
     \text{Actual Normal} & TN & FP \\
     \text{Actual Pneumonia} & FN & TP
     \end{matrix}
     $$

6. **ROC Curve & AUC (Area Under the Curve)**:  
   - **Definition**: The Receiver Operating Characteristic curve (ROC) is a graphical representation of a classifier's performance at various thresholds. The AUC is the area under the ROC curve, which quantifies the overall ability of the model to discriminate between positive and negative classes. A higher AUC indicates better performance.
   - **ROC Curve**: Plots the true positive rate (TPR) against the false positive rate (FPR).
   - **AUC**: Measures the entire two-dimensional area underneath the ROC curve.
   - **Formula**:  
     $$
     \text{AUC} = \int_0^1 \text{True Positive Rate} \, d(\text{False Positive Rate})
     $$

---

## 📈 Sample Code for Evaluation

The following Python code demonstrates how to compute and visualize the evaluation metrics for the trained model:

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Predict on test set
y_pred = model.predict(test_generator)
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred.round())
print("Confusion Matrix:")
print(cm)

# Classification Report (includes precision, recall, f1-score)
print("Classification Report:")
print(classification_report(y_true, y_pred.round()))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
---

### 6️⃣ Containerization with Docker

1. **Create a `Dockerfile`:**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt requirements.txt
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```
2. Build the Docker image:
   ```bash
   docker build -t pneumonia-classifier .
   ```

3. Test the container locally:
   ```bash
   docker run -p 5000:5000 pneumonia-classifier
   ```

---

### 7️⃣ Orchestration with Kubernetes
1. Install **kubectl** and configure the Kubernetes cluster on GCP:
   ```bash
   gcloud container clusters create pneumonia-cluster \
       --num-nodes=3 \
       --zone=us-central1-a
gcloud container clusters get-credentials pneumonia-cluster \
       --zone us-central1-a
   ```

2. Deploy the application:
   - Create a `deployment.yaml` file:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pneumonia-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pneumonia-app
  template:
    metadata:
      labels:
        app: pneumonia-app
    spec:
      containers:
      - name: pneumonia-app
        image: gcr.io/<your-project>/pneumonia-classifier:latest
        ports:
        - containerPort: 5000
```

3. Apply the deployment:
   ```bash
   kubectl apply -f deployment.yaml
   ```

4. Expose the service:
   ```bash
   kubectl expose deployment pneumonia-app --type=LoadBalancer --port=80 --target-port=5000
   ```

---

### 8️⃣ Monitoring and Observability
- Install **Prometheus** and **Grafana** in the Kubernetes cluster for monitoring.
- Set up dashboards to track:
  - Resource usage (CPU, memory).
  - Application response times.

---

🌐 **9️⃣ Accessing the Application**

Once deployed, you can access the application via the external IP address of the Kubernetes service or using the **LoadBalancer** setup. If you deployed it via **Google Cloud Kubernetes**, you can retrieve the external IP using:

```bash
kubectl get svc pneumonia-app
```
---

🚀 **🔟 Vertex AI Integration**

To improve the deployment and scaling of your model, you can integrate Vertex AI for model management and serving.

1. **Deploy your trained model to Vertex AI**:  
   Use **Vertex AI Workbench** to train models, or directly upload your trained model to Vertex AI for serving.  
   Follow the Vertex AI model deployment guide for a seamless deployment.

2. **Using Vertex AI for prediction**:  
   You can also set up a prediction pipeline on **Vertex AI** and make it accessible to your application for real-time inference.

---
# CI/CD Pipeline with Jenkins

## 🚀 1️⃣1️⃣ CI/CD Integration with Jenkins

Jenkins is a powerful tool for continuous integration and delivery. Here’s how to set up a CI/CD pipeline with Jenkins to automate the deployment of your pneumonia detection project.

---

## Installing Jenkins

### Option 1: Local Installation
```bash
sudo apt update
sudo apt install openjdk-11-jdk -y
wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo apt-key add -
sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
sudo apt update
sudo apt install jenkins -y
sudo systemctl start jenkins
sudo systemctl enable jenkins
```

### Option 2: Run Jenkins in a Docker Container
```bash
docker run -d -p 8080:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home jenkins/jenkins:lts
```

### Accessing the Jenkins Dashboard
Open your browser and navigate to `http://<your-server-ip>:8080`.

Retrieve the initial admin password:
```bash
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

Follow the instructions to complete the setup and install the recommended plugins.

### Configuring Required Plugins
Install the following plugins via Jenkins:
- Docker Pipeline
- Git
- Kubernetes CLI
- Pipeline: Stage View
- Blue Ocean (for an interactive pipeline view)

---

## Creating a CI/CD Pipeline

Create a new pipeline in Jenkins and configure the following `Jenkinsfile` in your Git repository:

### Jenkinsfile Example
```groovy
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "pneumonia-classifier"
        DOCKER_TAG = "latest"
        REGISTRY = "gcr.io/<your-project-id>"
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/your-repo/pneumonia-project.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pytest tests/'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_IMAGE:$DOCKER_TAG .'
            }
        }

        stage('Push Docker Image to Registry') {
            steps {
                sh 'docker tag $DOCKER_IMAGE:$DOCKER_TAG $REGISTRY/$DOCKER_IMAGE:$DOCKER_TAG'
                sh 'docker push $REGISTRY/$DOCKER_IMAGE:$DOCKER_TAG'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/logs/*.log', allowEmptyArchive: true
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
```

---

## Pipeline Steps

1. **Clone Repository**: Downloads the code from the GitHub repository.
2. **Install Dependencies**: Installs all necessary libraries using the `requirements.txt` file.
3. **Run Tests**: Runs unit tests with `pytest`.
4. **Build Docker Image**: Builds a Docker image for the application.
5. **Push Docker Image**: Publishes the Docker image to Google Container Registry (GCR).
6. **Deploy to Kubernetes**: Updates the Kubernetes deployment with the new image.

---

## Additional Configuration

### Docker Permissions for Jenkins
If Jenkins is installed locally, ensure it can access the Docker socket:
```bash
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

### Adding Kubernetes Credentials to Jenkins
Add the necessary configurations to allow Jenkins to deploy to your Kubernetes cluster using `kubectl`.

### GitHub Webhook Integration
Set up a webhook in your GitHub repository to automatically trigger Jenkins builds on every push:

1. Go to your repository settings on GitHub.
2. Under **Webhooks**, add a new webhook:
   - **Payload URL**: `http://<jenkins-server>/github-webhook/`
   - **Content type**: `application/json`
   - Enable **Push** events.

---

## Build Visualization and Monitoring

- View logs and pipeline status directly in Jenkins.
- Configure notifications (email, Slack) to stay informed about successes and failures.

---

## Final Outcome

Once the pipeline is set up, each change pushed to the repository will trigger a CI/CD pipeline that:

1. Tests the code.
2. Builds a new Docker image.
3. Automatically deploys the application to Kubernetes.
4. Provides real-time tracking via Jenkins.


---



## 📜 Requirements File (`requirements.txt`)
Include all Python dependencies:
```plaintext
flask
numpy
tensorflow
pandas
matplotlib
scikit-learn
```

---

## 📖 README File (`README.md`)

```markdown
# X-Ray Image Classification for Pneumonia Detection

## Description
This project aims to detect pneumonia from chest X-rays using deep learning and integrated MLOps practices.

## Features
- ResNet50-based model for binary classification.
- Interactive diagnostic interface.
- Scalable deployment with Docker and Kubernetes.
- Monitoring with Prometheus and Grafana.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Build the Docker image:
   ```bash
   docker build -t pneumonia-classifier .
   ```
3. Run the application:
   ```bash
   docker run -p 5000:5000 pneumonia-classifier
   ```

## Deployment on Google Cloud
1. Set up a Kubernetes cluster on Google Cloud.
2. Deploy the application with `kubectl`.
3. Monitor performance with Grafana.

## Contribution
Contributions are welcome. Please open an issue or submit a pull request.

## License
MIT License
```
project in progress 

[def]: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=fr