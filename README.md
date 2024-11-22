<h1 align="center">Brain Tumor Classification</h1>

An **end-to-end pipeline** for brain tumor prediction using **deep learning** from MRI scanner images. The pipeline covers data preprocessing, model training, and evaluation to classify and detect brain tumors. It enables automated, accurate tumor identification for enhanced clinical decision-making.

### Table of Contents
- [About The Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Training Pipeline](#training-pipeline)
- [Prediction Pipeline](#prediction-pipeline)
- [Deployment](#deployment)
- [Web Application](#web-application)
- [AWS CI/CD Deployment](#aws-cicd-deployment)
- [Export Environment Variables](#export-environment-variables)
- [Setup Github Secrets](#setup-github-secrets)

### About The Project:


The project encompasses **image preprocessing**, **model training**, **evaluation**, and **deployment** to **Azure** using **DVC** for **MLOps**, integrated with **Docker** and automated through **GitHub Actions**.


### Tech Stack:

- **Programming Language**: Python
- **Framework**: Tensorflow/Keras
- **Version Control**: DVC
- **Containerization**: Docker
- **Cloud Services**:
  - **Azure App Service**: Hosting the application
  - **Azure Container Registry (ACR)**: Docker image repository
- **CI/CD**: GitHub Actions
- **Web Application**: HTML

### How to Run:

Instructions to set up your local environment for running the project

1. Clone the repository
   ```bash
   git clone https://github.com/Sirikrrishna/braintumor.git
   cd braintumor
2. Set up a virtual environment
   ```bash
   conda create -n mlproject_env python=3.8 -y
   conda activate mlproject_env
3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   #run
   dvc repro
   python app.py

### Training Pipeline:

#### Data Ingestion
- Split the data into train and test sets.
- Preprocess data into a format suitable for deep learning models
  
#### Data Transformation
- Transform raw data into a suitable format for model building.
- Perform augmentation techniques such as rotation, flipping, and scaling.
- Normalize image data for training.

#### Model Trainer and Evaluation
- Utilised pre trained Deep Learning model **VGG16**.
- Tuned HyperParameters, including **Learning Rate**, **Epoch**, **Batch Size** to optimize model performance.
- A batch size of **16** is used for training, with the top layers of the model excluded for fine-tuning.
- Used the **ImageNet** weights for transfer learning.

### Prediction Pipeline:
- Execute the training pipeline for data processing, model training, evaluation, and deployment, using **DVC** for version control and reproducibility across stages.

### Deployment:
#### Containerize the Application
- Use **Docker** to containerize the application for easy deployment and scalability.
  ```bash
  docker build -t brain_tumor_classifier .

#### Deploy on Azure App Service:
- Push the Docker image to **Azure Container Registry (ACR)**.
- Link the container from **ACR** to an **Azure App Service**.

#### Automate Deployment with GitHub Actions
- Use **GitHub Actions** to automate the deployment workflow.
- On each code push:
  - Retrain the model using **DVC**.
  - Build the Docker image.
  - Push the image to **ACR**.
  - Deploy the container to **Azure App Service**.

### Web Application:
- Build a basic web application using **FLASK** and **HTML**,**CSS** to expose the model's prediction functionality.
- The web app allows users to Upload brain MRI images and receive predictions about the tumor class.
- Ensure that the front-end is user-friendly and responsive to enhance user experience.

###  Azure CI/CD Deployment:
1. Login to the **Azure Portal**.
2. Create an **Azure Container Registry**:
   - Store **Docker** images.
   - Retrieve the **ACR Login Server** URI.
3. Create an **Azure App Service**:
   - Select the **ACR image** to host the application.
4. Configure GitHub Actions:
   - Automate building, pushing, and deploying the application on Azure.
   

#### Required Permissions:
- Ensure your Azure service principal has access to ACR and App Service..


### Export Environment Variables:

Before running your application, make sure to export the following environment variables in your terminal:

       ```bash
       export AZURE_STORAGE_CONNECTION_STRING="<Azure Blob Storage connection string>"
       export AZURE_APP_SERVICE_NAME="<Azure App Service name>"
       export ACR_LOGIN_SERVER="<ACR Login Server URI>"
       export ACR_USERNAME="<Azure Service Principal username>"
       export ACR_PASSWORD="<Azure Service Principal password>"


