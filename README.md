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
  '''
  bash
  docker build -t brain_tumor_classifier .

#### Set Up AWS EC2 Instance
- Host the deployed application on an **AWS EC2** instance.
- Pull the Docker image from **AWS ECR** and run the application on EC2.

#### Automate Deployment with GitHub Actions
- Use **GitHub Actions** to automate the deployment workflow.
- On each code push:
  - Retrain the model.
  - Build the Docker image.
  - Push it to **AWS ECR**.
  - Pull the image to **EC2**.
  - Run the application.


### Web Application:
- Build a basic web application using **FLASK** and **HTML** to expose the model's prediction functionality.
- The web app allows users to input customer data and receive predictions on churn status.
- Ensure that the front-end is user-friendly and responsive to enhance user experience.

###  AWS CI/CD Deployment:
1. **Login to AWS Console.**
2. **Create IAM User for Deployment** with specific access:
   - **EC2 access:** It is a virtual machine.
   - **ECR:** Elastic Container Registry to save your Docker image in AWS.

#### Description of the Deployment Steps
- Build Docker image of the source code.
- Push your Docker image to **ECR**.
- Launch your **EC2** instance.
- Pull your image from **ECR** in **EC2**.
- Launch your Docker image in **EC2**.

#### Policy
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

#### Create ECR Repo to Store/Save Docker Image
- Save the URI: `235494811035.dkr.ecr.us-east-1.amazonaws.com/customer_churn`

#### Create EC2 Machine (Ubuntu)
- Open EC2 and Install Docker in the EC2 Machine:

#### Optional
- `sudo apt-get update -y`
- `sudo apt-get upgrade`

#### Required

    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker

#### Configure EC2 as Self-Hosted Runner
Go to **Settings > Actions > Runners > New Self-Hosted Runner**.
Choose your **OS** and run the provided commands one by one.

### Export Environment Variables:

Before running your application, make sure to export the following environment variables in your terminal:

       ```bash
       export MONGODB_URL="mongodb+srv://<username>:<password>...."
       export AWS_ACCESS_KEY_ID="<Your AWS Access Key ID>"
       export AWS_SECRET_ACCESS_KEY="<Your AWS Secret Access Key>"


### Setup GitHub Secrets:

To configure your GitHub repository secrets, add the following key-value pairs:

- **AWS_ACCESS_KEY_ID**: `<Your AWS Access Key ID>`
- **AWS_SECRET_ACCESS_KEY**: `<Your AWS Secret Access Key>`
- **AWS_REGION**: `us-east-1`
- **AWS_ECR_LOGIN_URI**: `235494811035.dkr.ecr.us-east-1.amazonaws.com`
- **ECR_REPOSITORY_NAME**: `customer_churn`

