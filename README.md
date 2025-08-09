# 🩺 Deep Learning Meets Dermatology – Automated Skin Disease Detection

The **Deep Learning Meets Dermatology - Skin disease detection project** is a **CNN-based medical image classification system** that detects and classifies skin diseases from dermatological images.
Using a modified **EfficientNetB4** architecture with transfer learning, data augmentation, and a Streamlit-powered web interface, the system achieves over **93% accuracy** in classifying **three common conditions**: *Acne, Atopic Dermatitis, and Basal Cell Carcinoma (BCC)*.

## 🚀 Features

- Dataset Preprocessing – Normalization, resizing, and augmentation of dermatological images
- CNN-based Model – Modified EfficientNetB4 for robust feature extraction
- Transfer Learning – Improves performance with limited medical datasets
- Data Augmentation – Rotation, scaling, flipping, and brightness adjustments for better generalization
- Attention Mechanism – Focuses on lesion-relevant features while reducing background noise
- Performance Metrics – Accuracy, Precision, Recall, and F1-score evaluation
- Streamlit Web Interface – Upload skin lesion images and get predictions with confidence scores
- Confidence Threshold Alerts – Warns users when prediction certainty is low, encouraging professional consultation

##  📂 Project Structure

- ├── model_training.py               # CNN model training with EfficientNetB4
- ├── data_preprocessing.py           # Dataset cleaning, augmentation, and preparation
- ├── app.py                           # Streamlit web app for real-time predictions
= ├── trained_model.h5                 # Saved CNN model
- ├── requirements.txt                 # Python dependencies
- ├── Deep Learning Meets Dermatology (Final).pdf # Detailed project report
- ├── README.md                        # Project documentation

## 🛠 Tech Stack

- Python
- TensorFlow / Keras – Deep learning model implementation
- OpenCV – Image preprocessing
- Streamlit – Web-based prediction interface
- NumPy / Pandas – Data handling
- Matplotlib / Seaborn – Data visualization

## 🔍 How It Works

1. Dataset Preparation
Source: Kaggle dermatology dataset with train, test, and validation splits
Classes: Acne, Atopic Dermatitis, Basal Cell Carcinoma (BCC)
Preprocessing: Image resizing (150×150), normalization, and quality assessment
Augmentation: Rotation, scaling, horizontal flips, and brightness changes

2. Model Training
Base Model: EfficientNetB4 (transfer learning)
Layers: Convolutional, pooling, attention, and fully connected layers
Optimizer: Adam (adaptive learning rate)
Loss Function: Categorical cross-entropy
Output: Softmax probabilities for each class
Evaluation Metrics: Accuracy, Precision, Recall, F1-score

3. Web Application (Streamlit)
Upload skin lesion image (JPG, JPEG, PNG)
Model outputs:
Predicted class
Confidence score per class
Low-confidence alerts
Results displayed in a user-friendly dashboard with visualizations

## 🖥 How to Run

### 1️⃣ Clone the Repository
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection

### 2️⃣ Install Requirements
pip install -r requirements.txt

### 3️⃣ Train the Model (Optional – model already provided)
python model_training.py

### 4️⃣ Run the Web App
streamlit run app.py

## 💡 Use Cases

- Healthcare Professionals – Quick pre-screening tool for dermatological conditions
- Remote & Rural Areas – Accessible diagnostics where dermatologists are unavailable
- Medical Education – Training aid for recognizing skin lesions
- Telemedicine Platforms – Integrate into existing systems for remote consultations

## 👩‍💻 Author
**Devadarshini P - UG Scholar, Department of Data Science, Kumaraguru College of Liberal Arts and Science**  
**Dr. M C S Geetha – Assistant Professor, Kumaraguru College of Technology**

[🔗 LinkedIn](https://www.linkedin.com/in/devadarshini-p-707b15202/)  
[💻 GitHub](https://github.com/Devadarshini9000)

"Early detection saves lives – bringing AI to dermatology for better healthcare access." 🩻

<img width="469" height="801" alt="image" src="https://github.com/user-attachments/assets/4e1d1c92-bb79-4bcf-a006-3e436bf02065" />
<img width="438" height="801" alt="image" src="https://github.com/user-attachments/assets/86d1e0f8-7396-4095-9272-1cbfe7f59b6a" />



