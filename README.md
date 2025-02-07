 # Brain Tumor Detection using MRI Images

This project is a deep learning-based application for detecting brain tumors in MRI images. It uses a Convolutional Neural Network (CNN) trained on a dataset of brain MRI images to classify whether a tumor is present or not. The app is built using **Streamlit** for a user-friendly interface.

The Dataset is Downloaded from  https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection/data



**About the data:**  
-----

The dataset contains 2 folders: yes and no which contains 6000 Brain MRI Images.

## **Features**
- **Upload MRI Images**: Users can upload MRI images for tumor detection.
- **Real-Time Prediction**: The app provides real-time predictions with a confidence score.
- **Visualization**: Includes heatmaps and Grad-CAM to highlight regions of interest.
- **Interactive Features**: Adjustable confidence threshold and batch processing for multiple images.
- **Performance Metrics**: Displays accuracy, loss, confusion matrix, and ROC curve.

## **How to Run the App**

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Hemasree-10/Brain_Tumor_Detection.git
   cd brain-tumor-detection

2. Install the required packages:

       pip install -r requirements.txt

3. Download the pre-trained model (Brain_tumor.h5) and place it in the project directory.

### **Running the App**   
  Start the Streamlit app:
                       
    streamlit run app.py

## **How to Use**

- Upload an MRI image using the file uploader.

- The app will display the uploaded image and provide a prediction (Tumor Detected/No Tumor Detected).
  

## **The Brain tumor detection app build using Streamlit app looks like this:**     


![scanpredict](https://github.com/user-attachments/assets/24f87918-1f9e-4c64-adba-7aa3c01b0e7e)

## **Result**  
- It correctly classified 98.58% of the images, which is a very high accuracy for a classification task.

- The loss value is very low, indicating that the model’s predictions are highly accurate and confident.

