import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('Brain_tumor.h5')

# Streamlit app title
st.title("Brain Tumor Detection using MRI Images")

# Sidebar for additional information
st.sidebar.title("About Brain Tumors and MRI Scans")

# Section 1: What is an MRI Scan?
st.sidebar.header("What is an MRI Scan?")
st.sidebar.write("""
An MRI (Magnetic Resonance Imaging) scan is a medical imaging technique used to visualize detailed internal structures of the body. 
It uses strong magnetic fields and radio waves to generate images of organs, tissues, and other structures. 
MRI scans are commonly used to detect brain tumors, injuries, and other abnormalities.
""")

# Section 2: What is a Brain Tumor?
st.sidebar.header("What is a Brain Tumor?")
st.sidebar.write("""
A brain tumor is an abnormal growth of cells in the brain. Tumors can be **benign** (non-cancerous) or **malignant** (cancerous). 
They can cause symptoms like headaches, seizures, and cognitive difficulties. Early detection is crucial for effective treatment.
""")

# Section 3: Therapies for Brain Tumors
st.sidebar.header("Therapies for Brain Tumors")
st.sidebar.write("""
- **Surgery**: Removal of the tumor.
- **Radiation Therapy**: Using high-energy rays to kill tumor cells.
- **Chemotherapy**: Using drugs to destroy cancer cells.
- **Targeted Therapy**: Targeting specific molecules involved in tumor growth.
- **Immunotherapy**: Boosting the immune system to fight the tumor.
""")

# Section 4: Symptoms of Brain Tumors
st.sidebar.header("Symptoms of Brain Tumors")
st.sidebar.write("""
- Persistent headaches
- Seizures
- Nausea or vomiting
- Vision or hearing problems
- Memory loss or confusion
- Difficulty walking or speaking
""")

# Main App: Brain Tumor Detection
st.header("Upload an MRI Image for Detection")

# File uploader for MRI images
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)

    # Preprocess the image for the model
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize

    # Expand dimensions to match model input shape
    input_img = np.expand_dims(image, axis=0)

    # Make a prediction
    result = model.predict(input_img)
    tumor_probability = result[0][0]

    # Display the result
    st.write("### Prediction Result")
    if tumor_probability > 0.5:
        st.error(f"**Tumor Detected** (Probability: {tumor_probability:.2f})")
        st.write("### Don't Worry, You Will Be Cured!")
        st.write("""
        - Early detection is the first step toward effective treatment.
        - Follow your doctor's advice and stay positive.
        - Modern therapies like surgery, radiation, and chemotherapy have high success rates.
        - You are stronger than you think, and you will overcome this!
        """)
    else:
        st.success(f"**No Tumor Detected** (Probability: {tumor_probability:.2f})")
        st.write("### Congratulations! No Tumor Detected")
        st.write("Here are 5 health tips to keep your brain healthy:")
        st.write("""
        1. **Eat a Balanced Diet**: Include fruits, vegetables, and omega-3 fatty acids.
        2. **Exercise Regularly**: Physical activity improves brain health.
        3. **Get Enough Sleep**: Aim for 7-8 hours of sleep per night.
        4. **Stay Mentally Active**: Solve puzzles, read, or learn new skills.
        5. **Manage Stress**: Practice mindfulness or meditation.
        """)

# Add a button to load a sample image
if st.button("Use Sample Image"):
    sample_image_path = "C:/GIT folder/Brain_Tumor_Detection/dataset/yes/y25.jpg"  # Path to a sample image
    try:
        image = Image.open(sample_image_path)
        st.image(image, caption='Sample MRI Image', use_container_width=True)

        # Preprocess the image for the model
        image = image.resize((64, 64))  # Resize to match model input size
        image = np.array(image)  # Convert to numpy array
        image = image / 255.0  # Normalize

        # Expand dimensions to match model input shape
        input_img = np.expand_dims(image, axis=0)

        # Make a prediction
        result = model.predict(input_img)
        tumor_probability = result[0][0]

        # Display the result
        st.write("### Prediction Result")
        if tumor_probability > 0.5:
            st.error(f"**Tumor Detected** (Probability: {tumor_probability:.2f})")
            st.write("### Don't Worry, You Will Be Cured!")
            st.write("""
            - Early detection is the first step toward effective treatment.
            - Follow your doctor's advice and stay positive.
            - Modern therapies like surgery, radiation, and chemotherapy have high success rates.
            - You are stronger than you think, and you will overcome this!
            """)
        else:
            st.success(f"**No Tumor Detected** (Probability: {tumor_probability:.2f})")
            st.write("### Congratulations! No Tumor Detected")
            st.write("Here are 5 health tips to keep your brain healthy:")
            st.write("""
            1. **Eat a Balanced Diet**: Include fruits, vegetables, and omega-3 fatty acids.
            2. **Exercise Regularly**: Physical activity improves brain health.
            3. **Get Enough Sleep**: Aim for 7-8 hours of sleep per night.
            4. **Stay Mentally Active**: Solve puzzles, read, or learn new skills.
            5. **Manage Stress**: Practice mindfulness or meditation.
            """)
    except Exception as e:
        st.error(f"Error loading sample image: {e}")   
        
    