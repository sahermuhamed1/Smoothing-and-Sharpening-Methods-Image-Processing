import streamlit as st
import cv2
import numpy as np
from PIL import Image
from filters import SmoothingFilters, SharpeningFilters

def main():
    st.title("Image Processing Application")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Filter selection
        filter_type = st.radio("Select Filter Type", ["Smoothing", "Sharpening"])
        
        if filter_type == "Smoothing":
            smoothing = SmoothingFilters()
            filter_name = st.selectbox(
                "Select Smoothing Filter",
                ["Mean", "Median", "Gaussian", "Bilateral", "Box", 
                 "Motion Blur", "Anisotropic", "Non-Local Means"]
            )
            
            # Process image based on selection
            processed = None
            if filter_name == "Mean":
                kernel_size = st.slider("Kernel Size", 3, 15, 3, 2)
                processed = smoothing.mean_filter(img_array, kernel_size)
            elif filter_name == "Median":
                kernel_size = st.slider("Kernel Size", 3, 15, 3, 2)
                processed = smoothing.median_filter(img_array, kernel_size)
            elif filter_name == "Gaussian":
                kernel_size = st.slider("Kernel Size", 3, 15, 3, 2)
                sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
                processed = smoothing.gaussian_filter(img_array, kernel_size, sigma)
            elif filter_name == "Bilateral":
                d = st.slider("Diameter", 5, 15, 9)
                sigma_color = st.slider("Sigma Color", 10, 150, 75)
                sigma_space = st.slider("Sigma Space", 10, 150, 75)
                processed = smoothing.bilateral_filter(img_array, d, sigma_color, sigma_space)
            elif filter_name == "Box":
                kernel_size = st.slider("Kernel Size", 3, 15, 3, 2)
                processed = smoothing.box_filter(img_array, kernel_size)
            elif filter_name == "Motion Blur":
                kernel_size = st.slider("Kernel Size", 3, 31, 15, 2)
                processed = smoothing.motion_blur(img_array, kernel_size)
            elif filter_name == "Anisotropic":
                iterations = st.slider("Iterations", 1, 20, 10)
                processed = smoothing.anisotropic_diffusion(img_array, iterations)
            elif filter_name == "Non-Local Means":
                h = st.slider("Filter Strength", 1, 20, 10)
                processed = smoothing.nlm_filter(img_array, h)
            
        else:  # Sharpening
            sharpening = SharpeningFilters()
            filter_name = st.selectbox(
                "Select Sharpening Filter",
                ["Laplacian", "High Boost", "Sobel", "Prewitt", 
                 "Unsharp Mask", "Roberts Cross", "Scharr", "DoG", 
                 "Emboss", "Kirsch Compass"]
            )
            
            # Process image based on selection
            processed = None
            if filter_name == "Laplacian":
                processed = sharpening.laplacian_filter(img_array)
            elif filter_name == "High Boost":
                alpha = st.slider("Alpha", 1.0, 3.0, 1.5, 0.1)
                processed = sharpening.high_boost_filter(img_array, alpha)
            elif filter_name == "Sobel":
                processed = sharpening.sobel_filter(img_array)
            elif filter_name == "Prewitt":
                processed = sharpening.prewitt_filter(img_array)
            elif filter_name == "Unsharp Mask":
                sigma = st.slider("Sigma", 0.1, 3.0, 1.0, 0.1)
                strength = st.slider("Strength", 0.1, 3.0, 1.5, 0.1)
                processed = sharpening.unsharp_masking(img_array, sigma, strength)
            elif filter_name == "Roberts Cross":
                processed = sharpening.roberts_cross(img_array)
            elif filter_name == "Scharr":
                processed = sharpening.scharr_filter(img_array)
            elif filter_name == "DoG":
                sigma1 = st.slider("Sigma 1", 0.1, 3.0, 1.0, 0.1)
                sigma2 = st.slider("Sigma 2", 0.1, 3.0, 2.0, 0.1)
                processed = sharpening.dog_filter(img_array, sigma1, sigma2)
            elif filter_name == "Emboss":
                processed = sharpening.emboss_filter(img_array)
            elif filter_name == "Kirsch Compass":
                processed = sharpening.kirsch_compass_filter(img_array)
        
        if processed is not None:
            # Convert BGR back to RGB for display
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(processed, caption=f"Processed Image ({filter_name})", use_column_width=True)

if __name__ == "__main__":
    main()
