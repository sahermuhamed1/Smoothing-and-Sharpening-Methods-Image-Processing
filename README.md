# Image Processing Application

This application provides a web interface for applying various image processing filters to uploaded images.

## Features

- Support for multiple image formats (JPG, PNG, JPEG)
- Smoothing filters:
  - Mean Filter
  - Median Filter
  - Gaussian Filter
  - Bilateral Filter
  - Box Filter
  - Motion Blur
  - Anisotropic Diffusion
  - Non-Local Means Filter
- Sharpening filters:
  - Laplacian Filter
  - High Boost Filter
  - Sobel Filter
  - Prewitt Filter
  - Unsharp Masking
  - Roberts Cross
  - Scharr Filter
  - DoG Filter
  - Emboss Filter
  - Kirsch Compass Filter

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Navigate to the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```
3. Open your web browser and go to `http://localhost:8501`

## Project Structure

```
Project/
├── src/
│   ├── app.py          # Main Streamlit application
│   └── filters.py      # Filter implementations
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```
