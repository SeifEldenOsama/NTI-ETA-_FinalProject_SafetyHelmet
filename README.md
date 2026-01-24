# Safety Helmet Detection using YOLO

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXhzeHVpa3QwdzdqYW43aGk4MjlxYmFuaHhja2R3M2YwZmU1eTlkdyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/msgHgCWZflkp9sbrCe/giphy.gif" width="600" height="750">



This project implements a **Safety Helmet Detection** system designed to monitor construction sites and industrial environments for compliance with safety regulations. By leveraging a lightweight object detection model, the system can identify workers wearing safety helmets, those with bare heads, and general personnel in real-time.

The core of the system is built around the **YOLO (You Only Look Once)** architecture, specifically trained and optimized for deployment on edge devices using the **TensorFlow Lite (TFLite)** format.

## Key Features

*   **Object Detection**: Accurately detects three distinct classes: `helmet`, `head` (bare head), and `person`.
*   **Optimized Model**: Utilizes a TFLite model (`best_float32.tflite`) for efficient, low-latency inference, suitable for deployment on embedded systems or mobile devices.
*   **Interactive Interface**: A user-friendly web interface is provided via **Gradio**, allowing for easy image upload and visualization of detection results.
*   **Clear Visualization**: Bounding boxes are drawn on the output image with distinct colors for immediate visual feedback on safety compliance.

## Project Structure

The repository is organized into a professional modular structure:

| Directory | Description |
| :--- | :--- |
| `src/` | **Source code directory** containing modular Python packages. |
| `src/models/` | Model wrapper for YOLO/TFLite inference. |
| `src/data/` | Data processing and annotation parsing utilities. |
| `src/app/` | Gradio web application implementation. |
| `src/config.py` | Centralized configuration for the project. |
| `model/` | Contains the trained object detection model file, `best_float32.tflite`. |
| `notebooks/` | Original Jupyter Notebooks for research and experimentation. |
| `presentation/` | Project's final presentation in PDF format. |

## Technical Implementation

The model was trained on the **Hard Hat Detection dataset** from Kaggle [1], which provides a diverse set of images for robust training.

### Detection Classes and Visualization

The model is configured to detect the following three classes. The application notebook uses specific color coding for the bounding boxes to highlight compliance status:

| Class | Description | Bounding Box Color (RGB) | Compliance Status |
| :--- | :--- | :--- | :--- |
| **helmet** | A person wearing a safety helmet. | Green (0, 255, 0) | **Compliant** |
| **head** | A person's bare head (no helmet). | Red (0, 0, 255) | **Non-Compliant** |
| **person** | General detection of a person. | Blue (255, 0, 0) | N/A |

### Model and Frameworks

The application is run using the `safety_helmet_application.ipynb` notebook, which incorporates the following key technologies:

*   **Model Loading**: The TFLite model is loaded using the **Ultralytics YOLO** library.
*   **Inference Logic**: The `predict_image` function handles the model inference, applying a confidence threshold of 0.25, and manually drawing the colored bounding boxes using the **OpenCV** library.
*   **Web Interface**: The application is wrapped in a **Gradio** interface for easy sharing and demonstration.

## Setup and Usage

To set up and run the application, you will need a Python environment with the necessary dependencies.

### Prerequisites

*   Python 3.x
*   Jupyter Notebook or JupyterLab

### Installation

Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Application

You can now run the application directly as a Python module:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SeifEldenOsama/SafetyHelmet.git
    cd SafetyHelmet
    ```
2.  **Launch the Gradio App:**
    ```bash
    python3 -m src.app.gradio_app
    ```

Alternatively, you can still use the original notebooks in the `notebooks/` directory for step-by-step execution.

## References

[1] Hard Hat Detection Dataset. *Kaggle*. [https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
