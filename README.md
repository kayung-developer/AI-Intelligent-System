# AI Intelligent System
<img src="Assets/ai.png" alt="" width="30" height="30">

This is an AI-powered system for machine learning, image processing, and model deployment. It provides a user-friendly interface to upload datasets, train AI models, process images, and deploy models via FastAPI.

---

## Features

- **Upload Dataset**: Easily upload CSV or Excel files for training.
- **Train AI Models**: Train TensorFlow-based models on your data with a single click.
- **Make Predictions**: Test the trained model on new data.
- **Image Processing**: Perform optimization and apply 3D transformations to images.
- **Model Deployment**: Deploy your trained model via FastAPI for easy integration with other services.
- **User Management**: Leverages SQLite for storing user data.
- **3D Visualizations**: Optional support for 3D plots using Plotly.

---

## Prerequisites

Ensure you have the following software installed before proceeding:
- Python 3.9+
- SQLite (included with Python)
- Git (for version control)

## Setup

1. **Clone the Repository**:
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/kayung-developer/ai-intelligent-system.git
   cd ai-intelligent-system
   ```
2. **Set up a virtual environment (optional but recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On
```
3. **Install required packages**:
```bash
pip install -r requirements.txt
```
## Usage
**To run the AI Intelligent System, execute the following command**:
```bash
python app.py
```
## Here’s how to use the system for [specific task or function]:

```bash
from ai_intelligent_system import AIModel
```
# Initialize the model
```bash
model = AIModel()
```
# Example usage
```bash
result = model.predict(data)
print(result)
```

## Support
**If you encounter any issues or have questions, please open an issue on this repository or contact us at [princelillwitty@gmail.com].**

## Development
**We welcome contributions to the AI Intelligent System project! To contribute**


## Future Advancements:

- Advanced Model Integration: Add ResNet, EfficientDet, and more models.

- Gesture Recognition & Voice Control: Hands-free AI interaction.

- Facial Recognition & Emotion Analysis.

- Data Logging & Reports Export.

- Cross-Device Synchronization & Augmented Reality (AR) integration.
  

## Cross-Platform Accessibility:

- Windows & Mac: Use PyInstaller for standalone apps.

- Linux: Create .deb or .rpm packages.

- iOS/Android: Port using Kivy or BeeWare.

- Cloud: Deploy on AWS, Azure, or GCP for global access.

- Web: Build a web app with Flask/Django & TensorFlow.js

These upgrades will enhance functionality and make the software accessible on any platform.


## Screenshots
**Here are some of a few screenshots of the application**:

| Feature            | Screenshots                                     |
|--------------------|-------------------------------------------------|
| Main Interface      | ![Main Interface](screenshots/3Dview.png)      |
| Image Processing    | ![Image Processing](screenshots/cv.png)          |
| Settings Page       | ![Settings Page](screenshots/settings.png) |

**Check Screenshot folder for all images**
