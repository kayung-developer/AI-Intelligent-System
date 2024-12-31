import customtkinter as ctk
from tkinter import messagebox, StringVar, IntVar
import cv2
import numpy as np
from tkinter import filedialog

import tf_keras
from tf_keras.models import load_model
from PIL import Image, ImageTk
from deepface import DeepFace
import requests
from tf_keras.applications import ResNet50
from tf_keras.applications.resnet50 import preprocess_input, decode_predictions

# Face++ API credentials
FACEPP_API_KEY = "C5ffSUUnc_1iujEZ0ZJabArjjJIZBzrl"
FACEPP_API_SECRET = "xcWxTBhU4dRV3lcKyxUZhyNZv5_r98TP"
FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
#model = ResNet50(weights="imagenet")
model = tf_keras.applications.ResNet50(weights="imagenet")

class ImagifyFrame(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)
        # Variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.camera_running = False


        # frame = ImagifyFrame(self)
        # frame.pack(fill="both", expand=True)

        # Main layout configuration
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Button Panel (Image-related buttons)
        self.left_panel = ctk.CTkFrame(self.main_frame, width=200)
        self.left_panel.grid(row=0, column=0, sticky="nsw", padx=10, pady=10)

        self.load_button = ctk.CTkButton(self.left_panel, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)


        self.blur_button = ctk.CTkButton(self.left_panel, text="Blur Image", command=self.blur_image)
        self.blur_button.grid(row=1, column=0, padx=5, pady=5)

        self.deblur_button = ctk.CTkButton(self.left_panel, text="Un-Blur Image", command=self.deblur_image)
        self.deblur_button.grid(row=2, column=0, padx=5, pady=5)

        # Right Button Panel (Other buttons)
        self.right_panel = ctk.CTkFrame(self.main_frame, width=200)
        self.right_panel.grid(row=0, column=2, sticky="nse", padx=10, pady=10)

        self.classify_button = ctk.CTkButton(self.right_panel, text="Classify Image", command=self.classify_image)
        self.classify_button.grid(row=0, column=0, padx=5, pady=5)

        self.recognize_button = ctk.CTkButton(self.right_panel, text="Recognize Face", command=self.recognize_face)
        self.recognize_button.grid(row=1, column=0, padx=5, pady=5)

        self.detect_button = ctk.CTkButton(self.right_panel, text="Detect Faces", command=self.detect_faces)
        self.detect_button.grid(row=2, column=0, padx=5, pady=5)

        self.model_button = ctk.CTkButton(self.right_panel, text="Load Model",
                                          command=lambda: self.load_model(model_name="Facenet"))
        self.model_button.grid(row=3, column=0, padx=5, pady=5)

        self.camera_button = ctk.CTkButton(self.right_panel, text="Start Camera", command=self.start_camera)
        self.camera_button.grid(row=4, column=0, padx=5, pady=5)

        self.stop_camera_button = ctk.CTkButton(self.right_panel, text="Stop Camera", command=self.stop_camera)
        self.stop_camera_button.grid(row=5, column=0, padx=5, pady=5)

        # Image Display (Centered)
        self.image_frame = ctk.CTkFrame(self.main_frame, width=600, height=800)
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.image_label = ctk.CTkLabel(self.image_frame, text="No Image Loaded")
        self.image_label.pack(fill="both", expand=True)

    def load_image(self):
        """Load an image from file"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.processed_image = None  # Reset processed image
            self.display_image(self.original_image)

    def display_image(self, img):
        """Display image in the interface"""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img_pil = Image.fromarray(rgb_image)
        img_pil = img_pil.resize((600, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk

    def classify_image(self):
        """Classify the loaded image"""
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return None

        model = load_model(ResNet50(weights="imagenet"))  # Specify your model path

        image_to_classify = cv2.resize(self.original_image, (224, 224))  # Resize for classification
        image_to_classify = image_to_classify.astype("float32") / 255.0
        image_to_classify = np.expand_dims(image_to_classify, axis=0)

        predictions = model.predict(image_to_classify)
        class_label = np.argmax(predictions, axis=1)  # Get the class label
        messagebox.showinfo("Classification", f"Predicted Class: {class_label}")

    def blur_image(self):
        """Apply a blur to the image"""
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        self.processed_image = cv2.GaussianBlur(self.original_image, (15, 15), 0)
        self.display_image(self.processed_image)

    def deblur_image(self):
        """Un-blur the image"""
        if self.processed_image is None:
            messagebox.showerror("Error", "Please blur an image first!")
            return

        # Un-blurring by applying sharpening filter iteratively
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
        self.display_image(self.processed_image)

    def recognize_face(self):
        """Recognize faces using DeepFace"""
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        try:
            results = DeepFace.analyze(self.original_image, actions=['age', 'gender', 'emotion'],
                                       enforce_detection=False)
            age = results[0]['age']
            gender = results[0]['gender']
            emotion = results[0]['dominant_emotion']
            messagebox.showinfo("DeepFace Results", f"Age: {age}, Gender: {gender}, Emotion: {emotion}")
        except Exception as e:
            messagebox.showerror("Recognition Error", str(e))

    def detect_faces(self):
        """Detect faces using Face++ API"""
        if self.image_path is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        image_data = open(self.image_path, "rb").read()
        params = {
            "api_key": FACEPP_API_KEY,
            "api_secret": FACEPP_API_SECRET,
            "return_attributes": "age,gender,emotion"
        }
        files = {"image_file": image_data}

        try:
            response = requests.post(FACEPP_URL, data=params, files=files)
            if response.status_code == 200:
                faces = response.json()["faces"]
                if len(faces) > 0:
                    attributes = faces[0]["attributes"]
                    age = attributes["age"]["value"]
                    gender = attributes["gender"]["value"]
                    emotions = attributes["emotion"]
                    dominant_emotion = max(emotions, key=emotions.get)
                    messagebox.showinfo("Face++ Results", f"Age: {age}, Gender: {gender}, Emotion: {dominant_emotion}")
                else:
                    messagebox.showinfo("No Faces", "No faces detected in the image.")
            else:
                messagebox.showerror("API Error", "Error calling Face++ API: " + response.text)
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))

    def load_model(self, model_name="VGG-Face"):
        """Load a specific model using DeepFace"""
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        try:
            result = DeepFace.verify(self.original_image, self.original_image, model_name=model_name)
            messagebox.showinfo("Model Loaded",
                                f"Loaded {model_name} model successfully. Verified: {result['verified']}")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))

    def start_camera(self):
        """Start the camera feed"""
        self.camera_running = True
        self.camera_capture = cv2.VideoCapture(0)
        self.update_camera()

    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_running = False
        if hasattr(self, "camera_capture") and self.camera_capture.isOpened():
            self.camera_capture.release()
        self.image_label.configure(image=None, text="No Image Loaded")

    def update_camera(self):
        """Update the camera feed"""
        if self.camera_running and self.camera_capture.isOpened():
            ret, frame = self.camera_capture.read()
            if ret:
                if self.original_image is not None:
                    frame = self.apply_transformation(frame)
                self.display_image(frame)
                self.after(10, self.update_camera)  # Update every 10ms

    def apply_transformation(self, frame):
        """Apply the transformation from loaded image to the live camera feed"""
        if self.original_image is None:
            return frame

        try:
            height, width = frame.shape[:2]
            transformed_image = cv2.resize(self.original_image, (width, height))
            blended_frame = cv2.addWeighted(frame, 0.5, transformed_image, 0.5, 0)
            return blended_frame
        except Exception as e:
            print(f"Transformation Error: {e}")
            return frame
