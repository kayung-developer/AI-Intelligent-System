import os
import platform
import sqlite3
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, StringVar, Entry, Menu, Image, PhotoImage, BooleanVar

import tf_keras
from PIL import Image as PILImage, ImageTk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from deepface import DeepFace
import matplotlib
#import face_recognition

from efficientnet_pytorch import EfficientNet
from tf_keras.src.applications import imagenet_utils

matplotlib.use('Agg')  # Use a non-interactive backend for compatibility

from yolov5 import YOLOv5  # Assuming YOLOv5 model
import cv2
import threading
import mediapipe as mp
from CTkMenuBar import *
import torch

# Load MobileNetV2 model and labels for object detection
LABELS = np.array(open("imagenet_labels.txt").read().splitlines())

# Disable TensorFlow OneDNN custom ops for cleaner logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load YOLOv5 model from the Ultralytics repository (online method)
YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu', pretrained=True)

# Initialize MediaPipe for body detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Load MobileNetV2 model and labels for object detection
MODEL_MOBILENET = tf_keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True)

# Load ResNet50 for image classification
MODEL_RESNET = tf_keras.applications.ResNet50(weights="imagenet")

efficientdet_model = EfficientNet.from_pretrained('efficientnet-b0')
# efficientdet_model = None
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Faster_RCNN =  torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)


try:
    LABELS = np.array(open("imagenet_labels.txt").read().splitlines())
except FileNotFoundError:
    LABELS = np.array([])
    font_color = "#FFFFFF"  # Default font color (white for dark theme)

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())


def detect_objects_mobilenet(frame):
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = tf_keras.applications.mobilenet_v2.preprocess_input(input_frame)
    input_frame = np.expand_dims(input_frame, axis=0)
    predictions = MODEL_MOBILENET.predict(input_frame)
    return predictions


def detect_objects_yolo(frame):
    results = YOLO_MODEL(frame)
    detections = []
    for detection in results.xyxy[0]:
        class_id = int(detection[5])
        confidence = detection[4]
        label = YOLO_MODEL.names[class_id]
        detections.append((class_id, label, confidence))
    return detections


def detect_objects_efficientdet(frame):
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(input_frame, (512, 512))
    detections = efficientdet_model.predict(np.expand_dims(input_frame, axis=0))
    return detections

def detect_objects_resnet(frame):
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = tf_keras.applications.resnet50.preprocess_input(input_frame)
    input_frame = np.expand_dims(input_frame, axis=0)
    predictions = MODEL_RESNET.predict(input_frame)
    top_predictions = tf_keras.applications.resnet50.decode_predictions(predictions, top=5)[0]
    return top_predictions

def detect(frame):
        bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
        person = 1
        for x, y, w, h in bounding_box_cordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {person}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            person += 1
            return frame
def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    detections = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detections.append("Face Detected")
    return detections
def detect_face(frame):
    """
    Detect faces in a given frame using Haar Cascade.
    Draws rectangles around detected faces and returns detection messages.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    detections = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detections.append((x, y, w, h))
    return detections

def draw_pose_landmarks(frame, results):
    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if the full human view is complete
        visibility_threshold = 0.5
        complete_human_detected = all(
            landmark.visibility > visibility_threshold for landmark in results.pose_landmarks.landmark
        )

        if not complete_human_detected:
            cv2.putText(frame, "Incomplete Human System", (8, frame.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 1)
        else:
            cv2.putText(frame, "Complete Human System", (8, frame.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 1)


def draw_object_labels(frame, detections):
    for i, detection in enumerate(detections):
        label = detection[1]
        confidence = detection[2]
        label_text = f"{label}: {confidence:.2f}"

        # Calculate badge position
        y_offset = 30 + i * 35  # Spacing of 40 pixels between labels

        # Draw badge background
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, 0.7, 1)[0]
        badge_x1 = 10
        badge_y1 = y_offset - 20
        badge_x2 = badge_x1 + text_size[0] + 10
        badge_y2 = badge_y1 + text_size[1] + 10
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)

        # Draw text
        cv2.putText(frame, label_text, (badge_x1 + 5, badge_y1 + text_size[1] + 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

def draw_rounded_rectangle(frame, top_left, bottom_right, radius, color, thickness):
    # Calculate coordinates for the rounded rectangle
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw the straight edges
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)  # Top side
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)  # Left and right sides
    cv2.rectangle(frame, (x1 + radius, y2 - radius), (x2 - radius, y2), color, thickness)  # Bottom side

    # Draw the four rounded corners
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, thickness)  # Top-left corner
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, thickness)  # Top-right corner
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, thickness)  # Bottom-left corner
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, thickness)  # Bottom-right corner


class VisionFrame(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)
        self.model_type = "MobileNetV2"
        self.pose = mp_pose.Pose()
        # Threading
        self.stop_event = threading.Event()
        self.video_capture = None
        self.recording = False
        self.save_path = "."

        # Create the UI
        self.create_uis()

    def create_uis(self):
        # self.center_frame = ctk.CTkFrame(self)
        # self.center_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        self.start = self.load_image("assets\\open-camera.png")  # Make image smaller
        self.stop = self.load_image("assets\\no-camera.png")
        self.save = self.load_image("assets\\save.png")
        self.record = self.load_image("assets\\recorder.jpg")
        self.no_record = self.load_image("assets\\no-recorder.jpg")

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Model Selection Dropdown
        self.model_select_label = ctk.CTkLabel(self.bottom_frame, text="Select Detection Model")
        self.model_select_label.pack(side="left", padx=5)

        self.model_selection = ctk.CTkOptionMenu(self.bottom_frame,
                                                 values=["MobileNetV2", "YOLO", "ResNet50", "EfficientDet(Soon)","FaceDetection(Soon)",
                                                         "Faster-RCNN(Soon)"],
                                                 command=self.set_model)
        self.model_selection.pack(side="left", padx=5)

        # Start/Stop Camera Buttons
        self.btn_start_camera = ctk.CTkButton(self.bottom_frame, text="Start Camera", image=self.start,
                                            compound="left",
                                            command=self.start_camera)
        self.btn_start_camera.pack(side="left", padx=5, pady=5)

        self.btn_stop_camera = ctk.CTkButton(self.bottom_frame, text="Stop Camera", image=self.stop,
                                              compound="left",
                                              command=self.stop_camera)
        self.btn_stop_camera.pack(side="left", padx=5, pady=5)

        self.btn_save_image = ctk.CTkButton(self.bottom_frame, text="Save Image", image=self.save,
                                             compound="left",
                                             command=self.save_image)
        self.btn_save_image.pack(side="left", padx=5, pady=5)


        # Record Keyframes Button
        self.btn_record_video = ctk.CTkButton(self.bottom_frame, text="Record Video", image=self.record,
                                           compound="left",
                                           command=self.toggle_recording)
        self.btn_record_video.pack(side="left", padx=5, pady=5)


        # Create a badge-like canvas for the image and text overlay
        self.badge_frame = ctk.CTkFrame(self, width=800, height=600, corner_radius=2)
        self.badge_frame.pack(expand=True, fill="both", padx=5, pady=5)

        # Create a canvas to draw the badge
        self.canvas = ctk.CTkCanvas(self.badge_frame, width=600, height=800, background="white", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both", padx=5, pady=5)

        # Background frame or badge effect (Optional)
        self.canvas.create_rectangle(10, 10, 1000, 800, fill="black", outline="gray")  # Enlarged badge-like square

        # Label for displaying the camera feed as image inside badge
        self.image_on_canvas = self.canvas.create_image(520, 260, anchor="center")

    def load_image(self, path):
        """Loads an image and converts it to a CTkImage object."""
        img = PILImage.open(path)  # Open the image using PIL
        return ctk.CTkImage(light_image=img, dark_image=img, size=(20, 20))

    def set_model(self, model):
        self.model_type = model
        messagebox.showinfo("Model Selected", f"{self.model_type} selected for object detection.")

    def start_camera(self):
        self.video_capture = cv2.VideoCapture(0)
        # Set camera resolution (increase size as needed)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 900)  # Width
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  # Height

        self.stop_event.clear()
        self.camera_thread = threading.Thread(target=self.process_camera_feed)
        self.camera_thread.start()

    def stop_camera(self):
        self.stop_event.set()
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.video_writer = self.initialize_video_writer()
            self.btn_record_video.configure(text="Stop Recording", image=self.no_record)
        else:
            self.video_writer.release()
            self.btn_record_video.configure(text="Record Video", image=self.record)

    def initialize_video_writer(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
        if not file_path:
            self.recording = False
            return None

        frame_width = int(self.video_capture.get(3))
        frame_height = int(self.video_capture.get(4))
        return cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

    def process_camera_feed(self):
        while not self.stop_event.is_set() and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Preprocess frame for pose detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (1280, 720))  # Resize for consistent processing
            results = self.pose.process(frame_rgb)

            # Draw landmarks on the frame
            draw_pose_landmarks(frame, results)

            # Detect objects using the selected model
            if self.model_type == "MobileNetV2":
                object_detections = detect_objects_mobilenet(frame)
            elif self.model_type == "YOLO":
                object_detections = detect_objects_yolo(frame)
            elif self.model_type == "ResNet50":
                object_detections = detect_objects_resnet(frame)
            elif self.model_type == "EfficientDet":
                #object_detections = detect_objects_efficientdet(frame)
                messagebox.showinfo("Coming Soon", "Stay tunned")
            elif self.model_type == "FaceDetection":
                messagebox.showinfo("Coming Soon", "Stay tunned")
                #object_detections = detect_faces(frame)
            elif self.model_type == "ResNet50":
                messagebox.showinfo("Coming Soon", "Stay tunned")

            elif self.model_type == "Faster-RCNN":
                #torch.no_grad()
                messagebox.showinfo("Coming Soon")


            # Draw labels
            draw_object_labels(frame, object_detections)

            emotions = self.detect_facial_emotion(frame)
            self.display_emotion(frame, emotions)

            # Convert frame to PIL format and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=image)

            # Keep a reference to avoid garbage collection

            # Update image on canvas
            self.canvas.itemconfig(self.image_on_canvas, image=imgtk)
            self.canvas.image = imgtk

            # Save frame to video if recording
            if self.recording and self.video_writer:
                self.video_writer.write(frame)

            self.update()

    def detect_facial_emotion(self, frame):
        """
        Detect the dominant facial emotion in a given frame using DeepFace.

        Parameters:
            frame: The input image/frame to analyze.

        Returns:
            A string representing the dominant emotion or None if detection fails.
        """
        try:
            emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            return emotions[0]['dominant_emotion']
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return None

    def display_emotion(self, frame, emotion):
        """
        Display the detected emotion on the given frame.

        Parameters:
            frame: The input image/frame to annotate.
            emotion: The detected emotion to display.
        """
        if emotion:
            # Dynamically calculate text size and position
            text = f"Emotion: {emotion}"
            font_scale = 1
            font_thickness = 1
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = 8
            text_y = frame.shape[0] - 5

            # Draw background rectangle for better visibility
            cv2.rectangle(frame,
                          (text_x - 2, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 5, text_y + 5),
                          (0, 0, 0),
                          -1)

            # Put emotion text on the frame
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                        font_thickness)

    def save_image(self):
        ret, frame = self.video_capture.read()
        if not ret:
            print("Failed to capture image")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, frame)
            print(f"Image saved to {file_path}")

    def quit_app(self):
        self.stop_camera()
        self.destroy()
