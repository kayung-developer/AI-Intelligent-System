import csv
import json
import os
import platform
import sqlite3
import time
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, StringVar, Entry, Menu, Image, PhotoImage, BooleanVar, OptionMenu

import tf_keras
from PIL import Image as PILImage, ImageTk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from customtkinter import CTkOptionMenu
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
import matplotlib

from efficientnet_pytorch import EfficientNet

import app
from about.about import AboutFrame
from app import AIApp
from finance.finance import FraudFrame
from medical.medical import MedicalFrame
from vision.imagify import ImagifyFrame
from vision.vision import VisionFrame

matplotlib.use('Agg')  # Use a non-interactive backend for compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

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

try:
    LABELS = np.array(open("imagenet_labels.txt").read().splitlines())
except FileNotFoundError:
    LABELS = np.array([])
    font_color = "#FFFFFF"  # Default font color (white for dark theme)


class HomeFrame(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.label_header = None
        self.profile_image_path = None

        # Variables
        self.model_type = "MobileNetV2"  # Default to MobileNetV2
        self.video_capture = None  # Camera object for OpenCV
        self.pose = mp_pose.Pose()
        self.create_main_frames()
        self.create_bottom_buttons()
        # Variables

        # Initialize dataset and model
        self.data = None
        self.model = None
        self.label_encoder = LabelEncoder()

        self.figure_2d = Figure(figsize=(5, 4), dpi=110)
        self.ax_2d = self.figure_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.figure_2d, master=self.center_frame)
        self.canvas_2d_widget = self.canvas_2d.get_tk_widget()

        self.figure_3d = Figure(figsize=(5, 4), dpi=120)
        self.ax_3d = self.figure_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.figure_3d, master=self.center_frame)
        self.canvas_3d_widget = self.canvas_3d.get_tk_widget()

        # Pack the default (2D) canvas initially
        self.canvas_2d_widget.pack(side="top", fill="both", expand=True)

    def create_main_frames(self):
        self.cv_image = self.load_image("assets\\vision.png")
        self.deploy_image = self.load_image("assets\\deploy.png")
        self.display_image = self.show_image("assets\\chatbot.png")
        self.upload_image = self.load_image("assets\\upload.png")
        self.train_image = self.load_image("assets\\train.png")
        self.predict_image = self.load_image("assets\\predict.png")
        self.export = self.load_image("assets\\download.jpeg")
        self.save_image = self.load_image("assets\\save.png")
        self.load_model_image = self.load_image("assets\\load.png")

        # Main frame divided into top, center, and bottom frames
        self.top_frame = ctk.CTkFrame(self, width=800, height=50)
        self.top_frame.pack(side="top", fill="x")

        self.center_frame = ctk.CTkFrame(self)
        self.center_frame.pack(fill="both", expand=True)

        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(side="bottom", fill="x", expand=True)

        self.slogan = ctk.CTkFrame(self, width=900)
        self.slogan.pack(side="bottom", fill="x", expand=True)


        # Header Label
        self.label_header = ctk.CTkLabel(self.top_frame,
                                         text="AI Intelligent System is an automated Artificial intelligence, Machine Learning and Computer Vision Prototype",
                                         font=("Arial", 12))
        self.label_header.pack(side="bottom", pady=10)

        set_frame = ctk.CTkFrame(self)
        set_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.plot_type = StringVar(value="2D")
        self.plot_type_menu = ctk.CTkOptionMenu(set_frame, width=80, values=["2D", "3D"], variable=self.plot_type, command=self.update_plot_view)
        self.plot_type_menu.grid(row=0, column=0, sticky="e", pady=2, padx=2)

        self.settings = {
            "sentiment_threshold": 0.1,
            "network_enabled": True,
            "developer_mode": False,
            "secure_server": False,
            "enable_3d_analysis": False
        }
        font_color = "#FFFFFF"
        # Load the saved theme for the new window

        # Configure grid layout with two columns
        self.grid_columnconfigure(0, weight=1)  # Left column (labels)
        self.grid_columnconfigure(1, weight=1)  # Right column (entries/switches)


        # Switch to disable/allow networks
        self.network_enabled_var = ctk.IntVar(value=int(self.settings["network_enabled"]))
        self.network_switch = ctk.CTkCheckBox(set_frame, text="",
                                              variable=self.network_enabled_var)
        ctk.CTkLabel(set_frame, text="Network:", text_color=font_color).grid(row=0, column=2, sticky="w",
                                                                             padx=10, pady=5)
        self.network_switch.grid(row=0, column=3, sticky="e", padx=10, pady=2)

        # Developer options
        self.developer_mode_var = ctk.IntVar(value=int(self.settings["developer_mode"]))
        self.developer_switch = ctk.CTkCheckBox(set_frame, text="",
                                                variable=self.developer_mode_var)
        ctk.CTkLabel(set_frame, text="Developer Mode:", text_color=font_color).grid(row=1, column=0,
                                                                                    sticky="w", padx=10,
                                                                                    pady=5)
        self.developer_switch.grid(row=1, column=1, sticky="e", padx=10, pady=2)

        # Secure model server
        self.secure_server_var = ctk.IntVar(value=int(self.settings["secure_server"]))
        self.secure_server_switch = ctk.CTkCheckBox(set_frame, text="",
                                                    variable=self.secure_server_var)
        ctk.CTkLabel(set_frame, text="Secure Model Server:", text_color=font_color).grid(row=1, column=2,
                                                                                         sticky="w", padx=10,
                                                                                         pady=5)
        self.secure_server_switch.grid(row=1, column=3, sticky="e", padx=10, pady=2)


        # Save Settings Button
        save_button = ctk.CTkButton(set_frame, text="Save Settings", fg_color="green", hover_color="red", command=self.save_settings)
        save_button.grid(row=2, column=0, pady=5)

        # Plot Data Button
        plot = ctk.CTkButton(set_frame, text="Plot", fg_color="navyblue", width=70, hover_color="red", command=self.plot_data)
        plot.grid(row=2, column=1, pady=5)

    def save_settings(self):
        #self.settings["sentiment_threshold"] = float(self.sentiment_threshold_var.get())
        self.settings["network_enabled"] = bool(self.network_enabled_var.get())
        self.settings["developer_mode"] = bool(self.developer_mode_var.get())
        self.settings["secure_server"] = bool(self.secure_server_var.get())
        #self.settings["enable_3d_analysis"] = bool(self.enable_3d_analysis_var.get())
        messagebox.showinfo("Settings", "Settings saved successfully!")
        #AIApp().save_theme_button_clicked()
        #AIApp.save_theme(self)
        theme = "/theme_settings.json"
        AIApp.save_theme(theme)

    def create_bottom_buttons(self):
        # Load images for buttons

        # Styled Buttons with Images
        self.btn_upload_data = ctk.CTkButton(self.bottom_frame, text="Upload Dataset", image=self.upload_image,
                                             compound="left", command=self.upload_dataset)
        self.btn_upload_data.pack(side="left", padx=5, pady=5)

        self.btn_train_model = ctk.CTkButton(self.bottom_frame, text="Train Model", image=self.train_image,
                                             compound="left",
                                             command=self.train_model)
        self.btn_train_model.pack(side="left", padx=5, pady=5)

        self.btn_predict = ctk.CTkButton(self.bottom_frame, text="Predict", image=self.predict_image, compound="left",
                                         command=self.predict_data)
        self.btn_predict.pack(side="left", padx=5, pady=5)

        self.btn_report = ctk.CTkButton(self.bottom_frame, text="Export", image=self.export, compound="left",
                                         command=self.export_report)
        self.btn_report.pack(side="left", padx=5, pady=5)


        self.btn_load_model = ctk.CTkButton(self.bottom_frame, text="Load Model", image=self.load_model_image,
                                            compound="left",
                                            command=self.load_model)
        self.btn_load_model.pack(side="left", padx=5, pady=5)

        self.btn_deploy_model = ctk.CTkButton(self.bottom_frame, text="Deploy Model", image=self.deploy_image,
                                              compound="right", command=self.deploy_model)
        self.btn_deploy_model.pack(side="right", padx=5, pady=5)

        self.label_footer = ctk.CTkLabel(self.slogan, text="Developed By Slogan Technologies", font=("Arial", 12))
        self.label_footer.pack(side="bottom", padx=5, pady=5)

    def display_image(self, image_path):
        image = PILImage.open(image_path)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def show_image(self, image_path):
        """Displays the image in the right frame."""
        img = PILImage.open(image_path)  # Open the image using PIL
        return ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))

    def load_image(self, path):
        """Loads an image and converts it to a CTkImage object."""
        img = PILImage.open(path)  # Open the image using PIL
        return ctk.CTkImage(light_image=img, dark_image=img, size=(20, 20))

    def update_plot_view(self, selected_plot):
        """Update the view based on the selected plot type."""
        if selected_plot == "3D":
            # Show 3D canvas and hide 2D canvas
            self.canvas_2d_widget.pack_forget()
            self.canvas_3d_widget.pack(side="top", fill="both", expand=True)
        elif selected_plot == "2D":
            # Show 2D canvas and hide 3D canvas
            self.canvas_3d_widget.pack_forget()
            self.canvas_2d_widget.pack(side="top", fill="both", expand=True)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "File uploaded successfully!")
                #self.plot_data()
            except Exception as e:
                messagebox.showerror("Error", f"Error uploading file: {e}")

    def plot_data(self):
        """Plot the data based on the selected plot type."""
        if self.data is None:
            messagebox.showwarning("Warning", "No data to plot.")
            return

        if self.plot_type.get() == "2D":
            self.plot_2d()
        elif self.plot_type.get() == "3D":
            self.plot_3d()

    def plot_2d(self):
        """Plot the data in 2D."""
        self.ax_2d.clear()
        self.ax_2d.plot(self.data.index, self.data.iloc[:, 0], "bo-", label="2D Data")
        self.ax_2d.set_title("2D View")
        self.ax_2d.set_xlabel("Index")
        self.ax_2d.set_ylabel("Values")
        self.ax_2d.legend()
        self.canvas_2d.draw()

    def plot_3d(self):
        """Create a 3D plot of the data."""
        if self.data is not None:
            self.figure_3d.clear()
            self.ax_3d = self.figure_3d.add_subplot(111, projection='3d')
            x = self.data.index
            y = self.data.iloc[:, 0]
            z = range(len(self.data))
            self.ax_3d.scatter(x, y, z, c='r', marker='o')
            self.ax_3d.set_title("3D View")
            self.ax_3d.set_xlabel("Index")
            self.ax_3d.set_ylabel("Value")
            self.ax_3d.set_zlabel("Z-axis")
            self.canvas_3d.draw()
        else:
            messagebox.showwarning("Warning", "No data available for plotting.")
    def train_model(self):
        """Train a neural network model."""
        if self.data is None:
            messagebox.showerror("Error", "Please upload a dataset first.")
            return

        try:
            features = self.data.iloc[:, :-1].values
            labels = self.label_encoder.fit_transform(self.data.iloc[:, -1].values)
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

            self.model = tf_keras.models.Sequential([
                tf_keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
                tf_keras.layers.Dense(64, activation="relu"),
                tf_keras.layers.Dense(1, activation="sigmoid")
            ])
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            def run_training():
                self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
                test_loss, test_acc = self.model.evaluate(X_test, y_test)
                messagebox.showinfo("Training Complete", f"Test Accuracy: {test_acc:.4f}")

            threading.Thread(target=run_training).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def predict_data(self):
        if self.model is None:
            messagebox.showerror("Error", "No model available. Train or load a model first.")
            return

        try:
            predictions = self.model.predict(self.data.iloc[:, :-1].values)
            predictions = np.round(predictions)
            messagebox.showinfo("Predictions", f"Predicted values: {predictions.flatten()}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train a model first!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".h5")
        if file_path:
            self.model.save(file_path)
            messagebox.showinfo("Save Model", "Model saved successfully!")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if file_path:
            self.model = tf_keras.models.load_model(file_path)
            messagebox.showinfo("Load Model", "Model loaded successfully!")

    def deploy_model(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train a model first!")
            return

        # Simulate deployment process
        messagebox.showinfo("Deploy Model", "Model deployment process started. Please wait...")
        threading.Thread(target=self.simulate_deployment).start()

    def simulate_deployment(self):
        time.sleep(2)  # Simulate time delay
        messagebox.showinfo("Deploy Server", "Model deployed at http://localhost:8000")

    def export_report(self, format="csv"):
        if format == "csv":
            with open('detection_log.csv', 'r') as file:
                csv_data = file.read()
            return csv_data
        elif format == "json":
            with open('detection_log.csv', 'r') as file:
                reader = csv.DictReader(file)
                json_data = json.dumps([row for row in reader])
            return json_data
