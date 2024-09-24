import os
import platform
import sqlite3
import subprocess
import threading
import time
import webbrowser

import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel, StringVar, Entry, Menu, Image, PhotoImage

from PIL import Image as PILImage, ImageTk

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from yolov5 import YOLOv5  # Assuming YOLOv5 model
import cv2
import threading
import mediapipe as mp
from CTkMenuBar import *


# Initialize CustomTkinter GUI
#ctk.set_appearance_mode("Dark")
#ctk.set_default_color_theme("black")

# Initialize SQLite database
db_path = "user_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')
conn.commit()

# Initialize MediaPipe for body detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load MobileNetV2 model and labels for object detection
MODEL_MOBILENET = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True)
LABELS = np.array(open("imagenet_labels.txt").read().splitlines())

# Load YOLOv5 model (assuming it is already set up)
YOLO_MODEL = YOLOv5('yolov5s.pt', device='cpu')

class AIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Intelligent System")
        self.geometry("1000x700")
        self.resizable(False, False)

        # Default settings
        self.settings = {
            "sentiment_threshold": 0.1,
            "network_enabled": True,
            "developer_mode": False,
            "secure_server": False,
            "enable_3d_analysis": False
        }

        # Variables
        self.model_type = "MobileNetV2"  # Default to MobileNetV2
        self.video_capture = None  # Camera object for OpenCV
        self.pose = mp_pose.Pose()
        # Variables


        # Initialize dataset and model
        self.dataset = None
        self.model = None


        #theme
        ctk.set_appearance_mode("dark")

        #app icon
        self.app_icon()
        #ai_vision.AIApp.create_ui()
        # Create the main layout
        self.create_menu_bar()
        self.create_main_frames()
        self.create_bottom_buttons()
        self.create_matplotlib_plot()
        self.create_3d_view()
        self.font = ctk.ThemeManager.theme["CTkFont"]["family"]

    def app_icon(self):
        if self.tk.call('tk', 'windowingsystem') == 'x11':  # For Linux and macOS
           # img = Image.open("custom_icon.png")
            img = PILImage.open("ai.png")
            self.tk.call('wm', 'iconphoto', self._w, ImageTk.PhotoImage(img))
        else:  # For Windows
            self.iconbitmap("ai.ico")

    def create_main_frames(self):
        # Main frame divided into top, center, and bottom frames
        self.top_frame = ctk.CTkFrame(self, width=800, height=50)
        self.top_frame.pack(side="top", fill="x")

        self.center_frame = ctk.CTkFrame(self, width=800, height=500)
        self.center_frame.pack(side="top", fill="both", expand=True)

        self.bottom_frame = ctk.CTkFrame(self, width=900, height=100)
        self.bottom_frame.pack(side="bottom", fill="x", expand=True)

        self.slogan = ctk.CTkFrame(self, width=900, height=100)
        self.slogan.pack(side="bottom", fill="x", expand=True)

        self.menu_frame = ctk.CTkFrame(self)
        self.menu_frame.pack(side="top", fill="x")

        # Header Label
        self.label_header = ctk.CTkLabel(self.top_frame, text="AI Intelligent System is an automated Artificial intelligence, Machine Learning and Computer Vision Prototype with aim in using Matplotlib, OpenCV, Modeling Analysis for AI's", font=("Arial", 12))
        self.label_header.pack(side="bottom", pady=10)
        # Footer Label


    def create_bottom_buttons(self):
        # Load images for buttons
        self.cv_image = self.load_image("vision.png")
        self.deploy_image = self.load_image("deploy.png")
        self.display_image = self.show_image("chatbot.png")
        self.upload_image = self.load_image("upload.png")
        self.train_image = self.load_image("train.png")
        self.predict_image = self.load_image("predict.png")
        self.save_image = self.load_image("save.png")
        self.load_model_image = self.load_image("load.png")
        self.network_on_img = self.load_image("connection.png")  # Make image smaller
        self.network_off_img = self.load_image("no-connection.png")

        # Styled Buttons with Images
        self.btn_upload_data = ctk.CTkButton(self.bottom_frame, text="Upload Dataset", image=self.upload_image,
                                             compound="left", command=self.upload_dataset)
        self.btn_upload_data.pack(side="left", padx=5, pady=5)

        self.btn_train_model = ctk.CTkButton(self.bottom_frame, text="Train Model", image=self.train_image, compound="left",
                                             command=self.train_model)
        self.btn_train_model.pack(side="left", padx=5, pady=5)

        self.btn_predict = ctk.CTkButton(self.bottom_frame, text="Predict", image=self.predict_image, compound="left",
                                         command=self.predict_data)
        self.btn_predict.pack(side="left", padx=5, pady=5)

        self.btn_load_model = ctk.CTkButton(self.bottom_frame, text="Load Model", image=self.load_model_image, compound="left",
                                            command=self.load_model)
        self.btn_load_model.pack(side="left", padx=5, pady=5)

        self.btn_deploy_model = ctk.CTkButton(self.bottom_frame, text="Deploy Model", image=self.deploy_image,
                                              compound="left", command=self.deploy_model)
        self.btn_deploy_model.pack(side="left", padx=5, pady=5)

        # Advanced Computer Vision Button
        self.cv_button = ctk.CTkButton(self.bottom_frame, text="Computer Vision", image=self.cv_image,
                                       compound="right", command=self.launch_vision)
        self.cv_button.pack(side="right", padx=5, pady=5)

        self.label_footer = ctk.CTkLabel(self.slogan, text="Developed By Slogan Technologies", font=("Arial", 12))
        self.label_footer.pack(side="bottom", padx=5, pady=5)

        # Create a switch for theme selection
        self.theme_switch = ctk.CTkSwitch(self.menu_frame, text="", command=self.toggle_theme)
        self.theme_switch.pack(side="right", padx=20)


        # Create a button for network "Off" and position it in the bottom right corner
        self.network_image_label = ctk.CTkLabel(
            self.menu_frame, image=self.network_off_img,
            cursor="hand2"  # Change cursor to hand to indicate it's clickable
        ).configure(text="")
        self.network_image_label.pack(side="right", padx=5)

        self.network_image_label.bind("<Button-1>", self.toggle_network)

        # Track the network status (Off by default)
        self.network_on = False
        self.check_network_status()

    def toggle_theme(self):
        """Toggles between light and dark themes based on the switch state."""
        if self.theme_switch.get():
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
    def toggle_network(self, event=None):
        """Toggle the network status between on and off."""
        if self.network_on:
            # Simulate turning the network off (can be customized for actual system control)
            self.disable_network()
            self.network_on = False
            self.network_image_label.configure(image=self.network_off_img)
        else:
            # Simulate turning the network on
            self.enable_network()
            self.network_on = True
            self.network_image_label.configure(image=self.network_on_img)

    def check_network_status(self):
        """Check if the network is currently reachable."""
        network_active = self.is_network_active()
        if network_active:
            self.network_on = True
            self.network_image_label.configure(image=self.network_on_img)
        else:
            self.network_on = False
            self.network_image_label.configure(image=self.network_off_img)

    def is_network_active(self):
        """Check if the network is active by pinging a well-known address."""
        try:
            if platform.system().lower() == "windows":
                output = subprocess.check_output(["ping", "-n", "1", "8.8.8.8"], timeout=3)
            else:
                output = subprocess.check_output(["ping", "-c", "1", "8.8.8.8"], timeout=3)
            return True  # Network is active
        except Exception:
            return False  # Network is inactive

    def disable_network(self):
        """Simulate disabling the network by blocking connectivity (e.g., system commands)."""
        if platform.system().lower() == "windows":
            # For Windows, we could use netsh to disable network interfaces
            os.system("netsh interface set interface 'Wi-Fi' admin=disable")
        else:
            # For Linux/macOS, we can use ifconfig to disable the interface
            os.system("sudo ifconfig eth0 down")
        print("Network disabled.")

    def enable_network(self):
        """Simulate enabling the network by re-enabling connectivity."""
        if platform.system().lower() == "windows":
            # Re-enable network interface in Windows
            os.system("netsh interface set interface 'Wi-Fi' admin=enable")
        else:
            # Re-enable network interface in Linux/macOS
            os.system("sudo ifconfig eth0 up")
            print("Network enabled.")

    def launch_vision(self):
        self.vision = IntelligentSystemApp(self)  # Pass a reference of AIApp to IntelligentSystemApp
        self.vision.mainloop()

    def create_matplotlib_plot(self):

        # Create a Matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.plot.plot([], [])  # Empty plot at the beginning
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.center_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def create_3d_view(self):
        # Create a 3D Matplotlib figure
        self.figure_3d = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_3d = self.figure_3d.add_subplot(111, projection='3d')

        # Example data for 3D plot
        self.ax_3d.plot([0, 1], [0, 1], [0, 1])
        self.canvas_3d = FigureCanvasTkAgg(self.figure_3d, master=self.center_frame)
        self.canvas_3d.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas_3d.get_tk_widget().pack_forget()  # Hide initially


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

    def create_menu_bar(self):

        file_menu = CTkMenuBar(self)
        file_bar = file_menu.add_cascade("File")
        about_bar = file_menu.add_cascade("About")
        account_bar = file_menu.add_cascade("Account")
        server_bar = file_menu.add_cascade("Server")
        settings_bar = file_menu.add_cascade("Settings")
        help_bar = file_menu.add_cascade("Help")


        file_menu = CustomDropdownMenu(widget=file_bar)
        file_menu.add_option(option="Upload Dataset", command=self.upload_dataset)
        file_menu.add_option(option="Create New Project", command=self.create_new_project)
        file_menu.add_option(option="Download Report", command=self.generate_report)
        file_menu.add_option(option="Exit", command=self.quit)
        file_menu.add_option(option="Save Model", command=self.save_model)


        file_menu = CustomDropdownMenu(widget=about_bar)
        file_menu.add_option(option="Other Projects", command=self.our_projects)
        file_menu.add_option(option="About US", command=self.open_about_window)

        file_menu = CustomDropdownMenu(widget=account_bar)
        file_menu.add_option(option="Register", command=self.user_register)
        file_menu.add_option(option="Login", command=self.user_login)

        file_menu = CustomDropdownMenu(widget=server_bar)
        file_menu.add_option(option="Deploy", command=self.deploy_model)

        file_menu = CustomDropdownMenu(widget=settings_bar)
        file_menu.add_option(option="Settings", command=self.open_settings)

        file_menu = CustomDropdownMenu(widget=help_bar)
        file_menu.add_option(option="Help", command=self.open_help_guide)


    def open_settings(self):
        settings_window = Toplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("400x400")
        settings_window.resizable(False, False)
        self.app_icon()

        # Configure grid layout with two columns
        settings_window.grid_columnconfigure(0, weight=1)  # Left column (labels)
        settings_window.grid_columnconfigure(1, weight=1)  # Right column (entries/switches)

        # Settings for sentiment threshold
        self.sentiment_threshold_var = StringVar(value=str(self.settings["sentiment_threshold"]))
        ctk.CTkLabel(settings_window, text="Sentiment Threshold:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.sentiment_threshold_entry = ctk.CTkEntry(settings_window, textvariable=self.sentiment_threshold_var)
        self.sentiment_threshold_entry.grid(row=0, column=1, sticky="e", padx=10, pady=5)

        # Switch to disable/allow networks
        self.network_enabled_var = ctk.IntVar(value=int(self.settings["network_enabled"]))
        self.network_switch = ctk.CTkCheckBox(settings_window, text="",
                                              variable=self.network_enabled_var)
        ctk.CTkLabel(settings_window, text="Network:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.network_switch.grid(row=1, column=1, sticky="e", padx=10, pady=2)

        # Developer options
        self.developer_mode_var = ctk.IntVar(value=int(self.settings["developer_mode"]))
        self.developer_switch = ctk.CTkCheckBox(settings_window, text="",
                                                variable=self.developer_mode_var)
        ctk.CTkLabel(settings_window, text="Developer Mode:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.developer_switch.grid(row=2, column=1, sticky="e", padx=10, pady=2)

        # Secure model server
        self.secure_server_var = ctk.IntVar(value=int(self.settings["secure_server"]))
        self.secure_server_switch = ctk.CTkCheckBox(settings_window, text="",
                                                    variable=self.secure_server_var)
        ctk.CTkLabel(settings_window, text="Secure Model Server:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.secure_server_switch.grid(row=3, column=1, sticky="e", padx=10, pady=2)

        # Allow 3D/2D analysis
        self.enable_3d_analysis_var = ctk.IntVar(value=int(self.settings["enable_3d_analysis"]))
        self.enable_3d_switch = ctk.CTkCheckBox(settings_window, text="",
                                                variable=self.enable_3d_analysis_var, command=self.toggle_3d_view)
        ctk.CTkLabel(settings_window, text="3D Analysis:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.enable_3d_switch.grid(row=4, column=1, sticky="e", padx=10, pady=2)

        # Save Settings Button
        save_button = ctk.CTkButton(settings_window, text="Save Settings", command=self.save_settings)
        save_button.grid(row=5, column=0, columnspan=2, pady=10)

    def toggle_3d_view(self):
        if self.enable_3d_analysis_var.get():
            # Show 3D plot and hide 2D plot
            self.canvas.get_tk_widget().pack_forget()
            self.canvas_3d.get_tk_widget().pack(side="top", fill="both", expand=True)
        else:
            # Show 2D plot and hide 3D plot
            self.canvas_3d.get_tk_widget().pack_forget()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def save_settings(self):
        self.settings["sentiment_threshold"] = float(self.sentiment_threshold_var.get())
        self.settings["network_enabled"] = bool(self.network_enabled_var.get())
        self.settings["developer_mode"] = bool(self.developer_mode_var.get())
        self.settings["secure_server"] = bool(self.secure_server_var.get())
        self.settings["enable_3d_analysis"] = bool(self.enable_3d_analysis_var.get())
        messagebox.showinfo("Settings", "Settings saved successfully!")


    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if file_path:
            try:
                if file_path.endswith(".csv"):
                    self.dataset = pd.read_csv(file_path)
                elif file_path.endswith(".xlsx"):
                    self.dataset = pd.read_excel(file_path)
                messagebox.showinfo("Dataset Upload", "Dataset uploaded successfully!")
                self.plot_data()
            except Exception as e:
                messagebox.showerror("Upload Error", f"Failed to upload dataset: {e}")
        else:
            messagebox.showwarning("Warning", "No file selected.")
    def plot_data(self):
        if self.data is not None:
            self.ax.clear()
            self.ax.plot(self.data.iloc[:, 0], self.data.iloc[:, 1], "bo")
            self.canvas.draw()

    def open_project(self):
        project_file = filedialog.askopenfilename(title="Open Project", filetypes=[("Text Files", "*.txt")])
        if project_file:
            messagebox.showinfo("Project Opened", f"Project '{project_file}' has been opened.")

    def delete_project(self):
        project_file = filedialog.askopenfilename(title="Delete Project", filetypes=[("Text Files", "*.txt")])
        if project_file:
            # Add logic to delete the project file
            messagebox.showinfo("Project Deleted", f"Project '{project_file}' has been deleted.")


    def create_new_project(self):
        project_name = filedialog.asksaveasfilename(title="Project", defaultextension=".txt")
        if project_name:
            messagebox.showinfo("Project Created", f"Project '{project_name}' has been created.")
        # Logic to create a new project
        messagebox.showinfo("Create Project", "New project created successfully!")

    def generate_report(self):
        # Logic to generate a report
        messagebox.showinfo("Report", "Report generated successfully!")

    def our_projects(self):
        webbrowser.open("https://sites.google.com/view/slogantechnologies/projects")

    def our_website(self):
        webbrowser.open("https://sites.google.com/view/slogantechnologies")
    def open_help_guide(self):
        webbrowser.open("http://www.helpguide.com")

    def show_about(self):
        """ Show the About window """
        about_window = ctk.CTkToplevel(self)
        about_window.title("About US")
        about_window.geometry("600x400")

        # About content
        about_content = (
            "Slogan Technologies LLC is a Nigerian-based startup ai and robotics development firm.\n"
            "We specialize in AI, ML, CV, Software Development, Web Development, Cybersecurity, and Technology Education.\n"
            "Our focus is on pushing the boundaries of AI and Robotics, aiming to bring Africa to the forefront of technological advancements.\n\n"
            "Empowering Africa through Technological Innovation\n"
            "Slogan Technologies LLC, incorporated on January 24th, 2024, is a pioneering AI software development company specializing in AI Development, Machine Learning, Computer Vision, Web Development, Applications Development, and Game Development.\n"
            "Our innovative solutions are designed to drive significant advancements across various sectors, fostering job opportunities, technological growth, business transformation, medical improvements, agricultural infrastructure, and educational support for Africans.\n\n"
            "Fostering Job Opportunities\n"
            "Our focus on cutting-edge AI and software development creates a plethora of high-skilled job opportunities, empowering local talent to engage in meaningful and impactful work.\n"
            "By nurturing a pool of skilled professionals, we contribute to reducing unemployment and enhancing the economic stability of communities.\n\n"
            "Driving Technological Growth\n"
            "Through our commitment to AI and machine learning, we are at the forefront of technological advancements.\n"
            "Our projects and research pave the way for new innovations, helping Africa to become a significant player in the global tech landscape.\n"
            "This growth fosters a tech-savvy culture and encourages continuous learning and development.\n\n"
            "Transforming Businesses\n"
            "Our expertise in web and application development provides businesses with advanced tools and platforms to streamline operations, enhance customer experiences, and boost productivity.\n"
            "By leveraging our technologies, businesses can achieve greater efficiency, scalability, and competitiveness in the market.\n\n"
            "Improving Medical Infrastructure\n"
            "Incorporating AI and computer vision into medical applications enables precise diagnostics, personalized treatment plans, and efficient patient care.\n"
            "Our developments in medical technology aim to improve healthcare accessibility and outcomes, leading to healthier communities and a stronger healthcare system.\n\n"
            "Enhancing Agricultural Infrastructure\n"
            "Our AI-driven solutions in agriculture help optimize crop management, enhance yield predictions, and improve resource utilization.\n"
            "By implementing smart agricultural practices, we contribute to food security and sustainable farming, essential for the economic well-being of rural areas.\n\n"
            "Supporting Education\n"
            "We are dedicated to creating educational tools and platforms that leverage AI and interactive technologies.\n"
            "These resources provide accessible and quality education to students across Africa, fostering a culture of innovation and learning.\n"
            "By empowering the next generation with knowledge and skills, we contribute to the continent's socio-economic development.\n\n"
            "At Slogan Technologies LLC, we are committed to leveraging our expertise to drive transformative change across various sectors, ultimately contributing to a brighter and more prosperous future for Africa.\n\n"
            "Copyrights (C) 2024. Slogan Technologies\n"
        )

        # Create and place a label with the about content
        about_label = ctk.CTkLabel(about_window, text=about_content, anchor="w", padx=10, pady=10)
        about_label.pack(fill="both", expand=True)
    def user_register(self):
        register_window = Toplevel(self)
        register_window.title("Register")
        register_window.geometry("400x400")
        register_window.resizable(False, False)

        ctk.CTkLabel(register_window, text="Username:").pack(pady=5)
        self.register_username = ctk.CTkEntry(register_window)
        self.register_username.pack(pady=5)

        ctk.CTkLabel(register_window, text="Password:").pack(pady=5)
        self.register_password = ctk.CTkEntry(register_window, show="*")
        self.register_password.pack(pady=5)

        register_button = ctk.CTkButton(register_window, text="Register", command=self.register_user)
        register_button.pack(pady=10)

    def register_user(self):
        username = self.register_username.get()
        password = self.register_password.get()
        hashed_password = generate_password_hash(password, method='sha256')

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            messagebox.showinfo("Success", "User registered successfully!")
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists!")

    def user_login(self):
        login_window = Toplevel(self)
        login_window.title("Login")
        login_window.geometry("400x400")
        login_window.resizable(False, False)
        ctk.CTkImage()

        ctk.CTkLabel(login_window, text="Username:").pack(pady=5)
        self.login_username = ctk.CTkEntry(login_window)
        self.login_username.pack(pady=5)

        ctk.CTkLabel(login_window, text="Password:").pack(pady=5)
        self.login_password = ctk.CTkEntry(login_window, show="*")
        self.login_password.pack(pady=5)

        login_button = ctk.CTkButton(login_window, text="Login", command=self.login_user)
        login_button.pack(pady=10)

    def login_user(self):
        username = self.login_username.get()
        password = self.login_password.get()

        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        result = cursor.fetchone()

        if result and check_password_hash(result[0], password):
            messagebox.showinfo("Success", "Logged in successfully!")
        else:
            messagebox.showerror("Error", "Invalid username or password!")

    def train_model(self):
        if self.data is None:
            messagebox.showerror("Error", "Please upload a dataset first.")
            return

        try:
            features = self.data.iloc[:, :-1].values
            labels = self.label_encoder.fit_transform(self.data.iloc[:, -1].values)

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            def run_training():
                self.model.fit(X_train, y_train, epochs=10, batch_size=32)
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

    def update_3d_plot(self, X, y):
        # For simplicity, we will perform PCA to reduce dimensions to 3D
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)

        self.ax_3d.clear()
        self.ax_3d.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
        self.ax_3d.set_title("3D Data Visualization")
        self.canvas_3d.draw()

    def predict_datas(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train a model first!")
            return

        # Simulate input data
        input_data = np.random.rand(1, self.dataset.shape[1] - 1)
        prediction = self.model.predict(input_data)
        messagebox.showinfo("Prediction", f"Predicted Value: {prediction[0][0]}")

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
            self.model = tf.keras.models.load_model(file_path)
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

    def open_about_window(self):
        """Open the About window with details about the application."""
        about_window = ctk.CTkToplevel(self)
        about_window.title("About")
        about_window.geometry("600x400")
        about_window.resizable(False, False)

        # Create a scrollable frame for the content
        canvas = ctk.CTkCanvas(about_window, width=600, height=400)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ctk.CTkScrollbar(about_window, command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        # Frame to contain the about sections
        about_frame = ctk.CTkFrame(canvas, width=580, height=800)
        canvas.create_window((0, 0), window=about_frame, anchor="nw")

        # Example sections with images and text
        self.add_about_section(about_frame, "About the App", "ai.png",
                               "AI Intelligent System is designed to provide users with advanced tools for AI/ML tasks.")

        self.add_about_section(about_frame, "Developers", "about.png",
                               "Developed by Slogan Technologies, aiming to revolutionize AI applications globally.")

        self.add_about_section(about_frame, "Technology Stack", "chatbot.png",
                               "Built with Python, TensorFlow, OpenCV, and CustomTkinter for a seamless user experience.")

        self.add_about_section(about_frame, "Version", "ai.png",
                               "Current Version: 1.0.0\nRelease Date: September 2024")

        # Scrollable window size adjustment
        about_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def add_about_section(self, parent_frame, section_title, image_path, description):
        """Helper function to add sections in the About window with image and text."""
        # Section Title
        section_label = ctk.CTkLabel(parent_frame, text=section_title, font=("Arial", 20))
        section_label.pack(pady=10)

        # Section Image
        image = self.load_images(image_path, size=(100, 100))  # Resize image if necessary
        image_label = ctk.CTkLabel(parent_frame, image=image)
        image_label.image = image  # Keep a reference to avoid garbage collection
        image_label.pack(pady=5)

        # Section Description
        description_label = ctk.CTkLabel(parent_frame, text=description, wraplength=500, justify="left")
        description_label.pack(pady=5)
    # Update load_image method to accept size
    def load_images(self, path, size=(20, 20)):
        img = PILImage.open(path)  # Open the image using PIL
        img = img.resize(PILImage.HUFFMAN_ONLY)  # Resize image
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)


    def run(self):
        self.mainloop()


class IntelligentSystemApp(ctk.CTk):
    def __init__(self, root):
        super().__init__()
        self.title("Intelligent Vision")
        self.geometry("900x600")
        self.resizable(False, False)
        self.root = root
        if self.tk.call('tk', 'windowingsystem') == 'x11':  # For Linux and macOS
            # img = Image.open("custom_icon.png")
            img = PILImage.open("ai.png")
            self.tk.call('wm', 'iconphoto', self._w, ImageTk.PhotoImage(img))
        else:  # For Windows
            self.iconbitmap("ai.ico")
        # Variables
        self.model_type = "MobileNetV2"  # Default to MobileNetV2
        self.video_capture = None  # Camera object for OpenCV
        self.pose = mp_pose.Pose()


        self.create_ui()

    def create_ui(self):
        # Frame for camera and control buttons
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Add Model Selection Dropdown
        self.model_select_label = ctk.CTkLabel(self.bottom_frame, text="Select Detection Model:")
        self.model_select_label.pack(side="left", padx=5)

        self.model_selection = ctk.CTkOptionMenu(self.bottom_frame, values=["MobileNetV2", "YOLO"],
                                                 command=self.set_model)
        self.model_selection.pack(side="left", padx=5)

        # Add Camera Functionality Buttons
        self.btn_start_camera = ctk.CTkButton(self.bottom_frame, text="Start Camera", command=self.start_camera)
        self.btn_start_camera.pack(side="left", padx=5)

        self.btn_stop_camera = ctk.CTkButton(self.bottom_frame, text="Stop Camera", command=self.stop_camera)
        self.btn_stop_camera.pack(side="left", padx=5)

        # Optional: Add a Quit Button at the bottom right
        self.btn_quit = ctk.CTkButton(self.bottom_frame, text="Back Home", command=self.quit_app)
        self.btn_quit.pack(side="right", padx=5)

        # Label for displaying the camera feed
        self.camera_feed_label = ctk.CTkLabel(self, text="")
        self.camera_feed_label.pack(expand=True, fill="both", padx=10, pady=10)

    def set_model(self, model):
        """Sets the object detection model based on user selection."""
        self.model_type = model
        messagebox.showinfo("Model Selected", f"{self.model_type} selected for object detection.")

    def start_camera(self):
        self.video_capture = cv2.VideoCapture(0)
        self.camera_thread = threading.Thread(target=self.process_camera_feed)
        self.camera_thread.start()

    def stop_camera(self):
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()

    def process_camera_feed(self):
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Detect body parts
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            self.draw_pose_landmarks(frame, results)

            # Detect objects and technologies based on selected model
            if self.model_type == "MobileNetV2":
                object_detections = self.detect_objects_mobilenet(frame)
            else:  # YOLO
                object_detections = self.detect_objects_yolo(frame)

            self.draw_object_labels(frame, object_detections)

            # Convert frame to ImageTk format and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PILImage.fromarray(frame_rgb)  # Change CTkImage to PILImage
            image = ImageTk.PhotoImage(image)  # Convert to ImageTk format
            self.camera_feed_label.configure(image=image)  # Use configure instead of config
            self.camera_feed_label.image = image  # Keep a reference to avoid garbage collection

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_camera()

    def draw_pose_landmarks(self, frame, results):
        if results.pose_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check if the full human view is complete
            visibility_threshold = 0.5
            complete_human_detected = all(
                landmark.visibility > visibility_threshold for landmark in results.pose_landmarks.landmark
            )

            if not complete_human_detected:
                cv2.putText(frame, "Incomplete Human View", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Complete Human View", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def detect_objects_mobilenet(self, frame):
        """Detects objects using the MobileNetV2 model."""
        # Preprocess frame for MobileNet model
        input_frame = cv2.resize(frame, (224, 224))
        input_frame = tf.keras.applications.mobilenet_v2.preprocess_input(input_frame)
        input_frame = np.expand_dims(input_frame, axis=0)

        # Run the object detection model
        predictions = MODEL_MOBILENET.predict(input_frame)
        top_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

        return top_predictions

    def detect_objects_yolo(self, frame):
        """Detects objects using the YOLO model."""
        results = YOLO_MODEL.predict(frame)
        detections = [(res['class'], res['label'], res['confidence']) for res in results]
        return detections

    def draw_object_labels(self, frame, object_detections):
        """Displays object detection labels on the frame."""
        height, width, _ = frame.shape

        # Display object detection results on the frame
        for i, (class_id, label, score) in enumerate(object_detections):
            if score > 0.5:  # Only display confident predictions
                label_text = f"{label}: {score:.2f}"
                y_position = 100 + i * 30
                cv2.putText(frame, label_text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def quit_app(self):
        if (self.stop_camera is True):
            self.destroy()
            self.root.deiconify()
        else:
            self.destroy()

        #self.stop_camera()
       # self.destroy()
        #self.root.deiconify()



if __name__ == "__main__":
    app = AIApp()
    #app.run()
    app.mainloop()
