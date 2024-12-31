import os
import platform
import sqlite3
import subprocess
import threading
import time
import webbrowser
import json
from logging import root

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
from reportlab.lib.colors import white, black
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib

from efficientnet_pytorch import EfficientNet

from about.about import AboutFrame
from finance.finance import FraudFrame
from medical.medical import MedicalFrame
from medical.medicaltest import AdvancedMedical
from finance.financetest import AdvancedFraud
from vision.imagify import ImagifyFrame
from vision.vision import VisionFrame
from face.facial import FacialFrame
from assets import *

matplotlib.use('Agg')  # Use a non-interactive backend for compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from yolov5 import YOLOv5  # Assuming YOLOv5 model
import cv2
import threading
import mediapipe as mp
from CTkMenuBar import *
import torch


# Initialize CustomTkinter GUI
# ctk.set_appearance_mode("Dark")
# ctk.set_default_color_theme("black")

# Initialize SQLite database
db_path = "user_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Update the table creation query to include the new columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL, 
        email TEXT,
        phone TEXT,
        dob TEXT,
        address TEXT,
        profile_image TEXT  -- New column for storing the profile image path
    )
''')
conn.commit()

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
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Faster_RCNN = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)


try:
    LABELS = np.array(open("imagenet_labels.txt").read().splitlines())
except FileNotFoundError:
    LABELS = np.array([])
    font_color = "#FFFFFF"  # Default font color (white for dark theme)


class AIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Intelligent System")
        self.geometry("1080x900")
        self.resizable(False, False)


        self.app_icon()
        self.check_login_state()
        self.theme_color = "dark"  # Default theme
        self.load_theme("theme_settings.json")  # Load the theme if saved

        # Create the TabBar
        self.tab_bar = ctk.CTkTabview(self, width=850, height=550, corner_radius=10, fg_color="white")
        self.tab_bar.pack(fill="both", expand=True, padx=20, pady=20)



        # Set custom styling for the tabs
        self.tab_bar.tab_button_width = 200  # Wider tab buttons # Tab text and hover color
        self.tab_bar.tab_button_height = 40
        self.tab_bar.tab_button_spacing = 10  # Add spacing between the tabs
        self.tab_bar.tab_button_color = "#f0f0f0"  # Inactive tab color
        self.tab_bar.tab_button_hover_color = "#dcdcdc"  # Hover tab color
        self.tab_bar.tab_button_active_color = "#3498db"  # Active tab color
        self.tab_bar.text_font = ("Arial", 14)  # Font for the tab labels

        self.tab_bar.add("Home")
        self.tab_bar.add("Open Vision")
        self.tab_bar.add("Imagify")
        self.tab_bar.add("Facial")
        self.tab_bar.add("About")
        # self.tab_bar.add("AR")
        self.network_on_img = self.load_image("assets\\connection.png")  # Make image smaller
        self.network_off_img = self.load_image("assets\\no-connection.png")

        # Main Screen

        self.main_tab = HomeFrame(self.tab_bar.tab("Home"))
        self.main_tab.pack(fill="both", expand=True)

        # Vision Screen
        self.vision_tab = VisionFrame(self.tab_bar.tab("Open Vision"))
        self.vision_tab.pack(fill="both", expand=True)



        # Imagiify Screen
        self.imagify_tab = ImagifyFrame(self.tab_bar.tab("Imagify"))
        self.imagify_tab.pack(fill="both", expand=True)

        # Facial Screen
        self.facial_tab = FacialFrame(self.tab_bar.tab("Facial"))
        self.facial_tab.pack(fill="both", expand=True)

        # About Screen
        self.about_tab = AboutFrame(self.tab_bar.tab("About"))
        self.about_tab.pack(fill="both", expand=True)

        self.tab_bar.set("Home")

        # theme
        ctk.set_appearance_mode("dark")
        # Create the main layout
        self.create_menu_bar()
        self.font = ctk.ThemeManager.theme["CTkFont"]["family"]
        self.menu_frame = ctk.CTkFrame(self)

        self.menu_frame.pack(side="top", fill="x")
        # Footer Label

        # Create a switch for theme selection
        self.theme_switch = ctk.CTkSwitch(self.menu_frame, text="", command=self.toggle_theme)
        self.theme_switch.pack(side="right", padx=20)

        # Set switch state based on current theme
        if self.theme_color == "dark":
            self.theme_switch.select()  # Dark mode is selected by default
        else:
            self.theme_switch.deselect()  # Light mode is deselected

        # Create a button for network "Off" and position it in the bottom right corner
        # Create a button for network "Off" and position it in the top right corner of the menu bar
        self.network_image_label = ctk.CTkLabel(
            self.menu_frame,
            image=self.network_off_img,
            cursor="hand2"  # Change cursor to hand to indicate it's clickable
        )
        self.network_image_label.configure(text="")

        # Bind the click event to the toggle_network method
        self.network_image_label.bind("<Button-1>", self.toggle_network)

        # Position the label in the top right corner of the menu frame
        self.network_image_label.place(relx=1.0, rely=0.0, anchor='ne')  # 'ne' = north-east (top-right)

        # Track the network status (Off by default)
        self.network_on = False
        self.check_network_status()

    def app_icon(self):
        if self.tk.call('tk', 'windowingsystem') == 'x11':  # For Linux and macOS
            # img = Image.open("assets\\custom_icon.png")
            img = PILImage.open("assets\\ai.png")
            self.tk.call('wm', 'iconphoto', self._w, ImageTk.PhotoImage(img))
        else:  # For Windows
            self.iconbitmap("assets\\ai.ico")

    def toggle_theme(self):
        """Switch between light and dark themes."""
        if self.theme_switch.get() == 1:  # Dark mode is on
            self.theme_color = "dark"
            self.font_color = "white"  # White font for dark mode
            ctk.set_appearance_mode("dark")
        else:  # Light mode is on
            self.theme_color = "light"
            self.font_color = "black"  # Black font for light mode
            ctk.set_appearance_mode("light")

        # Apply the changes to the current window
        self.apply_theme()

    def save_theme(self, filename):
        """Save the current theme settings to a JSON file."""
        theme_settings = {
            "color": self.theme_color,
            "font_color": self.font_color
        }
        with open(filename, 'w') as f:
            json.dump(theme_settings, f)

    def load_theme(self, filename):
        """Load the theme settings from a JSON file."""
        try:
            with open(filename, 'r') as f:
                theme_settings = json.load(f)
            self.theme_color = theme_settings.get("color", "dark")
            self.font_color = theme_settings.get("font_color", "#FFFFFF")
            self.apply_theme()
        except FileNotFoundError:
            print("Theme settings not found, using default theme.")

    def apply_theme(self):
        """Apply the theme settings to the current window."""

    # self.some_label.configure(text_color=self.font_color)

    def save_theme_button_clicked(self):
        """Button callback to save the theme."""
        self.save_theme("theme_settings.json")

    def load_theme_button_clicked(self):
        """Button callback to load the theme."""
        self.load_theme("theme_settings.json")

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

    def create_menu_bar(self):

        file_menu = CTkMenuBar(self)
        file_bar = file_menu.add_cascade("File")
        account_bar = file_menu.add_cascade("Profile")
        others_bar = file_menu.add_cascade("Website")
        server_bar = file_menu.add_cascade("Server")

        file_menu = CustomDropdownMenu(widget=file_bar)
        file_menu.add_option(option="Upload Dataset", command=self.upload_data)
        file_menu.add_option(option="Create New Project", command=self.create_new_project)
        file_menu.add_option(option="Download Report", command=self.generate_report)
        file_menu.add_option(option="Exit", command=self.destroy)
        file_menu.add_option(option="Save Model", command=self.save)

        file_menu = CustomDropdownMenu(widget=others_bar)
        file_menu.add_option(option="Website", command=self.our_website)

        file_menu = CustomDropdownMenu(widget=account_bar)
        file_menu.add_option(option="Register", command=self.user_register)
        file_menu.add_option(option="Login", command=self.user_login)

        file_menu = CustomDropdownMenu(widget=server_bar)
        file_menu.add_option(option="Deploy", command=self.deploy)

    def upload_data(self):
        HomeFrame(master=self).upload_dataset()
    def deploy(self):
        HomeFrame(master=self).deploy_model()

    def save(self):
        HomeFrame(master=self).save_model()

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
    def user_register(self):
        """Open the User Registration window with extended features."""
        register_window = ctk.CTkToplevel(self)
        register_window.title("Register")
        register_window.geometry("500x500")
        register_window.resizable(False, False)

        # Set the window icon
        register_window.iconbitmap('ai.ico')  # Replace with your icon path

        # Load the saved theme for the new window
        self.load_theme("theme_settings.json")

        # Apply the loaded theme
        self.apply_theme()

        # Center image (above username/password fields)
        app_logo = self.load_images("assets\\account.png", size=(100, 100))
        logo_label = ctk.CTkLabel(register_window, image=app_logo, text="")  # Image with no text
        logo_label.image = app_logo
        logo_label.pack(pady=20)

        form_frame = ctk.CTkFrame(register_window)
        form_frame.pack(pady=20)

        # Username
        ctk.CTkLabel(form_frame, text="Username:", text_color=self.font_color).grid(row=0, column=0, padx=10, pady=5,
                                                                                    sticky="w")
        self.register_username = ctk.CTkEntry(form_frame)
        self.register_username.grid(row=0, column=1, padx=10, pady=5)

        # Email Address
        ctk.CTkLabel(form_frame, text="Email Address:", text_color=self.font_color).grid(row=1, column=0, padx=10,
                                                                                         pady=5, sticky="w")
        self.register_email = ctk.CTkEntry(form_frame)
        self.register_email.grid(row=1, column=1, padx=10, pady=5)

        # Phone Number
        ctk.CTkLabel(form_frame, text="Phone Number:", text_color=self.font_color).grid(row=2, column=0, padx=10,
                                                                                        pady=5, sticky="w")
        self.register_phone = ctk.CTkEntry(form_frame)
        self.register_phone.grid(row=2, column=1, padx=10, pady=5)

        # Date of Birth
        ctk.CTkLabel(form_frame, text="Date of Birth:", text_color=self.font_color).grid(row=3, column=0, padx=10,
                                                                                         pady=5, sticky="w")
        self.register_dob = ctk.CTkEntry(form_frame)
        self.register_dob.grid(row=3, column=1, padx=10, pady=5)

        # Location Address
        ctk.CTkLabel(form_frame, text="Location Address:", text_color=self.font_color).grid(row=4, column=0, padx=10,
                                                                                            pady=5, sticky="w")
        self.register_address = ctk.CTkEntry(form_frame)
        self.register_address.grid(row=4, column=1, padx=10, pady=5)

        # Password
        ctk.CTkLabel(form_frame, text="Password:", text_color=self.font_color).grid(row=5, column=0, padx=10, pady=5,
                                                                                    sticky="w")
        self.register_password = ctk.CTkEntry(form_frame, show="*")
        self.register_password.grid(row=5, column=1, padx=10, pady=5)

        # Terms and conditions checkbox
        self.terms_var = ctk.IntVar()
        terms_checkbox = ctk.CTkCheckBox(form_frame, text="I agree to the Terms and Conditions",
                                         variable=self.terms_var)
        terms_checkbox.grid(row=6, columnspan=2, padx=10, pady=5)

        # Register button centered at the bottom
        register_button = ctk.CTkButton(form_frame, text="Register", command=self.register_user)
        register_button.grid(row=8, columnspan=2, pady=20)

    def register_user(self):
        """Handle user registration logic."""
        username = self.register_username.get()
        password = self.register_password.get()
        email = self.register_email.get()
        phone = self.register_phone.get()
        dob = self.register_dob.get()
        address = self.register_address.get()
        hashed_password = generate_password_hash(password, method='sha256')

        # Check if terms and conditions are accepted
        if not self.terms_var.get():
            messagebox.showerror("Error", "You must accept the Terms and Conditions to register.")
            return

        try:
            # Insert user data into the database
            cursor.execute("""
                INSERT INTO users (username, hashed_password, email, phone, dob, address) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, hashed_password, email, phone, dob, address))
            conn.commit()
            messagebox.showinfo("Success", "User registered successfully!")
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username or email already exists!")

    def user_login(self):
        # Create a new top-level window for login
        login_window = ctk.CTkToplevel(self)
        login_window.title("Login")
        login_window.geometry("400x400")
        login_window.resizable(False, False)

        # Load theme and set app icon
        self.load_theme("theme_settings.json")
        # self.app_icon(login_window)
        login_window.iconbitmap('assets\\ai.ico')

        # Apply the loaded theme
        self.apply_theme()

        # Set background image
        #bg_image = self.load_images("assets\\account.png", size=(400, 400))
        #bg_label = ctk.CTkLabel(login_window, image=bg_image)
        #bg_label.image = bg_image  # Keep reference to avoid garbage collection
        #bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Set image to cover the entire window

        # Center image (above username/password fields)
        app_logo = self.load_images("assets\\account.png", size=(100, 100))
        logo_label = ctk.CTkLabel(login_window, image=app_logo, text="")  # Image with no text
        logo_label.image = app_logo
        logo_label.pack(pady=20)

        # Username and password fields
        ctk.CTkLabel(login_window, text="Username:", text_color=self.font_color).pack(pady=5)
        self.login_username = ctk.CTkEntry(login_window)
        self.login_username.pack(pady=5)

        ctk.CTkLabel(login_window, text="Password:", text_color=self.font_color).pack(pady=5)
        self.login_password = ctk.CTkEntry(login_window, show="*")
        self.login_password.pack(pady=5)

        # Login button
        login_button = ctk.CTkButton(login_window, text="Login", command=self.login_user)
        login_button.pack(pady=10)

        # Forget Password option
        forget_password_button = ctk.CTkButton(login_window, text="Forget Password",
                                               command=self.forget_password_window, fg_color="blue")
        forget_password_button.pack(pady=5)

    def forget_password_window(self):
        """Logic for forget password option."""
        forget_window = ctk.CTkToplevel(self)
        forget_window.title("Reset Password")
        forget_window.geometry("350x200")
        forget_window.resizable(False, False)

        # Theme and icon for forget password window
        self.load_theme("theme_settings.json")
        # self.app_icon()
        self.apply_theme()

        ctk.CTkLabel(forget_window, text="Enter your username:", text_color=self.font_color).pack(pady=10)
        self.reset_username = ctk.CTkEntry(forget_window)
        self.reset_username.pack(pady=5)

        # Submit button to handle password reset
        reset_button = ctk.CTkButton(forget_window, text="Reset Password", command=self.reset_password)
        reset_button.pack(pady=20)

    def reset_password(self):
        """Handle the reset password process."""
        username = self.reset_username.get()

        cursor.execute("SELECT email FROM users WHERE username=?", (username,))
        result = cursor.fetchone()

        if result:
            email = result[0]
            # Simulate password reset process
            messagebox.showinfo("Success", f"A password reset link has been sent to {email}")
            # Logic to send reset email could be added here
        else:
            messagebox.showerror("Error", "Invalid username!")

    def login_user(self):
        """Handle the user login logic."""
        username = self.login_username.get()
        password = self.login_password.get()

        # Query the hashed password instead of the password column
        cursor.execute("SELECT hashed_password, email, phone, dob, address FROM users WHERE username=?", (username,))
        result = cursor.fetchone()

        if result and check_password_hash(result[0], password):
            messagebox.showinfo("Success", "Logged in successfully!")
            self.save_login_state()
            self.logged_in_user = username  # Save logged-in user information
            self.user_email = result[1]  # Fetch email
            self.user_phone = result[2]  # Fetch phone
            self.user_dob = result[3]  # Fetch dob
            self.user_address = result[4]  # Fetch address

            # Hide the login button and show the dashboard
            self.hide_login_menu()
            self.show_dashboard()
        else:
            messagebox.showerror("Error", "Invalid username or password!")

    def check_login_state(self):
        """Check if the user is already logged in."""
        if os.path.exists("login_state.json"):
            with open("login_state.json", "r") as file:
                login_state = json.load(file)
                if login_state.get("logged_in"):
                    # self.welcome = ctk.CTkLabel(self.top_frame, text=f"Welcome, {self.logged_in_user}").pack(pady=10)
                    self.save_login_state()

                    return
        self.user_login()

    def save_login_state(self):
        """Save login state to a file."""
        login_state = {"logged_in": True}
        with open("login_state.json", "w") as file:
            json.dump(login_state, file)

    def hide_login_menu(self):
        """Hide login menu or button after the user is logged in."""
        # self.login_menu.pack_forget()  # Assuming `self.login_menu` is your login menu item
        self.menu_frame.pack_forget()

    def show_dashboard(self):
        """Display the user's dashboard after login."""
        dashboard_window = ctk.CTkToplevel(self)
        dashboard_window.title("User Dashboard")
        dashboard_window.geometry("500x500")

        # Frame for the profile image and user information
        info_frame = ctk.CTkFrame(dashboard_window)
        info_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nw")  # Positioning to the right of the image

        # Check if a profile image exists, else load default image
        if self.profile_image_path:
            try:
                img = self.load_images(self.profile_image_path)
            except FileNotFoundError:
                img = self.load_images("assets\\account.png")  # Load a default image if not found
        else:
            img = self.load_images("assets\\account.png")  # Default image if no image is set

        # Create a label and place the image in it
        profile_image_label = ctk.CTkLabel(info_frame, image=img, text="")
        profile_image_label.image = img  # Keep a reference to avoid garbage collection
        profile_image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")  # Positioning the image

        # Display user profile information using grid instead of pack
        ctk.CTkLabel(info_frame, text=f"Welcome, {self.logged_in_user}", font=("Arial", 16)).grid(row=1, column=0,
                                                                                                  padx=5, pady=5,
                                                                                                  sticky="w")
        ctk.CTkLabel(info_frame, text=f"Email: {self.user_email}").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(info_frame, text=f"Phone: {self.user_phone}").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(info_frame, text=f"Date of Birth: {self.user_dob}").grid(row=4, column=0, padx=5, pady=5,
                                                                              sticky="w")
        ctk.CTkLabel(info_frame, text=f"Location Address: {self.user_address}").grid(row=5, column=0, padx=5, pady=5,
                                                                                     sticky="w")

        # Edit profile button
        edit_profile_button = ctk.CTkButton(dashboard_window, text="Edit Profile", command=self.edit_profile)
        edit_profile_button.grid(row=1, column=2, padx=5, pady=5, sticky="w")  # Position below the info frame

        # Logout button
        logout_button = ctk.CTkButton(dashboard_window, text="Sign Out", command=self.logout_user)
        logout_button.grid(row=2, column=2, padx=5, pady=5, sticky="w")

    def edit_profile(self):
        """Open a window to edit user profile information."""
        edit_window = ctk.CTkToplevel(self)
        edit_window.title("Edit Profile")
        edit_window.geometry("400x400")

        edit_frame = ctk.CTkScrollableFrame(edit_window, width=580, height=380)
        edit_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(edit_frame, text="Profile Image").pack(pady=5)
        self.profile_image_path = None  # Store the path of the uploaded image
        self.profile_image_label = ctk.CTkLabel(edit_frame, text="No image selected")
        self.profile_image_label.pack(pady=5)

        # Button to upload image
        upload_button = ctk.CTkButton(edit_frame, text="Upload Image", command=self.upload_file)
        upload_button.pack(pady=5)

        # Form for editing profile
        ctk.CTkLabel(edit_frame, text="Email Address:").pack(pady=5)
        self.edit_email_entry = ctk.CTkEntry(edit_frame)
        self.edit_email_entry.insert(0, self.user_email)  # Pre-fill with current email
        self.edit_email_entry.pack(pady=5)

        ctk.CTkLabel(edit_frame, text="Phone Number:").pack(pady=5)
        self.edit_phone_entry = ctk.CTkEntry(edit_frame)
        self.edit_phone_entry.insert(0, self.user_phone)  # Pre-fill with current phone
        self.edit_phone_entry.pack(pady=5)

        ctk.CTkLabel(edit_frame, text="Date of Birth:").pack(pady=5)
        self.edit_dob_entry = ctk.CTkEntry(edit_frame)
        self.edit_dob_entry.insert(0, self.user_dob)  # Pre-fill with current DOB
        self.edit_dob_entry.pack(pady=5)

        ctk.CTkLabel(edit_frame, text="Location Address:").pack(pady=5)
        self.edit_address_entry = ctk.CTkEntry(edit_frame)
        self.edit_address_entry.insert(0, self.user_address)  # Pre-fill with current address
        self.edit_address_entry.pack(pady=5)

        # Save Changes Button
        save_changes_button = ctk.CTkButton(edit_frame, text="Save Changes", command=self.save_changes)
        save_changes_button.pack(pady=20)

    def upload_file(self):
        """Open a file dialog to select a profile image."""
        file_path = filedialog.askopenfilename(title="Select Profile Image",
                                               filetypes=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")))
        if file_path:
            self.profile_image_path = file_path
            # Update label to show the selected image path
            self.profile_image_label.configure(text=self.profile_image_path)

            # Load the image and display it
            img = self.load_profile_image(file_path)  # Call the corrected image loader
            if img:  # Ensure the image was loaded correctly
                img_tk = ImageTk.PhotoImage(img)  # Convert image to Tkinter-compatible format

                self.profile_image_label.config(image=img_tk, text="")
                self.profile_image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def save_changes(self):
        """Save the edited profile information to the database."""
        new_email = self.edit_email_entry.get()
        new_phone = self.edit_phone_entry.get()
        new_dob = self.edit_dob_entry.get()
        new_address = self.edit_address_entry.get()
        # Save the profile image path if it exists
        profile_image = self.profile_image_path if self.profile_image_path else None

        try:
            # Update user information in the database
            cursor.execute("""
                    UPDATE users SET email=?, phone=?, dob=?, address=?, profile_image=?
                    WHERE username=?
                """, (new_email, new_phone, new_dob, new_address, profile_image, self.logged_in_user))
            conn.commit()

            # Update local user information
            self.user_email = new_email
            self.user_phone = new_phone
            self.user_dob = new_dob
            self.user_address = new_address

            messagebox.showinfo("Success", "Profile updated successfully!")
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def load_profile_image(self, file_path):
        """Load an image from the file path using PIL and return the image object."""
        try:
            img = Image.open(file_path)  # Open the image file
            img = img.resize((150, 150))  # Resize to fit the UI (optional)
            return img
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def logout_user(self):
        """Log the user out and show the login menu again."""
        self.logged_in_user = None  # Clear logged-in user information
        if os.path.exists("login_state.json"):
            os.remove("login_state.json")
            self.user_login()
            self.show_login_menu()  # Show the login menu again
        # Close the dashboard window

    def show_login_menu(self):
        """Show login menu or button."""
        self.menu_frame.pack()

    # Update load_image method to accept size
    def load_image(self, path):
        """Loads an image and converts it to a CTkImage object."""
        img = PILImage.open(path)  # Open the image using PIL
        return ctk.CTkImage(light_image=img, dark_image=img, size=(20, 20))

    def load_images(self, path, size=(20, 20)):
        img = PILImage.open(path)  # Open the image using PIL
        # img = img.resize(PILImage.HUFFMAN_ONLY)  # Resize image
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)

    def load_profie_image(self, path):
        """Loads an image and converts it to a CTkImage object."""
        img = PILImage.open(path)  # Open the image using PIL
        return ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))


if __name__ == "__main__":
    from main import HomeFrame
    app = AIApp()
    #app.run()
    app.mainloop()
