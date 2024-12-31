import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np

class FacialFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        # UI Layout
        self.configure_grid()
        self.create_widgets()
        self.camera_feed_running = False
    def configure_grid(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

    def create_widgets(self):
        # Left-side buttons
        self.button_frame = ctk.CTkFrame(self, width=200)
        self.button_frame.grid(row=0, column=0, sticky="nswe")

        self.load_image_button = ctk.CTkButton(self.button_frame, text="Load Image", command=self.load_image)
        self.load_image_button.pack(pady=10, padx=10)

        self.apply_filter_button = ctk.CTkButton(self.button_frame, text="Apply Filter", command=self.apply_filter)
        self.apply_filter_button.pack(pady=10, padx=10)

        self.start_camera_button = ctk.CTkButton(self.button_frame, text="Start Camera", command=self.start_camera)
        self.start_camera_button.pack(pady=10, padx=10)

        self.stop_camera_button = ctk.CTkButton(self.button_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_camera_button.pack(pady=10, padx=10)

        self.save_button = ctk.CTkButton(self.button_frame, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=10, padx=10)

        # Right-side image preview
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, sticky="nswe")

        self.image_label = ctk.CTkLabel(self.image_frame, text="", anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.current_image = None
        self.current_cv_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            image = Image.open(file_path)
            self.display_image(image)

    def display_image(self, image):
        self.current_image = image
        self.current_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize the image to fit the label dimensions
        self.image_label.update_idletasks()
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        resized_image = self.current_image.resize((label_width, label_height), Image.HUFFMAN_ONLY)
        img_tk = ImageTk.PhotoImage(resized_image)

        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def apply_filter(self):
        if self.current_cv_image is not None:
            # Apply a simple color filter as an example
            hsv_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([100, 50, 50])
            upper_bound = np.array([130, 255, 255])
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            result = cv2.bitwise_and(self.current_cv_image, self.current_cv_image, mask=mask)
            self.display_image(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))

    def start_camera(self):
        if not self.camera_feed_running:
            self.camera_feed_running = True
            self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
            self.camera_thread.start()

    def stop_camera(self):
        self.camera_feed_running = False

    def update_camera_feed(self):
        cap = cv2.VideoCapture(0)
        while self.camera_feed_running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                resized_frame = cv2.resize(frame, (800, 600))
                self.current_cv_image = resized_frame
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                self.display_image(image_pil)
            self.update()
        cap.release()

    def save_image(self):
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp")])
            if file_path:
                self.current_image.save(file_path)

