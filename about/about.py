import customtkinter as ctk
from PIL import Image as PILImage, ImageTk
from PIL import Image, ImageTk


class AboutFrame(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master)
        # Example sections with images and text
        self.add_about_section(self, "About the App", "assets\\ai.png",
                               "AI Intelligent System is designed to provi8de users with advanced tools for AI/ML tasks.")

        self.add_about_section(self, "Developers", "assets\\about.png",
                               "Developed by Slogan Technologies, aiming to revolutionize AI applications globally.")

        self.add_about_section(self, "Technology Stack", "assets\\chatbot.png",
                               "Built with Python, TensorFlow, OpenCV, and CustomTkinter for a seamless user experience.")

        self.add_about_section(self, "Version", "assets\\ai.png",
                               "Current Version: 1.0.0\nRelease Date: September 2024")

        self.add_about_section(self, "About Us", "assets\\slogan.png",
                               "We are a startup robotics research development firm powered by game development, "
                               "operating in the USA and Nigeria.")

    def add_about_section(self, parent_frame, section_title, image_path, description):
        """Helper function to add sections in the About window with image and text."""
        section_frame = ctk.CTkFrame(parent_frame)
        section_frame.pack(pady=10, padx=10, fill="x")

        # Section Image
        image = self.load_about_images(image_path)
        image_label = ctk.CTkLabel(section_frame, image=image, text="")
        image_label.image = image  # Keep reference to avoid garbage collection
        image_label.grid(row=0, column=0, padx=10, pady=5)

        # Section Text
        text_frame = ctk.CTkFrame(section_frame)
        text_frame.grid(row=0, column=1, sticky="w")

        section_label = ctk.CTkLabel(text_frame, text=section_title, font=("Arial", 20))
        section_label.pack(anchor="w")

        description_label = ctk.CTkLabel(text_frame, text=description, wraplength=400, justify="left")
        description_label.pack(anchor="w", pady=5)

    def load_about_images(self, path, size=(100, 100)):
        img = PILImage.open(path)  # Open the image using PIL
        img = img.resize(size)  # Resize image
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
