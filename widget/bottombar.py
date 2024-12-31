import customtkinter as ctk

class BottomBar(ctk.CTk):
    def __init__(self):
        super().__init__()


        # Create a main container to hold the content and bottom navbar
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Create a content frame that will change based on navigation
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, pady=(0, 50))  # Reserve space for navbar at bottom

        # Create the bottom navigation bar frame
        self.navbar_frame = ctk.CTkFrame(self.main_frame, height=50, fg_color="#333")
        self.navbar_frame.pack(side="bottom", fill="x")

        # Create bottom navigation buttons/icons
        self.home_button = ctk.CTkButton(self.navbar_frame, text="Home", command=self.show_home, width=150, fg_color="transparent")
        self.home_button.grid(row=0, column=0, padx=10)

        self.search_button = ctk.CTkButton(self.navbar_frame, text="Search", command=self.show_search, width=150, fg_color="transparent")
        self.search_button.grid(row=0, column=1, padx=10)

        self.profile_button = ctk.CTkButton(self.navbar_frame, text="Profile", command=self.show_profile, width=150, fg_color="transparent")
        self.profile_button.grid(row=0, column=2, padx=10)

        # Create frames for each section of the app
        self.home_frame = ctk.CTkFrame(self.content_frame)
        self.search_frame = ctk.CTkFrame(self.content_frame)
        self.profile_frame = ctk.CTkFrame(self.content_frame)

        # Add content to the frames
        self.home_label = ctk.CTkLabel(self.home_frame, text="Welcome to the Home Screen!", font=("Arial", 24))
        self.home_label.pack(pady=20)

        self.search_label = ctk.CTkLabel(self.search_frame, text="Search for content here.", font=("Arial", 24))
        self.search_label.pack(pady=20)

        self.profile_label = ctk.CTkLabel(self.profile_frame, text="Your Profile", font=("Arial", 24))
        self.profile_label.pack(pady=20)

        # Start by showing the Home frame
        self.show_home()

    def show_home(self):
        self.hide_all_frames()
        self.home_frame.pack(fill="both", expand=True)

    def show_search(self):
        self.hide_all_frames()
        self.search_frame.pack(fill="both", expand=True)

    def show_profile(self):
        self.hide_all_frames()
        self.profile_frame.pack(fill="both", expand=True)

    def hide_all_frames(self):
        self.home_frame.pack_forget()
        self.search_frame.pack_forget()
        self.profile_frame.pack_forget()

if __name__ == "__main__":
    app = BottomBar()
    # app.run()
    app.mainloop()