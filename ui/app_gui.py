import tkinter as tk
import requests
from io import BytesIO
from PIL import Image, ImageTk
from tkinter import filedialog
from edit_page import EditImagePage


class WelcomePage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Welcome Page")
        
        self.label = tk.Label(self)
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self, text="Upload Image", command=self.load_image)
        self.upload_button.pack(pady=5)

        self.url_label = tk.Label(self, text="Or enter URL:")
        self.url_label.pack()

        self.url_entry = tk.Entry(self, width=40)
        self.url_entry.pack()

        self.load_button = tk.Button(self, text="Load from URL", command=self.load_from_url)
        self.load_button.pack(pady=5)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack()
        
        # Set initial window size to the maximum size of the monitor
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")

    def load_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if filename:
            image = Image.open(filename)
            self.open_edit_page(image)


    def load_from_url(self):
        url = self.url_entry.get()
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            self.open_edit_page(image)
        except Exception as e:
            self.status_label.config(text="Failed to load image from URL")

    def display_image(self, image):
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo

    def open_edit_page(self, image):
        self.withdraw()  # Hide the main window
        self.edit_page = EditImagePage(self, image)
        self.edit_page.protocol("WM_DELETE_WINDOW", self.on_edit_page_close)
        self.edit_page.mainloop()

    def on_edit_page_close(self):
        self.edit_page.destroy()
        self.deiconify()  # Show the main window again


def main():
    app = WelcomePage()
    app.mainloop()

if __name__ == "__main__":
    main()
