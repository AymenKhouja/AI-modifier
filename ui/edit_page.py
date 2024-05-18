import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class EditImagePage(tk.Toplevel):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.parent = parent
        self.title("Edit Image")

        self.image = image
        self.zoom_factor = 1.0

        # Add scrollbar for zooming
        self.scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.on_scroll)
        self.scrollbar.pack(side="bottom", fill="x")

        # Create Canvas for displaying the image
        self.canvas = tk.Canvas(self, bg="white", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Display the image on the canvas
        self.display_image()

        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        # Bind mouse wheel events for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        # Bind mouse events for panning
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move)

        self.start_x = None
        self.start_y = None

    def display_image(self):
        # Clear previous image
        self.canvas.delete("image")

        # Resize image based on zoom factor
        width = int(self.image.width * self.zoom_factor)
        height = int(self.image.height * self.zoom_factor)
        resized_image = self.image.resize((width, height), Image.NEAREST)

        # Create PhotoImage object from resized image
        photo = ImageTk.PhotoImage(resized_image)

        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="image")
        self.canvas.image = photo

        # Update scrollbar position
        self.scrollbar.set(0, 1.0)

    def on_scroll(self, *args):
        # Update zoom factor based on scrollbar position
        self.zoom_factor = float(args[0])
        self.display_image()

    def on_mousewheel(self, event):
        # Change zoom factor based on mouse wheel direction
        if event.delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor /= 1.1  # Zoom out

        # Limit zoom factor to avoid excessive zooming
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))

        # Update scrollbar position
        self.scrollbar.set(0, 1.0)

        # Display the image with the new zoom factor
        self.display_image()

    def on_button_press(self, event):
        # Record starting position for panning
        self.start_x = event.x
        self.start_y = event.y

    def on_move(self, event):
        if self.start_x is not None and self.start_y is not None:
            # Calculate distance moved
            delta_x = event.x - self.start_x
            delta_y = event.y - self.start_y
            
            # Update starting position
            self.start_x = event.x
            self.start_y = event.y

            # Move the image
            self.canvas.xview_scroll(-delta_x, "units")
            self.canvas.yview_scroll(-delta_y, "units")
