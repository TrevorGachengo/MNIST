from tkinter import *
from PIL import Image, ImageGrab
import os
import signal
import sys
import numpy as np
import imageio.v2 as iio

module_dir = os.path.abspath('network')
sys.path.append(os.path.dirname(module_dir))
from network import Network

# Global variables
root = Tk()
root.title("Interfacer")
b1 = "up"
xold, yold = None, None
image_counter = 1
save_dir = 'interfacer/temp-images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize Network
net = Network()
net.load_parameters()

# Signal handler for cleanup on exit
def signal_handler(sig, frame):
    print("Cleaning up temporary images...")
    for entry in os.listdir(save_dir):
        file_path = os.path.join(save_dir, entry)
        if os.path.isfile(file_path):
            os.remove(file_path)
    sys.exit()

def on_close():
    for entry in os.listdir(save_dir):
        file_path = os.path.join(save_dir, entry)
        if os.path.isfile(file_path):
            os.remove(file_path)
    root.destroy()
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)

# Function to save canvas content as images
def save_image():
    global image_counter
    x = root.winfo_rootx() + drawing_area.winfo_x()
    y = root.winfo_rooty() + drawing_area.winfo_y()
    x1 = x + drawing_area.winfo_width()
    y1 = y + drawing_area.winfo_height()

    path = os.path.join(save_dir, f"{image_counter}.png")
    ImageGrab.grab().crop((x, y, x1, y1)).save(path)
    image_counter += 1
    root.after(100, save_image)  # Schedule the next save

def delete_image():
    global image_counter
    if image_counter > 2:
        file_path = os.path.join(save_dir, f"{image_counter - 2}.png")
        os.remove(file_path)
    root.after(100, delete_image)

# Function to predict and update label
def test_image(network):
    dir = 'interfacer/temp-images'
    guesses = []
    for img in os.listdir(dir):
        newdir = os.path.join(dir, img)
        img_array = convert_image(iio.imread(newdir)).reshape((784, 1))
        guess = network.use_network(img_array)
        guesses.append(f"Guess = {guess}")

    result_label.config(text="\n".join(guesses))
    root.after(300, test_image, network)

# Convert image to greyscale 28x28
def convert_image(image):
    pil_image = Image.fromarray(image) if image.ndim == 3 else Image.fromarray(image, mode='L')
    resized_image = pil_image.resize((28, 28), Image.LANCZOS)
    grey_image = np.array(resized_image.convert('L'))
    return 1 - (grey_image / 255)

# Drawing functions
def b1down(event):
    global b1, xold, yold
    b1 = "down"
    xold, yold = event.x, event.y

def b1up(event):
    global b1
    b1 = "up"
    xold, yold = None, None

def motion(event):
    global xold, yold
    if b1 == "down":
        drawing_area.create_line(xold, yold, event.x, event.y, width=50, fill='black', capstyle=ROUND, smooth=True)
        xold, yold = event.x, event.y

# Canvas and Widgets Setup
drawing_area = Canvas(root, width=600, height=600, bg='white')
drawing_area.pack()
drawing_area.bind("<Motion>", motion)
drawing_area.bind("<ButtonPress-1>", b1down)
drawing_area.bind("<ButtonRelease-1>", b1up)

clear_button = Button(root, fg="green", text="Clear", command=lambda: drawing_area.delete("all"))
clear_button.pack(side=LEFT)

# Label for displaying results
result_label = Label(root, font=("Arial", 14))
result_label.pack(side=TOP, pady=10)

# Start periodic image saving
save_image()
delete_image()

# Start predictions
test_image(net)

# Clear temp images on close
root.protocol("WM_DELETE_WINDOW", on_close)
# Run the Tkinter main loop
root.mainloop()
