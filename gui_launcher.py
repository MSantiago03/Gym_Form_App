import tkinter as tk
from tkinter import filedialog
from evaluate_model import evaluate_video

# -------------------------------
# GUI SETUP
# -------------------------------
def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        evaluate_video(file_path)

root = tk.Tk()
root.title("Form Correction Evaluator")
root.geometry("300x150")

label = tk.Label(root, text="Select an MP4 video to evaluate", font=("Helvetica", 12))
label.pack(pady=20)

button = tk.Button(root, text="Choose Video", command=select_video, font=("Helvetica", 12))
button.pack()

root.mainloop()
