import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from evaluate_model import evaluate_video

# -------------------------------
# Model paths for each exercise
# -------------------------------
MODEL_PATHS = {
    "pushup": "models/form_rnn_pushup_FINAL_VERSION.pth",
    "squat": "models/form_rnn_squat.pth"
}

def select_video(exercise_type):
    """
    Opens a file dialog to select a video, then evaluates it
    using the corresponding model for the chosen exercise.
    """
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        model_path = MODEL_PATHS.get(exercise_type)
        evaluate_video(file_path, model_path)

# -------------------------------
# GUI setup
# -------------------------------
root = tk.Tk()
root.title("Form Correction Evaluator")
root.geometry("1000x500")  # Set window size

# -------------------------------
# Custom Orange Button Style (ttk)
# -------------------------------
style = ttk.Style()
style.theme_use("clam")  # Use a theme that allows full styling

# Define layout and color for custom button
style.layout("Orange.TButton",
    [('Button.border', {'children': [
        ('Button.padding', {'children': [
            ('Button.label', {'sticky': 'nswe'})
        ]})
    ]})]
)

# Apply styling for normal and active button states
style.configure("Orange.TButton",
    background="#FFA500",      # Orange background
    foreground="white",        # White text
    font=("Helvetica", 14),
    padding=10,
    relief="flat"
)
style.map("Orange.TButton",
    background=[("active", "#e69500"), ("!active", "#FFA500")],
    foreground=[("disabled", "gray"), ("!disabled", "white")]
)

# -------------------------------
# Header (title, profile, sign out)
# -------------------------------
header_frame = tk.Frame(root, pady=10, bg="white")
header_frame.pack(fill=tk.X)

# App title
app_name = tk.Label(header_frame, text="Form Correction App", font=("Helvetica", 18, "bold"), bg="white", fg="black")
app_name.pack(side=tk.TOP)

# Right side of header (profile + sign out)
right_frame = tk.Frame(header_frame, bg="white")
right_frame.pack(side=tk.RIGHT)

# Profile icon (text fallback)
profile_label = tk.Label(right_frame, text="👤", font=("Helvetica", 14), bg="white", fg="black")
profile_label.pack(side=tk.LEFT, padx=5)

# Sign out button (styled as a clickable label)
signout_button = tk.Button(
    right_frame,
    text="Sign Out",
    font=("Helvetica", 10, "underline"),
    bg="white",
    fg="blue",
    activebackground="white",
    activeforeground="blue",
    relief=tk.FLAT,
    bd=0,
    cursor="hand2",
    takefocus=0
)
signout_button.pack(side=tk.LEFT, padx=5)

# -------------------------------
# Main Content Area
# -------------------------------
main_frame = tk.Frame(root)
main_frame.pack(pady=40)

# Instruction label
label = tk.Label(main_frame, text="Select an MP4 video to evaluate", font=("Helvetica", 14))
label.pack(pady=20)

# Orange action buttons for each exercise type
ttk.Button(main_frame, text="Evaluate Pushups", command=lambda: select_video("pushup"), style="Orange.TButton").pack(pady=10)
ttk.Button(main_frame, text="Evaluate Squats", command=lambda: select_video("squat"), style="Orange.TButton").pack(pady=10)

# Start the GUI event loop
root.mainloop()
