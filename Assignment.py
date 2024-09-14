import tkinter as tk

def on_button_click():
    label.config(text="Button Clicked!")

# Create the main window
root = tk.Tk()
root.title("Simple GUI")

# Create a label
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

# Create a button
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack()

# Run the application
root.mainloop()
