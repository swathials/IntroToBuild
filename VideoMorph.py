import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

#Initialize Tkinter window
root = tk.Tk()
root.title("Motion Blur Webcam")
root.attributes('-fullscreen', True)
video_label = tk.Label(root)
video_label.pack(fill=tk.BOTH, expand=True)

#Open webcam
cap = cv2.VideoCapture(0)
alpha = 0.2
blurred_frame = None

def update_frame():
    global blurred_frame  # Keeps the previous frame reference
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return
    frame = frame.astype(np.float32)  # Convert to float32 for accumulation

    # Initialize blurred frame only once
    if blurred_frame is None:
        blurred_frame = np.zeros_like(frame)

    # Apply in-place motion blur effect
    cv2.accumulateWeighted(frame, blurred_frame, alpha)
    blurred= (cv2.convertScaleAbs(blurred_frame)).astype(np.float32)
    blended = cv2.divide(frame, 255-blurred, scale=256)
    #corrected=cv2.divide(blurred,255-blended, scale=256)
    corrected=cv2.absdiff(blurred,blended)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(corrected)
    r = np.roll(r, shift=5, axis=1)  # Shift red channel
    g = np.roll(g, shift=-5, axis=0) # Shift green channel
    rgb = np.clip((cv2.merge([b, g, r])),0,255).astype(np.uint8)

    final=np.clip(cv2.addWeighted((blended).astype(np.uint8),0.9,rgb,0.1,5), 0, 255).astype(np.uint8)
    frame_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(np.fliplr(frame_rgb))

    # Resize only if needed (to prevent redundant memory operations)
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    if img_pil.size != (screen_width, screen_height):
        img_pil = img_pil.resize((screen_width, screen_height), Image.Resampling.LANCZOS)

    imgtk = ImageTk.PhotoImage(img_pil)

    # Update Label with new frame (keep reference to prevent memory leak)
    video_label.config(image=imgtk)
    video_label.imgtk = imgtk  

    # Call update_frame again after 10ms
    root.after(10, update_frame)

# Start updating frames
update_frame()

# Run Tkinter main loop
root.mainloop()

# Release webcam when window is closed
cap.release()
cv2.destroyAllWindows()
