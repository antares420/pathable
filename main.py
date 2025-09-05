import cv2
import numpy as np
import torch
import pyttsx3
import time

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Slow, clear speech
engine.setProperty('voice', 'english')  # Neutral voice

# Load MiDaS model for depth estimation
model_type = "DPT_Large"  # MiDaS model variant
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Load MiDaS transforms
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Connect to phone camera (e.g., via DroidCam IP stream)
cap = cv2.VideoCapture("http://192.168.1.100:4747/video")  # Replace with your phone’s IP stream

def estimate_distance(frame):
    # Preprocess frame for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    # Normalize depth to approximate meters (heuristic, not precise)
    depth_map = depth_map / depth_map.max() * 10  # Scale to ~10m max
    return depth_map

def analyze_frame(frame, depth_map):
    # Split frame into left and right halves
    height, width = frame.shape[:2]
    left_depth = depth_map[:, :width//2]
    right_depth = depth_map[:, width//2:]

    # Average distance in each half (higher depth = farther obstacles)
    left_avg = np.mean(left_depth)
    right_avg = np.mean(right_depth)

    # Find closest obstacle overall
    min_distance = np.min(depth_map)

    return min_distance, "left" if left_avg > right_avg else "right"

def provide_guidance(distance, direction):
    if distance < 1.0:  # Threshold ~1 meter
        instruction = f"Turn {direction} now"
        print(instruction)
        engine.say(instruction)
        engine.runAndWait()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Estimate distance using depth map
        depth_map = estimate_distance(frame)
        min_distance, direction = analyze_frame(frame, depth_map)
        print(f"Closest obstacle: {min_distance:.2f} meters")

        # Provide guidance if obstacle is too close
        provide_guidance(min_distance, direction)

        # Display frame and depth map (for debugging)
        cv2.imshow('Camera', frame)
        cv2.imshow('Depth Map', cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # Control loop speed

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()