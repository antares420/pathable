#!/usr/bin/env python3
"""
Local Camera Navigation Test
Run this on your laptop to test with your built-in camera
"""

import cv2
import numpy as np
import time
import pyttsx3
import threading

class LocalNavigationSystem:
    def __init__(self):
        self.running = False
        self.cap = None
        
        # Initialize text-to-speech
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.tts_available = True
        except:
            print("Text-to-speech not available")
            self.tts_available = False
    
    def speak(self, text):
        """Speak text using TTS"""
        if self.tts_available:
            def tts_thread():
                self.engine.say(text)
                self.engine.runAndWait()
            
            thread = threading.Thread(target=tts_thread)
            thread.daemon = True
            thread.start()
        else:
            print(f"SPEECH: {text}")
    
    def analyze_frame(self, frame):
        """Analyze camera frame for navigation guidance"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Split frame into left and right halves
        height, width = edges.shape
        left_half = edges[:, :width//2]
        right_half = edges[:, width//2:]
        
        # Count edge pixels (lower count = more open space)
        left_score = np.sum(left_half) / 255
        right_score = np.sum(right_half) / 255
        
        # Estimate distance based on bottom portion of frame
        bottom_portion = edges[int(height*0.7):, :]
        edge_density = np.sum(bottom_portion) / (bottom_portion.shape[0] * bottom_portion.shape[1] * 255)
        
        # Convert edge density to distance estimate (rough approximation)
        estimated_distance = max(20, 200 - (edge_density * 1000))
        
        return {
            'distance': int(estimated_distance),
            'direction': 'left' if left_score < right_score else 'right',
            'edge_density': edge_density
        }
    
    def connect_camera(self, camera_id=0):
        """Connect to laptop camera"""
        print(f"Attempting to connect to camera {camera_id}...")
        
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return False
        
        # Test if we can read a frame
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Camera opened but can't read frames")
            self.cap.release()
            return False
        
        print(f"✅ Successfully connected to camera {camera_id}")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        return True
    
    def start_navigation(self):
        """Start the navigation system"""
        if not self.cap or not self.cap.isOpened():
            print("❌ No camera connected!")
            return
        
        self.running = True
        print("\n🧭 NAVIGATION SYSTEM STARTED")
        print("=" * 50)
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press SPACE to toggle voice guidance")
        print("- Camera window shows your view")
        print("=" * 50)
        
        self.speak("Navigation system started")
        
        voice_enabled = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Lost camera connection")
                    break
                
                # Analyze frame for navigation
                analysis = self.analyze_frame(frame)
                distance = analysis['distance']
                direction = analysis['direction']
                
                # Create visual feedback on frame
                display_frame = frame.copy()
                
                # Draw center line
                height, width = display_frame.shape[:2]
                cv2.line(display_frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
                
                # Add text overlay
                status_text = f"Distance: {distance}cm | Direction: {direction}"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Color code the distance
                if distance < 50:
                    color = (0, 0, 255)  # Red for danger
                    status = "OBSTACLE DETECTED!"
                    instruction = f"Turn {direction} now!"
                    
                    if voice_enabled:
                        self.speak(instruction)
                        
                elif distance < 100:
                    color = (0, 165, 255)  # Orange for caution
                    status = "CAUTION"
                    instruction = f"Obstacle ahead - prepare to turn {direction}"
                else:
                    color = (0, 255, 0)  # Green for safe
                    status = "PATH CLEAR"
                    instruction = "Safe to continue"
                
                # Add status overlay
                cv2.putText(display_frame, status, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display_frame, instruction, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show frame
                cv2.imshow('Navigation Camera - Laptop Test', display_frame)
                
                # Print status to console
                print(f"📍 Distance: {distance:3d}cm | Direction: {direction:>5} | Status: {status}")
                
                # Handle keyboard input
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Navigation stopped by user")
                    break
                elif key == ord(' '):
                    voice_enabled = not voice_enabled
                    status_msg = "enabled" if voice_enabled else "disabled"
                    print(f"🔊 Voice guidance {status_msg}")
                    self.speak(f"Voice guidance {status_msg}")
                
        except KeyboardInterrupt:
            print("\n🛑 Navigation stopped")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def main():
    """Main function to run the local navigation test"""
    print("🧭 LOCAL NAVIGATION SYSTEM")
    print("=" * 40)
    
    nav = LocalNavigationSystem()
    
    # Try to connect to camera
    camera_connected = False
    for camera_id in [0, 1, 2]:
        if nav.connect_camera(camera_id):
            camera_connected = True
            break
    
    if not camera_connected:
        print("\n❌ No cameras found!")
        print("Make sure your laptop camera is not being used by another application")
        return
    
    print("\n✅ Camera ready!")
    print("Press ENTER to start navigation or 'q' to quit")
    
    user_input = input().strip().lower()
    if user_input == 'q':
        nav.cleanup()
        return
    
    # Start navigation
    nav.start_navigation()

if __name__ == "__main__":
    main()