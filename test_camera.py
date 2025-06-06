#!/usr/bin/env python3
"""
Simple camera test for Mirror Mirror system
"""

import cv2
import numpy as np
import time

def test_camera(camera_id: int = 0):
    """Test camera capture and basic functionality"""
    print(f"Testing camera {camera_id}...")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        print("Available cameras:")
        for i in range(5):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"  - Camera {i}: Available")
                test_cap.release()
            else:
                print(f"  - Camera {i}: Not available")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened: {width}x{height} @ {fps:.1f} FPS")
    
    # Capture frames
    frame_count = 0
    start_time = time.time()
    
    print("Capturing frames... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        frame_count += 1
        
        # Add frame counter and FPS
        current_time = time.time()
        elapsed = current_time - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Size: {width}x{height}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Camera Test', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    final_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"✅ Test completed: {frame_count} frames in {elapsed:.1f}s ({final_fps:.1f} FPS)")
    
    return True

if __name__ == "__main__":
    import sys
    
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    print("🪞 Mirror Mirror - Camera Test")
    print("===============================")
    print("Press 'q' to quit, 's' to save frame")
    print("")
    
    success = test_camera(camera_id)
    
    if success:
        print("✅ Camera test passed!")
    else:
        print("❌ Camera test failed!")
        sys.exit(1) 