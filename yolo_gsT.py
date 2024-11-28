import cv2
import torch
from ultralytics import YOLO
import gi
import numpy as np
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from threading import Thread

def draw_corner_brackets(img, bbox, color=(0, 255, 0), thickness=2, length=15):
    """
    Draws corner brackets on the image for a given bounding box.
    """
    x1, y1, x2, y2 = bbox

    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

class GstCamera:
    def __init__(self):
        Gst.init(None)
        
        # Create GStreamer pipeline
        self.pipeline_str = (
            'v4l2src device=/dev/video0 ! '
            'image/jpeg,width=1280,height=720,framerate=30/1 ! '
            'jpegdec ! videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink name=sink emit-signals=True sync=false max-buffers=1 drop=True'
        )
        
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        self.sink.connect('new-sample', self.on_new_sample)
        
        # Initialize variables
        self.frame = None
        self.latest_frame = None
        
        # Start the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Get buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        # Convert buffer to numpy array
        self.latest_frame = np.ndarray(
            shape=(720, 1280, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def get_frame(self):
        return self.latest_frame.copy() if self.latest_frame is not None else None

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)

def main():
    # Initialize YOLO model
    model = YOLO('yolo11s.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Using device: {device}")

    # Initialize GStreamer camera
    camera = GstCamera()
    
    # Create display window
    cv2.namedWindow('YOLOv11s Tracking', cv2.WINDOW_NORMAL)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            
            # Run tracking instead of just detection
            results = model.track(frame, tracker="botsort.yaml", persist=True)
            tracked_objects = results[0]
            
            # Process tracking results
            if tracked_objects.boxes is not None:
                for box in tracked_objects.boxes:
                    # Get the bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class, confidence and tracking ID
                    cls_id = int(box.cls[0])
                    score = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Get tracking ID if available
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    # Draw detections with different colors for different tracks
                    if track_id is not None:
                        # Generate unique color for each track_id
                        color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
                        draw_corner_brackets(frame, (x1, y1, x2, y2), color=color)
                        label = f"{class_name}-{track_id}: {score:.2f}"
                    else:
                        draw_corner_brackets(frame, (x1, y1, x2, y2), color=(0, 255, 0))
                        label = f"{class_name}: {score:.2f}"
                    
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 128), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('YOLOv11s Tracking', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()