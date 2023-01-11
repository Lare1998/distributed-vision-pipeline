
from distributed_vision.pipeline import DistributedVisionPipeline
import os
import cv2
import numpy as np

if __name__ == "__main__":
    print("Starting Distributed Vision Pipeline application...")

    # Create a dummy video file for testing
    dummy_video_path = "dummy_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dummy_video_path, fourcc, 20.0, (640, 480))
    for _ in range(100): # 100 frames
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    pipeline = DistributedVisionPipeline(dummy_video_path, num_workers=4)
    results = pipeline.run()
    print(f"Total frames processed: {len(results)}")
    
    # Clean up dummy video
    os.remove(dummy_video_path)

    print("Distributed Vision Pipeline application finished.")

# Update on 2023-01-02 00:00:00
# Update on 2023-01-02 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-06 00:00:00
# Update on 2023-01-09 00:00:00
# Update on 2023-01-09 00:00:00
# Update on 2023-01-11 00:00:00
# Update on 2023-01-11 00:00:00