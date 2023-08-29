
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
# Update on 2023-01-13 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-18 00:00:00
# Update on 2023-01-19 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-25 00:00:00
# Update on 2023-02-03 00:00:00
# Update on 2023-02-06 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-09 00:00:00
# Update on 2023-02-10 00:00:00
# Update on 2023-02-10 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-15 00:00:00
# Update on 2023-02-16 00:00:00
# Update on 2023-02-22 00:00:00
# Update on 2023-02-23 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-28 00:00:00
# Update on 2023-02-28 00:00:00
# Update on 2023-03-02 00:00:00
# Update on 2023-03-03 00:00:00
# Update on 2023-03-07 00:00:00
# Update on 2023-03-07 00:00:00
# Update on 2023-03-07 00:00:00
# Update on 2023-03-08 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-10 00:00:00
# Update on 2023-03-15 00:00:00
# Update on 2023-03-20 00:00:00
# Update on 2023-03-23 00:00:00
# Update on 2023-03-23 00:00:00
# Update on 2023-03-24 00:00:00
# Update on 2023-03-28 00:00:00
# Update on 2023-03-29 00:00:00
# Update on 2023-03-29 00:00:00
# Update on 2023-03-30 00:00:00
# Update on 2023-04-03 00:00:00
# Update on 2023-04-06 00:00:00
# Update on 2023-04-06 00:00:00
# Update on 2023-04-07 00:00:00
# Update on 2023-04-07 00:00:00
# Update on 2023-04-07 00:00:00
# Update on 2023-04-10 00:00:00
# Update on 2023-04-11 00:00:00
# Update on 2023-04-12 00:00:00
# Update on 2023-04-13 00:00:00
# Update on 2023-04-13 00:00:00
# Update on 2023-04-14 00:00:00
# Update on 2023-04-14 00:00:00
# Update on 2023-04-17 00:00:00
# Update on 2023-04-21 00:00:00
# Update on 2023-04-21 00:00:00
# Update on 2023-04-25 00:00:00
# Update on 2023-04-27 00:00:00
# Update on 2023-04-28 00:00:00
# Update on 2023-05-01 00:00:00
# Update on 2023-05-01 00:00:00
# Update on 2023-05-04 00:00:00
# Update on 2023-05-05 00:00:00
# Update on 2023-05-08 00:00:00
# Update on 2023-05-09 00:00:00
# Update on 2023-05-09 00:00:00
# Update on 2023-05-10 00:00:00
# Update on 2023-05-11 00:00:00
# Update on 2023-05-15 00:00:00
# Update on 2023-05-17 00:00:00
# Update on 2023-05-17 00:00:00
# Update on 2023-05-19 00:00:00
# Update on 2023-05-19 00:00:00
# Update on 2023-05-19 00:00:00
# Update on 2023-05-23 00:00:00
# Update on 2023-05-23 00:00:00
# Update on 2023-05-24 00:00:00
# Update on 2023-05-25 00:00:00
# Update on 2023-05-25 00:00:00
# Update on 2023-05-26 00:00:00
# Update on 2023-05-30 00:00:00
# Update on 2023-05-31 00:00:00
# Update on 2023-06-06 00:00:00
# Update on 2023-06-07 00:00:00
# Update on 2023-06-07 00:00:00
# Update on 2023-06-08 00:00:00
# Update on 2023-06-12 00:00:00
# Update on 2023-06-12 00:00:00
# Update on 2023-06-15 00:00:00
# Update on 2023-06-22 00:00:00
# Update on 2023-06-23 00:00:00
# Update on 2023-06-23 00:00:00
# Update on 2023-06-29 00:00:00
# Update on 2023-06-30 00:00:00
# Update on 2023-07-04 00:00:00
# Update on 2023-07-04 00:00:00
# Update on 2023-07-06 00:00:00
# Update on 2023-07-06 00:00:00
# Update on 2023-07-07 00:00:00
# Update on 2023-07-07 00:00:00
# Update on 2023-07-11 00:00:00
# Update on 2023-07-11 00:00:00
# Update on 2023-07-13 00:00:00
# Update on 2023-07-13 00:00:00
# Update on 2023-07-17 00:00:00
# Update on 2023-07-18 00:00:00
# Update on 2023-07-19 00:00:00
# Update on 2023-07-21 00:00:00
# Update on 2023-07-21 00:00:00
# Update on 2023-07-24 00:00:00
# Update on 2023-07-27 00:00:00
# Update on 2023-07-27 00:00:00
# Update on 2023-08-02 00:00:00
# Update on 2023-08-03 00:00:00
# Update on 2023-08-04 00:00:00
# Update on 2023-08-08 00:00:00
# Update on 2023-08-10 00:00:00
# Update on 2023-08-11 00:00:00
# Update on 2023-08-15 00:00:00
# Update on 2023-08-17 00:00:00
# Update on 2023-08-17 00:00:00
# Update on 2023-08-18 00:00:00
# Update on 2023-08-21 00:00:00
# Update on 2023-08-22 00:00:00
# Update on 2023-08-25 00:00:00
# Update on 2023-08-29 00:00:00