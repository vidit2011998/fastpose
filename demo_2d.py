from datetime import time
import cv2
import sys
import os
from src.system.interface import AnnotatorInterface
from src.utils.drawer import Drawer
import time
from sys import argv
from pathlib import Path

print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)



"""
Read the movie located at moviePath, perform the 2d pose annotation and display
Run from terminal : python demo_2d.py [movie_file_path] [max_persons_detected]
with all parameters optional.
Keep holding the backspace key to speed the video 30x
"""



def start(video_file, max_persons):
    video_path = Path(video_file)
    
    annotator = AnnotatorInterface.build(max_persons=max_persons)
    annotator.set_video_label(video_path)
    
    cap = cv2.VideoCapture(str(video_path))

    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tmpTime = time.time()
        persons = annotator.update(frame)
        fps = int(1/(time.time()-tmpTime))

        poses = [p['pose_2d'] for p in persons]
        ids = [p['id'] for p in persons]
        frame = Drawer.draw_scene(frame, poses, ids, fps, cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(33) == ord(' '):
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

    annotator.terminate()
    cap.release()
    cv2.destroyAllWindows()



def process_directory(directory_path, max_persons):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Add more if needed
    directory = Path(directory_path)
    
    print(f"Scanning directory: {directory}")
    
    video_count = 0
    for video_file in directory.glob('**/*'):
        if video_file.suffix.lower() in video_extensions:
            video_count += 1
            print(f"Processing video {video_count}: {video_file}")
            start(video_file, max_persons)
    
    if video_count == 0:
        print("No video files found in the specified directory.")




if __name__ == "__main__":
    print("start frontend")

    max_persons = 1

    if len(argv) < 2:
        print("Usage: python demo_2d.py <directory_path> [max_persons]")
        exit(1)

    directory_path = argv[1]
    if len(argv) == 3:
        max_persons = int(argv[2])

    print(f"Directory path: {directory_path}")
    print(f"Max persons: {max_persons}")

    try:
        process_directory(directory_path, max_persons)
    except Exception as e:
        print(f"An error occurred: {e}")

    print("Processing complete.")


