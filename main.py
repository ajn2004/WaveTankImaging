import numpy as np
from typing import List
from PIL import Image
import cv2

# functions we'll need
def load_camera_video(videoFile: str, max_frames: int|None=None, convert_to_rgb: bool=False) -> np.ndarray:
    ''' This function loads a camera file into computer memory
    This should output image stack
    Inputs:
    '''

    cap = cv2.VideoCapture(videoFile)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and count >= max_frames):
            break

        if convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame)
        frames.append(pil_image)
        count += 1

    cap.release()
    return np.array(frames)


def analyze_frame_stack(video_stack: np.ndarray) -> List:
    '''
    Takes in camera video frame stack, and loops through each frame to determine amplitude
    Should output Array<Array<Floats>> <frame, amplitude>
    '''
    video_amplitudes = []
    for frame in video_stack:
        amplitude_array = analyze_frame(frame)
        video_amplitudes.append(amplitude_array)
    return video_amplitudes

def analyze_frame(frame: np.ndarray):
    ''' Determine the amplitude of the wave on a frame
    This should output array(float) this array is our array of amplitudes
    '''
    pass

def determine_pixel_size(frame: np.ndarray):
    '''
    Determine pixel size from a image frame
    Outputs a float
    '''
    pass

if __name__ == '__main__':
    # Set video path
    video_file_path = 'data/wave_movie_1.mp4'
    # say what we're trying to do
    print(f"Attempting to load a video at {video_file_path}")
    # load video
    video = load_camera_video(video_file_path)
    # if video is successfully loaded, size should be larger than 0
    print(f"Loaded video with {video.size} frames")
