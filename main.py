import numpy as np
from typing import List
import PIL

# functions we'll need
def load_camera_video():
    ''' This function loads a camera file into computer memory
    This should output image stack
    Inputs:
    '''
    pass

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

if __name__ == '__main___':
    print("Hello World")
    print("you didn't think I was going to code that much for you did you?")
    
