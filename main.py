import numpy as np
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
from sklearn.decomposition import PCA
from scipy import ndimage

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

def user_select_pixel_height(frame: np.ndarray) -> float:
    '''
    this function should present the frame to the user and ask them to select 2 points
    Then calculate the absolute height difference between those points, and return
    that difference to the user
    '''
    # Store clicked points
    clicked_points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            clicked_points.append((int(event.xdata), int(event.ydata)))
            print(f"Point {len(clicked_points)}: ({int(event.xdata)}, {int(event.ydata)})")

            if len(clicked_points) == 2:
                plt.close()

    # Display the image
    fig, ax = plt.subplots()
    manager = plt.get_current_fig_manager()
    try:
        # For most backends like TkAgg
        manager.full_screen_toggle()
    except AttributeError:
        # For Qt or other GUI backends, use this fallback
        manager.window.showMaximized()
    ax.imshow(frame)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click two points")
    plt.show()

    y1 = clicked_points[0][1]
    y2 = clicked_points[1][1]
    height_diff = abs(y1 - y2)

    return height_diff

def determine_pixel_size(frame: np.ndarray) -> float:
    '''
    Determine pixel size from a image frame
    Outputs a float
    '''
    # offer user the ability to select 2 pixels (high, low)
    pixel_height = user_select_pixel_height(frame)
    # ask user for the size of the wave tank
    units = set(['in','cm','mm']);
    while True:
        print("Give the height of the wave tank in a decimal with units, i.e. 20.3in or 34.2cm")
        user_input = input("What is the height of the wave tank?")
        user_units = user_input[-2:]
        if user_input[-2:] in units: # force user to specify measurement units
            height = float(user_input.split(user_units)[0].strip())
            break
    # convert to mm
    if user_units is 'in':
        height *= 25.4
    if user_units is 'cm':
        height *= 10
    print(f'You entered a height of {height} mm')
    pixel_size = height/pixel_height
    print(f"Your pixels size is {pixel_size} mm/pixel")
    
    return pixel_size

def user_select_pca_box(frame: np.ndarray) -> np.ndarray:
    '''
    This function presents a maximized frame image and allows the user to select
    pixels from which to calculate the PCA transform
    should return 
    '''
    # --- Store ROI coordinates ---
    roi_coords = []

    # Allow a bounding box
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        roi_coords.append((min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        plt.close()
    # Display the image
    fig, ax = plt.subplots()
    manager = plt.get_current_fig_manager()
    try:
        # For most backends like TkAgg
        manager.full_screen_toggle()
    except AttributeError:
        # For Qt or other GUI backends, use this fallback
        manager.window.showMaximized()
    ax.imshow(frame)

    plt.title("Select ROI with mouse")
    rs = RectangleSelector(ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True)
    plt.show()
    return roi_coords

def build_pca_transform(frame: np.ndarray) -> np.ndarray:
    '''
    allows user to select a regiom for PCA analysis
    returns PCA transform and threshold for meniscus id
    '''
   
    roi_coords = user_select_pca_box(frame)
    x, y, w, h = roi_coords[0]
    roi_pixels = frame[y:y+h, x:x+w,:].reshape(-1, 3)  # reshape to (n_pixels, 3)

    # --- Perform PCA ---
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(roi_pixels)  # shape: (n_pixels, 3)
    
    # --- Plot PC1 vs PC2 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], s=2, alpha=0.5, c=roi_pixels / 255.0)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Scatter Plot of Selected Region")
    plt.grid(True)
    plt.show()
    return pcs

def show_edge_filter(frame: np.ndarray):
    # Convert to grayscale (luminance approximation)
    sobel_y = np.array([
        [ 2,  1,  0, -1, -2],
        [ 3,  2,  0, -2, -3],
        [ 4,  3,  0, -3, -4],
        [ 3,  2,  0, -2, -3],
        [ 2,  1,  0, -1, -2]
                           ]).transpose()

    # Apply Sobel in Y direction to each RGB channel
    sobel_r = ndimage.convolve(frame[:, :, 0], sobel_y)
    sobel_g = ndimage.convolve(frame[:, :, 1], sobel_y)
    sobel_b = ndimage.convolve(frame[:, :, 2], sobel_y)

    # Compute gradient magnitude across channels (Euclidean norm)
    sobel_color_mag = np.sqrt(sobel_r**2 + sobel_g**2 + sobel_b**2)

    # Normalize for display
    sobel_norm = (sobel_color_mag - np.min(sobel_color_mag)) / np.ptp(sobel_color_mag)

    # Show result
    plt.imshow(sobel_norm, cmap='gray')
    # Display result
    plt.title("Vertical Sobel Filter (Gy) - Highlights Meniscus")
    plt.axis('off')
    plt.show()
    return 0

if __name__ == '__main__':
    # Set video path
    video_file_path = 'data/wave_movie_1.mp4'
    # say what we're trying to do
    print(f"Attempting to load a video at {video_file_path}")
    # load video for pixel size
    video = load_camera_video(video_file_path,max_frames=1,convert_to_rgb=True)
    # if video is successfully loaded, size should be larger than 0
    print(f"Loaded video with {video.shape[0]} frames")
    # we need to figure out our pixel size
    # pixel_size = determine_pixel_size(video[0,:,:,:])
    # pca_transform = build_pca_transform(video[0,:,:,:])
    show_edge_filter(video[0,:,:,:])
