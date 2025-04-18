import cv2
import os
import tqdm

# Function to crop a frame to a square using the smaller dimension
def crop_to_square(image, bbox, expected_size):
    h, w = image.shape[:2]
    x, y, bw, bh = map(int, bbox)

    # Determine the smallest dimension
    min_dim = min(w, h)

    # Calculate cropping region
    center_x, center_y = x + bw // 2, y + bh // 2
    half_dim = min_dim // 2

    # Ensure cropping does not go out of bounds
    x1 = max(0, center_x - half_dim)
    y1 = max(0, center_y - half_dim)
    x2 = min(w, center_x + half_dim)
    y2 = min(h, center_y + half_dim)

    # Crop the image
    cropped_frame = image[y1:y2, x1:x2]

    # Resize to ensure consistent frame size
    if cropped_frame.shape[:2] != expected_size:
        cropped_frame = cv2.resize(cropped_frame, expected_size)

    return cropped_frame

# Specify the folder containing the videos
input_folder = r"C:\Users\ADMIN\Pictures\Camera Roll\ANP bichitos 2.0\ANP bichitos 2.0\Amoeba radiosa"
output_folder = input_folder

# Get a list of all video files in the folder
video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Loop through each video file in the folder
for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    
    # Check if the video has already been processed
    if video_file.startswith("processed_"):
        print(f"Skipping already processed video: {video_file}")
        continue  # Skip this video and move to the next one
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    print(f"Video open: {video_path}")

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        continue

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_dim = min(frame_width, frame_height)  

    # Define the codec and create VideoWriter object
    output_video_path = os.path.join(output_folder, f"processed_{video_file}")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Change this if MJPG doesn't work (e.g., try 'DIVX' or 'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (min_dim, min_dim))

    # Initialize the tracker
    tracker = cv2.TrackerCSRT_create()

    # Frame selection improvements
    selected_frame = 0  # Start at frame 0
    num_input = ""

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read video at frame {selected_frame}.")
            break

        # Display the frame number
        frame_display = frame.copy()
        cv2.putText(frame_display, f"Frame: {selected_frame}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Select Frame (Arrow Keys to Navigate, Enter to Confirm, Type Frame Number)", frame_display)
        
        key = cv2.waitKey(0) & 0xFF

        if key == ord('\r') or key == ord('\n'):  # Enter key to confirm selection
            break
        elif key == ord('q'):  # Quit
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == 81 or key == ord('a'):  # Left arrow or "A" key - go back one frame
            selected_frame = max(0, selected_frame - 1)
        elif key == 83 or key == ord('d'):  # Right arrow or "D" key - go forward one frame
            selected_frame = min(total_frames - 1, selected_frame + 1)
        elif 48 <= key <= 57:  # Number keys (0-9)
            num_input += chr(key)  # Append the number to the input string
            print(f"Frame number input: {num_input}")
        elif key == 8:  # Backspace to delete last digit
            num_input = num_input[:-1]
        elif key == ord('\r') or key == ord('\n'):  # Enter key
            if num_input.isdigit():
                selected_frame = min(total_frames - 1, int(num_input))
                num_input = ""

    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read selected frame: {selected_frame}")
        cap.release()
        continue

    # Select the region to track
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    tracker.init(frame, bbox)

    # Initialize progress bar
    progress = tqdm.tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame")

    frames_written = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        ret, bbox = tracker.update(frame)

        if ret:
            # Crop and resize frame
            square_frame = crop_to_square(frame, bbox, (min_dim, min_dim))

            # Ensure frame is valid
            if square_frame.shape[0] > 0 and square_frame.shape[1] > 0:
                out.write(square_frame)  
                frames_written += 1  

            # Display tracking
            display_frame = frame.copy()
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(display_frame, p1, p2, (0, 255, 0), 2)

            cv2.imshow("Tracking", display_frame)

        else:
            print("Tracking lost - skipping frame")

        progress.update(1)

        # Quit tracking
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    progress.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Check if frames were written
    if frames_written == 0:
        print(f"ERROR: No frames were written! Video {video_file} may be empty.")
        os.remove(output_video_path)

    # Check processed video validity
    cap_test = cv2.VideoCapture(output_video_path)
    if not cap_test.isOpened():
        print(f"ERROR: Processed video {output_video_path} could not be opened!")
    else:
        print(f"Processed video saved to: {output_video_path}")
        cap_test.release()
