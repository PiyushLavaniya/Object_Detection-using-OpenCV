import cv2
import streamlit as st
import os

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter objects for both outputs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_contours_edges = cv2.VideoWriter('contours_edges.mp4', fourcc, fps, (350, 450))
    output_video_contours_binary = cv2.VideoWriter('contours_binary.mp4', fourcc, fps, (350, 450))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If there are no more frames to read, break the loop
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[300:300+450, 70:70+350]

        # Convert the frame to grayscale
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(blurred, 20, 150)

        # Thresholding
        _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

        # Find contours on edges
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_edges = cropped_frame.copy()
        for cnt in contours_edges:
            cv2.drawContours(frame_edges, [cnt], -1, (0, 255, 0), 2)
        output_video_contours_edges.write(frame_edges)

        # Find contours on binary mask
        contours_binary, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_binary = cropped_frame.copy()
        for cnt in contours_binary:
            cv2.drawContours(frame_binary, [cnt], -1, (0, 255, 0), 2)
        output_video_contours_binary.write(frame_binary)

    # Release resources
    cap.release()
    output_video_contours_edges.release()
    output_video_contours_binary.release()

def convert_to_h264(input_video, output_video):
    if not os.path.exists(output_video):
        command = f"ffmpeg -i {input_video} -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 192k {output_video}"
        os.system(command)
    else:
        st.info("File already exists. Skipped conversion.")

def main():
    st.title('Video Contour Detection')

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        video_path = "video.mp4"  # Temporary file path to save the uploaded video
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Process the uploaded video
        process_video(video_path)

        # Convert the output videos to H.264
        convert_to_h264('contours_edges.mp4', 'contours_edges_h264.mp4')
        convert_to_h264('contours_binary.mp4', 'contours_binary_h264.mp4')
        
        col1, col2 = st.columns(2)
        # Display the processed videos side by side
        with col1:
            st.write("Detections using the edges")
            st.video("contours_edges_h264.mp4", start_time=0)
        with col2:
            st.write("Detections using the Binary Mask")
            st.video("contours_binary_h264.mp4", start_time=0)
       

if __name__ == '__main__':
    main()
