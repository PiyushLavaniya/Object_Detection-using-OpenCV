import cv2
import streamlit as st
import numpy as np
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

        # Find contours on binary mask
        contours_binary, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_binary = cropped_frame.copy()
        for cnt in contours_binary:
            cv2.drawContours(frame_binary, [cnt], -1, (0, 255, 0), 2)
        
        # Count the number of contours detected
        count_edges = len(contours_edges)
        count_binary = len(contours_binary)

        # Write a message on the frame if contours are more than 8
        if count_edges > 4:
            cv2.putText(frame_edges, "White stripe exists in Centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame_edges, "White stripe does not exist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if count_binary > 4:
            cv2.putText(frame_binary, "White stripe exists in Centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame_binary, "White stripe does not exist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        output_video_contours_edges.write(frame_edges)
        output_video_contours_binary.write(frame_binary)
        
    # Release resources
    cap.release()
    output_video_contours_edges.release()
    output_video_contours_binary.release()
            

##Converting the Video to an encoding that is supported in Streamlit            
def convert_to_h264(input_video, output_video):
    if not os.path.exists(output_video):
        command = f"ffmpeg -i {input_video} -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 192k {output_video}"
        os.system(command)
    else:
        st.info("File already exists. Skipped conversion.")



def main():
    st.title('Centre Stripe Detection App')
    
    st.sidebar.image("logo.png", use_column_width = "always")
    
    options = st.sidebar.selectbox("Select Image or Video", options = ["Image", "Video"])
    
    if options == 'Image':
        st.sidebar.header('Threshold Settings')

        # Upload image
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        # Default threshold values
        low_threshold = st.sidebar.slider('Low Threshold', 0, 255, 20)
        high_threshold = st.sidebar.slider('High Threshold', 0, 255, 150)

        if uploaded_file is not None:
            # Read uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            cropped_image = image[300:300+450, 70:70+350]

            # Convert to RGB format for display in Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detection
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the image
            image_with_contours = cropped_image.copy()
            for cnt in contours:
                cv2.drawContours(image_with_contours, [cnt], -1, (0, 255, 0), 2)
                
            num_contours = len(contours)
            if num_contours > 6:
                st.write("White Stripe detected in Centre, passed.")
            else:
                st.write("White Stripe could not be detected, Failed.")

            # Convert to RGB format for display in Streamlit
            image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)

            # Display images side by side
            col1, col2 = st.columns(2)
            col1.image(cropped_image_rgb, caption='Cropped Image', use_column_width=True)
            col2.image(image_with_contours_rgb, caption='Image with Contours', use_column_width=True)
    
    
    
    elif options == "Video":
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
            
            st.info("Videos can have some problems rendering, so for some issue if the video is not able to be rendered, then no need to worry because the Video with Detections will still be saved into your Directory, so you can watch the Video from there as well.")
                    

if __name__ == "__main__":
    main()
