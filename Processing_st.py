import streamlit as st
import cv2
import numpy as np
import math
import os
import time
from tqdm import tqdm
from tensorflow.keras.models import load_model
import csv
import json
import base64
import pandas as pd
def ellipse_circumference(a, b):
    """
    Calculate the circumference of an ellipse.

    Parameters:
    - a (float): Semi-major axis length
    - b (float): Semi-minor axis length

    Returns:
    - float: Circumference of the ellipse
    """
    circumference = math.pi * math.sqrt(2 * (a**2 + b**2))
    return circumference

def fit_rotated_ellipse_ransac(data,iter=50,sample_num=10,offset=80.0):

    count_max = 0
    effective_sample = None

    for i in range(iter):
        sample = np.random.choice(len(data), sample_num, replace=False)

        xs = data[sample][:,0].reshape(-1,1)
        ys = data[sample][:,1].reshape(-1,1)

        J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float64))) )
        Y = np.mat(-1*xs**2)
        P= (J.T * J).I * J.T * Y

        # fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
        a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
        ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

        # threshold 
        ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])

        if(len(ran_sample) > count_max):
            count_max = len(ran_sample) 
            effective_sample = ran_sample

    return fit_rotated_ellipse(effective_sample)


def fit_rotated_ellipse(data):

    xs = data[:,0].reshape(-1,1) 
    ys = data[:,1].reshape(-1,1)

    J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float64))) )
    Y = np.mat(-1*xs**2)
    P= (J.T * J).I * J.T * Y

    a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
    theta = 0.5* np.arctan(b/(a-c))  
    
    cx = (2*c*d - b*e)/(b**2-4*a*c)
    cy = (2*a*e - b*d)/(b**2-4*a*c)

    cu = a*cx**2 + b*cx*cy + c*cy**2 -f
    w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
    h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

    ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

    error_sum = np.sum([ellipse_model(x,y) for x,y in data])
    print('fitting error = %.3f' % (error_sum))

    return (cx,cy,w,h,theta)

def apply_blur(image, blur_amount=5):
    return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

def fitPupil(image,circ_thresh=0.5,thresh_val=60,kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),min_thresh=170,max_thresh=280):
        #print(image)
        
        temp_image = image.copy()
       
        inf_img = temp_image.copy()
        
        image_gray = cv2.cvtColor(temp_image , cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_gray,(3,3),0)
        
        ret,thresh1 = cv2.threshold(blur,thresh_val,255,cv2.THRESH_BINARY)
        
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        
        #size_el = 0
        temp_image  = 255 - closing
        
        
        contours, hierarchy = cv2.findContours(temp_image , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hull = []
        for i in range(len(contours)):
          hull.append(cv2.convexHull(contours[i], False)) 
        
        cx,cy,w,h,theta = 0.0,0.0,0.0,0.0,0.0
        for con in hull:
            
            approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
            area = cv2.contourArea(con)
            perimeter = cv2.arcLength(con, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))


            if circularity > circ_thresh:
                
            
        
        
                if(len(approx) > 10 and area > 70):
                        try:
                        
                            cx,cy,w,h,theta = fit_rotated_ellipse_ransac(con.reshape(-1,2))
                            #size_el = ellipse_circumference(w,h)

                            # if size_el > min_thresh and size_el < max_thresh:
                            
                            #xcoordinates.append(cx)
                            #ycoordinates.append(cy)
                            cv2.ellipse(inf_img,(int(cx),int(cy)),(int(w),int(h)),theta*180.0/np.pi,0.0,360.0,(0,255,255),1)
                            inf_img = cv2.drawMarker(inf_img, (int(cx),int(cy)),(0, 255, 255),cv2.MARKER_CROSS,2,1)
                            #else: cx,cy,w,h,theta = 0.0,0.0,0.0,0.0,0.0
                            
                        except Exception as e: pass
            
        return inf_img,[cx,cy,w,h,theta]
    
            
     

# Function to process frames
def process_frame(frame, processing_step1, processing_step2):
    # Your processing logic using OpenCV
    # Example: Apply GaussianBlur and Canny edge detection
    out,cnt = fitPupil(frame,circ_thresh=processing_step1,thresh_val=processing_step2)
    
    return out,cnt
temp_file_to_save = './temp_file_1.mp4'
temp_csv_file = "./temp_file.csv"
temp_file_result  = './temp_file_2.mp4'
# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())
def calculate_deviation(values, num_frames):
    deviations = []
    previous_values = []

    for i, value in enumerate(values):
        if i == 0:
            previous_values.append(value)
            deviations.append(0)
        elif i < num_frames:
            if value == (0, 0):
                deviations.append(0)
            else:
                deviation = ((value[0] - previous_values[-1][0])**2 + (value[1] - previous_values[-1][1])**2)**0.5
                deviations.append(deviation)
            previous_values.append(value)
        else:
            if value == (0, 0):
                deviations.append(0)
            else:
                deviation_sum = 0
                for j in range(1, num_frames + 1):
                    deviation_sum += ((value[0] - previous_values[-j][0])**2 + (value[1] - previous_values[-j][1])**2)**0.5
                deviation = deviation_sum / num_frames
                deviations.append(deviation)
            previous_values.append(value)
            previous_values.pop(0)

    return deviations
def main():
    st.title("Video Processing with Streamlit and OpenCV")

    coll1, coll2 = st.columns(2)

    cnts = []

    # Upload CSV files
    with coll1:
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])

    with coll2:
        csv_file = st.file_uploader("Upload CSV",type=["csv"])

    # Upload video file
    
    

    if video_file is not None and csv_file is not None:
        
        # Read the video using OpenCV
        write_bytesio_to_file(temp_file_to_save, video_file)
        write_bytesio_to_file(temp_csv_file, csv_file)
        video_capture = cv2.VideoCapture(temp_file_to_save)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame selection slider
        selected_frame = st.sidebar.slider("Select a frame", 1, total_frames, 1)

        # Processing sliders
        processing_step1 = st.sidebar.slider("Circularity Threshold", 0.0, 1.0, 0.5, 0.01)
        processing_step2 = st.sidebar.slider("Binary Threshold", 1, 255, 40,1)

     
        # Process the selected frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frame - 1)
        ret, frame = video_capture.read()

     
        
        

        st.success("Video loaded and frames processed!")
        st.header("Frames")
        col1, col2 = st.columns(2)

        with col1:
            st.image(frame, caption="Unprocessed", use_column_width=True)

        with col2:
            processed_frame,cnt = process_frame(frame, processing_step1, processing_step2)
            st.image(processed_frame, caption="Processed", use_column_width=True)
        
        #min_max_values= st.select_slider("Select a minimum-maximum Threshold",options=range(0,1000),value=(0,50))
        min_max_values = (0,0)
        
        if cnt:
            data = {"Ellipse Size":(cnt[2],cnt[3]),"Ellipse Position":(cnt[0],cnt[1])}


            df = pd.DataFrame(data,index=(["x","y"]))

      
         
            st.table(df)
            
            st.metric("Ellipse Circumference",round(ellipse_circumference(cnt[2],cnt[3]),2))

        
        if st.button("Apply"):
            # Navigate to the results page
            st.empty()
            video_capture.release()
           
            results_page(min_max_values,temp_file_to_save,processing_step1,processing_step2,temp_file_to_save,csv_file)
        
def display_video(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
def get_number_of_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    # Iterate through the frames to count them
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

    # Release the video file
    cap.release()
def process_csv(csv_file):
    try:
        heartbeat_values = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                heartbeat_values.append(int(row['Heart Rate (BPM)']))
        return heartbeat_values
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while processing {csv_file}: {e}")
        return None
def compute_summary(data,nof=0,csv_file=None, save=False,output_path="./"):
    num_dicts = len(data)
    if num_dicts == 0:
        return {"number of frames": nof, "number of frames per second": 0, "ellipse average": None}

    # Calculate number of frames
    num_frames = num_dicts
    # Assuming frames are sequential and timestamps are in seconds, calculate frames per second
    fps = num_frames / data[-1]['timestamp']
    
    # Initialize variables to hold sums of coordinates
    sum_x1, sum_y1, sum_x2, sum_y2, sum_radius = 0, 0, 0, 0, 0

    for entry in data:
        ellipse = entry['ellipse']
        # Check if the ellipse is a tuple
        if isinstance(ellipse, tuple):
            x1, y1 = ellipse[0]
            x2, y2 = ellipse[1]
            radius = ellipse[2]
            sum_x1 += x1
            sum_y1 += y1
            sum_x2 += x2
            sum_y2 += y2
            sum_radius += radius

    # Calculate average coordinates and radius
    average_x1 = sum_x1 / num_dicts
    average_y1 = sum_y1 / num_dicts
    average_x2 = sum_x2 / num_dicts
    average_y2 = sum_y2 / num_dicts
    average_radius = sum_radius / num_dicts

    average_ellipse = ((average_x1, average_y1), (average_x2, average_y2), average_radius)
    summary = {"number of frames": nof, "number of frames per second": fps, "ellipse average": average_ellipse}

    average_heartbeat = None
    if csv_file:
        heartbeat_values = process_csv(csv_file)
        if heartbeat_values:
            average_heartbeat = sum(heartbeat_values) / len(heartbeat_values)

    summary = {"number of frames": nof, "number of frames per second": fps,
               "ellipse average": average_ellipse, "average heartbeat": average_heartbeat,
               "blinks": data[-1]["blinks"]
               }

    if save:
        with open(output_path+"\summary.json", "w") as f:
            json.dump(summary, f)

    return output_path+"\summary.json"
def calculate_average_deviation(frame_centers, block_size):
    average_deviations_x = []
    average_deviations_y = []
    num_frames = len(frame_centers)
    
    for i in range(0, num_frames - block_size + 1, block_size):
        block_centers = frame_centers[i:i+block_size]
        deviations_x = []
        deviations_y = []
        
        for j in range(len(block_centers) - 1):
            x_deviation = abs(block_centers[j+1][0] - block_centers[j][0])
            y_deviation = abs(block_centers[j+1][1] - block_centers[j][1])
            deviations_x.append(x_deviation)
            deviations_y.append(y_deviation)
        
        average_deviation_x = sum(deviations_x) / len(deviations_x)
        average_deviation_y = sum(deviations_y) / len(deviations_y)
        
        average_deviations_x.append(average_deviation_x)
        average_deviations_y.append(average_deviation_y)
    
    return average_deviations_x, average_deviations_y

def results_page(min_max_values,video,circ_thresh,bin_thresh,video_file,csv_file):
    st.title("Pupil and Parameter Caption")

    model_path = "blink_detection_model_100Epochs.h5"
    model = load_model(model_path)
    # write_bytesio_to_file(temp_file_to_save, video_file)
    
    cap = cv2.VideoCapture(temp_file_to_save)
    blinks = 0
    frame_counter = 0
    consecutive_closed_frames = 0
    blink_threshold = 6
    is_blinking = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_vid = cv2.VideoWriter(temp_file_result, fourcc, fps, (width, height))

    progress_text = f"Infering using {model_path} with parameters: {circ_thresh,bin_thresh,min_max_values[0],min_max_values[1]}"
    my_bar = st.progress(0, text=progress_text)
    data_file_path = "processed_data.json"

    frame_metadata_list = []
    cnts = []
    f"FPS: {int(fps)}, Dim: {width}x{height}, Total Frames: {total_frames}"
    ret, frame = cap.read()
    
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        
        mode = 'w' if frame_counter == 0 else 'a'
        if not ret:
            print("Error Reading Frame")
            break

        # Preprocess the frame
        preprocessed_frame = cv2.resize(frame, (224, 224))  # Resize to match the model input size
        preprocessed_frame = preprocessed_frame / 255.0  # Normalize pixel values to [0, 1]

        # Make predictions using the model
        prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0),verbose=None)[0][0]
       
        frame,cnt = fitPupil(frame,circ_thresh=circ_thresh,thresh_val=bin_thresh)
        
        # Adjust the threshold based on your model and dataset
        threshold = 0.5

        frame_counter += 1
        cnts.append((cnt[0],cnt[1]))

        # Display the result
        if prediction > threshold:
            consecutive_closed_frames = 0
            cv2.putText(frame, f'Open - {blinks}', (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            consecutive_closed_frames += 1 
            
            cv2.putText(frame, f'Closed - {blinks}', (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if consecutive_closed_frames >= blink_threshold and not is_blinking:
            cv2.putText(frame, 'Blink Detected', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            blinks+=1
            is_blinking = True
            # Reset the frame counter after a blink is detected
            frame_counter = 0
        if is_blinking and consecutive_closed_frames == 0:
            is_blinking = False
        
        #print(cnt)
        if cnt:
            data = {"Frame#":frame_counter,"pupilDetected": True,"Ellipse_Size":cnt[1],"Ellipse_Position_center":cnt[0],"Ellipse_Angle":cnt[2],"Blinks":blinks}
        else:
            data = {"Frame#":frame_counter,"pupilDetected": False,"Ellipse_Size":0,"Ellipse_Position_center":0,"Ellipse_Angle":0,"Blinks":blinks}
        
        frame_metadata = {
            'frame_number': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC),
            'ellipse': cnt,
            "blinks": blinks
        }
        frame_metadata_list.append(frame_metadata)
        # with open(data_file_path, mode, newline='') as csvfile:
        # # Specify the field names (header) for the CSV file
        #     fieldnames = data.keys()

        #     # Create a CSV writer object
        #     csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #     # Write the header to the CSV file only for the first iteration
        #     if frame_counter == 0:
        #         csv_writer.writeheader()

        #     # Write the data (dictionary) to the CSV file
        #     csv_writer.writerow(data)
        out_vid.write(frame)
        my_bar.progress(int((frame_num/total_frames)*100), text=progress_text)
    #cnts
    df = pd.DataFrame({'ellipse [x , y]':cnts})
    st.table(df)
    cadx,cady = calculate_average_deviation(cnts,10)
    # df = pd.DataFrame({'X': cadx, 'Y': cady})

    # st.table(df)
    out_vid.release()
    file_path = "full_data.json"
    with open(file_path, "w") as json_file:
        json.dump(frame_metadata_list, json_file, indent=4)
    json_temp = compute_summary(frame_metadata_list,total_frames,temp_csv_file,save="True")
    json_file = open(json_temp, 'rb')
    json_bytes = json_file.read()
    json_file.close()
    json_b64 = base64.b64encode(json_bytes).decode()
    csv_link = f'<a href="data:application/json;base64,{json_b64}" download="processed_data.json">Download CSV</a>'
    st.markdown(csv_link, unsafe_allow_html=True)
    # Download button for processed video
    st.markdown(
        f"Download processed video",
        unsafe_allow_html=True,
    )
    try:
        video_file = open(temp_file_result, 'rb')
        video_bytes = video_file.read()
        video_file.close()
        video_b64 = base64.b64encode(video_bytes).decode()
        video_link = f'<a href="data:video/mp4;base64,{video_b64}" download="temp_file_2.mp4">Download Video</a>'
        st.markdown(video_link, unsafe_allow_html=True)
    except: "Error"
    if st.button("Go Back"):
        st.empty()
        os.remove(data_file_path)
        os.remove(temp_file_result)
        os.remove(temp_file_to_save)
        main()

    # Add your logic for displaying the results here
if __name__ == "__main__":

    main()
   