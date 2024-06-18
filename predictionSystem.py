import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import serial
import time
# Load the tflite model and the labels file
model_path ="cnnmodel'.tflite"
port = serial.Serial("/dev/ttyS0",baudrate=115200,timeout=1)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
labels = ["mud", "soil"]

# Define the video capture object
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256) # set the img width to 256
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256) # set the img height to 256
cap.set(cv2.CAP_PROP_FPS, 10)
# Define the font and the color for the text
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)

#function to calculate time based on date time running on raspberry pi 4
def print_time_with_milliseconds():
    while True:
        # Get the current time in seconds
        current_time_seconds = time.time()
        # Extract milliseconds from the seconds value
        milliseconds = int(current_time_seconds * 1000) % 1000

        # Format the time string, including milliseconds
        formatted_time = time.strftime("%H:%M:%S") + "." + f"{milliseconds:03d}"

        # Print the formatted time, overwriting previous output
        print(formatted_time, end="\n")

        # Pause briefly for a near-real-time update
        time.sleep(0.001)
        break  
# Loop until the user presses 'q' key
while True:
    start_time = time.time()
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize and normalize the frame
    resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    input_data = np.float32(input_data)
	
    # Run inference on the frame
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data[0])
    result = str(prediction)
    port.write(result.encode())

    # Get the label and the confidence of the prediction
    label = labels[prediction]
    confidence = output_data[0][prediction]
    
    print_time_with_milliseconds()
    print("prediction: ",' ',prediction,"label: ",labels[prediction], "confidence", confidence)

    # Put the label and the confidence on the frame
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), font, 1, color, 2)

    # Show the frame
    cv2.imshow('Image Classification', frame)
    
    time_comp = time.time()
    elp_time = time_comp - start_time
    #calculate time from system reading a single frame of video until sending data to esp32
    print("Waktu Komputasi: ", elp_time, "detik") 
    #press q to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
