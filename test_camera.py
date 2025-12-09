import numpy as np
import tflite_runtime.interpreter as tflite

delegate = tflite.load_delegate("libethosu_delegate.so")
#import the model
interpreter = tflite.Interpreter(model_path="one_node_vela65.tflite", experimental_delegates=[delegate])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print("Model input shape:", input_details[0]['shape'])
output_details = interpreter.get_output_details()
print("Output shape: ", output_details[0]['shape'])

#import a video
import cv2
#cap = cv2.VideoCapture("horizontal-welding.mp4") #add a video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Error opening video file or camera")

#run predictions on it
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("frame", frame)
    #preprocessing
    frame = cv2.resize(frame, (224,224))
    frame = frame.astype(np.int8)
    frame = np.transpose(frame, (2, 0, 1))
    input_tensor = np.expand_dims(frame, axis=0)

    #set tensor
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_tensor)
    #run interpreter on the tensor
    interpreter.invoke()
    #get the output
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    print(output_data)
    #with open("output.txt", "a") as f:
     #   f.write(str(output_data) + "\n")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #break on q press?

cap.release()
cv2.destroyAllWindows()
