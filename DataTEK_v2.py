import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
#For RPi Modify this line 
import tensorflow as tf
from sys import getsizeof
import cv2
import keyboard
from time import sleep


chlist = {
	0: "&TV", #
	1: "Animal Planet",#385
	2: "Colors",#3
	3: "Discovery Channel",#381
	4: "History TV",#382
	5: "Hungama",#365
	6: "Nick",#61
	7: "Pogo",#363
	8: "Sonic",#367
	9: "Sony SAB",#5
	10: "Sony PAL",#10
	11: "Sony YAY",#366
	12: "Star Sports Hindi 1",#304
	13: "TLC",#382
	14: "UTV Movies",#59
	15: "Star Bharat",#6
	16: "UTV Action" #56
}


IMG_WIDTH = 100
IMG_HEIGHT = 100

TF_LITE_MODEL_FILE_NAME = "model_av.tflite"

#for RPi Modify this line
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

def predict_ch(interpreter, image):
	resizedimg = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
	cropped_frame = image[1:150, 440:640]
	resized_cropped_frame = cv2.resize(cropped_frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
	resized_cropped_frame = resized_cropped_frame.astype(np.float32)
	resized_cropped_frame = resized_cropped_frame.reshape((1,100,100,3))

	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]['index'], resized_cropped_frame)
	interpreter.invoke()
	tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
	return tflite_model_predictions



# frame = cv2.imread("2_1.jpg", cv2.IMREAD_COLOR)

# print(predict_ch(interpreter,frame))

cap = cv2.VideoCapture(1)

# ret,frame = cap.read()
# plt.imshow(frame/255)
# plt.show()

i=1
while True:
		ret,frame = cap.read()
		cv2.imwrite(str(i)+ "frame.jpg",frame)
		prediction = predict_ch(interpreter, frame)
		predicted_class = prediction.argmax()
		#print(predicted_class)
		if(prediction[0][predicted_class] > 0.6):
			print(chlist[predicted_class])
			print(prediction[0][predicted_class])
		else: 
			print("Unknown")
		i=i+1
		sleep(0.5)






