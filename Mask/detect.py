import cv2,imutils
import tensorflow as tf
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from tensorflow import Graph
from keras.preprocessing.image import image
tf_graph = Graph()
with tf_graph.as_default():
	tf_session = tf.compat.v1.Session()
	with tf_session.as_default():
		model = load_model("./weights/maskE5.h5")
		print("model loaded")

cascade = cv2.CascadeClassifier('./haarcascade_files/haarcascade_frontalface_default.xml')

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
	def __del__(self):
		self.video.release()
	def get_frame(self):
		_,frame = self.video.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces =  cascade.detectMultiScale(gray,1.3,5)
		img = cv2.resize(frame,(256,256))
		img4 = img.reshape(1,256,256,3)
		with tf_graph.as_default():
			with tf_session.as_default():
				pred = model.predict(img4)
		
		if pred[0][0] ==0:
			cv2.putText(frame,"Mask Detected",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
		if pred[0][0] ==1:
			cv2.putText(frame,"Mask Not Detected",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
			

				
		
			
		__,jpg = cv2.imencode('.jpg',frame)
		return jpg.tobytes()


