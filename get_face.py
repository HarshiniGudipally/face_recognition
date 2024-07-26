from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2
import os

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'fastmtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]


metrics = ["cosine", "euclidean", "euclidean_l2"]


image1_path =  "C:/Users/harsh/Work/face_recognition/data/images/prabhas_g3.jpg"
image2 = "C:/Users/harsh/Work/face_recognition/data/images/vka1.jpg"

target_path = "C:/Users/harsh/Work/face_recognition/data/faces/vk/"


#face verification
result = DeepFace.verify(
  img1_path = image1_path,
  img2_path = image2,
  detector_backend = backends[3],
  distance_metric = metrics[0],
  model_name = models[1],
  threshold = 0.7
)

print(result)
image1 = cv2.imread(image1_path)
facial_areas = result["facial_areas"]
print(facial_areas)


for face in facial_areas:

  x, y, w, h = facial_areas[face]['x'], facial_areas[face]['y'], facial_areas[face]['w'], facial_areas[face]['h']

  cropped_face = image1[y:y+h, x:x+w]

  target_face_path = os.path.join(target_path, face+".jpg")
  cv2.imwrite(target_face_path, cropped_face)

