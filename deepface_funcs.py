from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
import csv

# C:/Users/harsh/Work/face_recognition/data
# DeepFace.stream(db_path = "C:/Users/harsh/Work/face_recognition/data")


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

"""
image1 = "C:/Users/harsh/Work/face_recognition/data/faces/vk/vk1.jpg"
image2 = "C:/Users/harsh/Work/face_recognition/data/images/vka1.jpg"

#face verification
result = DeepFace.verify(
  img1_path = image1,
  img2_path = image2,
  detector_backend = backends[3],
  distance_metric = metrics[0],
  model_name = models[1],
  threshold = 0.7
)

print(result)
image1 = mpimg.imread(image1)
image2 = mpimg.imread(image2)

facial_areas = result["facial_areas"]

#Create subplots to display the images side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#Plot the first image with bounding box
axs[0].imshow(image1)
rect1 = patches.Rectangle(
    (facial_areas['img1']['x'], facial_areas['img1']['y']),
    facial_areas['img1']['w'],
    facial_areas['img1']['h'],
    linewidth=2,
    edgecolor='r',
    facecolor='none'
)
axs[0].add_patch(rect1)
axs[0].set_title('image1')
axs[0].axis('off')

#Plot the second image with bounding box
axs[1].imshow(image2)
rect2 = patches.Rectangle(
    (facial_areas['img2']['x'],
     facial_areas['img2']['y']),
    facial_areas['img2']['w'],
    facial_areas['img2']['h'],
    linewidth=2,
    edgecolor='r',
    facecolor='none'
)
axs[1].add_patch(rect2)
axs[1].set_title('image2')
axs[1].axis('off')

plt.show()


"""

#face recognition from a directory
img1 = "C:/Users/harsh/Work/face_recognition/data/faces/perry/perry.jpg"
dfs = DeepFace.find(
  img_path = img1,
  db_path = "C:/Users/harsh/Work/face_recognition/data/images",
  detector_backend = backends[3],
  distance_metric = metrics[0],
  model_name = models[1],
  threshold = 0.6
)
print(dfs[0].head())

output_file = "C:\\Users\\harsh\\Work\\face_recognition\\face_matchings.csv"
dfs[0]['face_path'] = img1
dfs[0].to_csv(output_file, index = True, na_rep = 'nothing')


"""
#embeddings
embedding_objs = DeepFace.represent(
  img_path = "C:/Users/harsh/Work/face_recognition/data/faces/prabhas/prabhs_face.jpg",
  model_name = models[1],
)


# #facial analysis
# demographies = DeepFace.analyze(
#   img_path = "img4.jpg",
#   detector_backend = backends[3]
# )

#face detection and alignment
face_objs = DeepFace.extract_faces(
  img_path = "C:/Users/harsh/Work/face_recognition/data/images/vka1.jpg",
  detector_backend = backends[3]
)


img_face = DeepFace.detectFace(
  img_path = "C:/Users/harsh/Work/face_recognition/data/images/vka1.jpg",
  detector_backend = backends[3])

"""
