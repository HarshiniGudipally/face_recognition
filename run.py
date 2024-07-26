from deepface import DeepFace
import os
import pandas as pd
# !pip install --ignore-installed --no-cache-dir deepface


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

# faces_dir = "C:\\Users\\harsh\\Work\\face_recognition\\data\\faces"
faces_dir = "C:\\Users\\harsh\\Work\\face_recognition\\data\\faces"
def find_faces(img1,images_dir):
  # face recognition from a directory
  dfs = DeepFace.find(
    img_path=img1,
    db_path=images_dir,
    detector_backend=backends[3],
    distance_metric=metrics[0],
    model_name=models[1],
    threshold=0.6,
    enforce_detection = False
  )

  return dfs

# images_dir =  "C:\\Users\\harsh\\Work\\face_recognition\\data\\images"
images_dir = "C:\\Users\\harsh\\Work\\face_recognition\\data\\reception_resized"
# images_dir = "C:\\Users\\harsh\\Work\\face_recognition\\data\\images"
faces_folders = os.listdir(faces_dir)


# Create an empty DataFrame with the desired columns
columns = ['identity', 'hash', 'target_x', 'target_y', 'target_w', 'target_h',
           'source_x', 'source_y', 'source_w', 'source_h', 'threshold', 'distance']
dfs = pd.DataFrame(columns=columns)


for face_folder in faces_folders:
  face_folder_path = os.path.join(faces_dir, face_folder)
  print(face_folder_path)
  face_images = os.listdir(face_folder_path)
  for face in face_images:
    face_path = os.path.join(face_folder_path, face)
    print(face_path)
    img1 = face_path
    df_face = find_faces(img1,images_dir)
    df_face[0]['face_path'] = img1
    df_face[0]['face_name'] = face_folder
    dfs = pd.concat([dfs, df_face[0]], ignore_index=True)


output_file = "C:\\Users\\harsh\\Work\\face_recognition\\face_matchings.csv"

dfs.to_csv(output_file, index = True, na_rep = 'nothing')


