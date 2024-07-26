# !pip install deepface

import pandas as pd
import os
import shutil

# Read entire columns from a CSV file
df = pd.read_csv("face_matchings.csv")
paths = df['identity'].tolist()

# print(paths)
# print(len(paths))
# print(len(set(paths)))

face_matches = df[['identity', 'face_name']].apply(tuple, axis=1)
final_face_matches = list(set(face_matches))

print(len(face_matches))
print(len(final_face_matches))
print(final_face_matches)

result_dir = "C:\\Users\\harsh\\Work\\face_recognition\\data\\results"

for match in final_face_matches:
    image_path = match[0]
    face_name = match[1]

    #check if results folder is present
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    face_match_folder_path = os.path.join(result_dir, face_name)

    if not os.path.exists(face_match_folder_path):
        os.mkdir(face_match_folder_path)

    shutil.copy(image_path, face_match_folder_path)

