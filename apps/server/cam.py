import os.path

import cv2
import numpy as np
import mediapipe as mp

import face_recognition

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

known_users = {}

USER_ROOT = "users"

for person in os.listdir(USER_ROOT):
    if not os.path.isdir(f"{USER_ROOT}/{person}"):
        continue

    known_users[person] = []
    for img in os.listdir(f"{USER_ROOT}/{person}"):
        image = face_recognition.load_image_file(f"{USER_ROOT}/{person}/{img}")
        encoding = face_recognition.face_encodings(image)[0]
        known_users[person].append(encoding)

def calcular_distancias(encodings_guardados, encoding_actual):
    distancias = []
    for enc in encodings_guardados:
        d = np.sqrt(np.sum((np.array(enc) - np.array(encoding_actual)) ** 2))
        distancias.append(d)
    return distancias

def reconocer_usuario(encoding_actual):
    umbral = 1.5
    min_distance = float('inf')
    for usuario, encodings in known_users.items():
        distancias = calcular_distancias(encodings, encoding_actual)
        min_distance = min(distancias) if distancias else float('inf')
        if distancias and min(distancias) < umbral:
            return usuario, min(distancias)
    return "Desconocido", min_distance

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(     
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      # Get encodings for the detected faces
      encoding = face_recognition.face_encodings(results)

      if encoding:
        usuario, umbral = reconocer_usuario(encoding[0])
        cv2.putText(image, f"{usuario}-{umbral:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, bottomLeftOrigin=False)

      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
