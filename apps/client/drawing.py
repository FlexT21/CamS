from mediapipe.python.solutions import drawing_styles, drawing_utils, face_mesh


def draw_face_mesh(image, results) -> None:
    for face_landmarks in results.multi_face_landmarks:
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
