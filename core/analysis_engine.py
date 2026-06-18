import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from core.face_detector import FaceDetector

try:
    import mediapipe as mp
except ImportError:
    mp = None


class AnalysisEngine:
    def __init__(self, settings):
        self.settings = settings
        self.face_detector = FaceDetector()
        self.emotion_model = self._load_model(settings["models"]["emotion"])
        self.age_model = self._load_model(settings["models"]["age"])
        self.gender_model = self._load_model(settings["models"]["gender"])
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.genders = ["Male", "Female"]
        self.mp = mp
        self.face_mesh = self._initialize_face_mesh()

    def _initialize_face_mesh(self):
        if self.mp is None:
            return None
        return self.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _load_model(self, path):
        if not os.path.exists(path):
            return None
        return load_model(path)

    def _preprocess_face(self, face_image, size=(64, 64)):
        face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, size)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        return face

    def _preprocess_face_color(self, face_image, size=(128, 128)):
        face = cv2.resize(face_image, size)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        return face

    def analyze(self, frame):
        output = {
            "faces": [],
            "summary": {
                "emotion": "Unknown",
                "age": "N/A",
                "gender": "Unknown",
                "attention": "N/A",
                "stress": "N/A",
                "face_count": 0,
            },
        }

        faces = self.face_detector.detect_faces(frame)
        output["summary"]["face_count"] = len(faces)

        mesh_landmarks = self._get_face_mesh(frame)
        for (x, y, w, h) in faces:
            face_region = frame[y : y + h, x : x + w]
            emotion = self.predict_emotion(face_region)
            age = self.predict_age(face_region)
            gender = self.predict_gender(face_region)
            head_pose = self.estimate_head_pose(mesh_landmarks)
            attention = self.estimate_attention(face_region, head_pose)
            stress = self.estimate_stress(emotion)

            output["faces"].append(
                {
                    "box": (x, y, w, h),
                    "emotion": emotion,
                    "age": age,
                    "gender": gender,
                    "attention": attention,
                    "stress": stress,
                    "head_pose": head_pose,
                }
            )

        if len(output["faces"]) > 0:
            output["summary"]["emotion"] = output["faces"][0]["emotion"]
            output["summary"]["age"] = output["faces"][0]["age"]
            output["summary"]["gender"] = output["faces"][0]["gender"]
            output["summary"]["attention"] = output["faces"][0]["attention"]
            output["summary"]["stress"] = output["faces"][0]["stress"]
            output["summary"]["head_pose"] = output["faces"][0].get("head_pose", "Center")

        return output

    def predict_emotion(self, face_region):
        if self.emotion_model is None:
            return "Neutral"
        face = self._preprocess_face(face_region)
        results = self.emotion_model.predict(face)
        return self.emotions[int(np.argmax(results))]

    def predict_age(self, face_region):
        if self.age_model is None:
            return "25"
        face = self._preprocess_face_color(face_region)
        result = self.age_model.predict(face)
        return str(int(np.squeeze(result)))

    def predict_gender(self, face_region):
        if self.gender_model is None:
            return "Male"
        face = self._preprocess_face_color(face_region)
        results = self.gender_model.predict(face)
        return self.genders[int(np.argmax(results))]

    def _get_face_mesh(self, frame):
        if self.face_mesh is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def _compute_ear(self, landmarks, width, height):
        indices = {
            "left": [33, 160, 158, 133, 153, 144],
            "right": [362, 385, 387, 263, 373, 380],
        }

        def landmark_point(index):
            point = landmarks.landmark[index]
            return np.array([point.x * width, point.y * height])

        left = [landmark_point(i) for i in indices["left"]]
        right = [landmark_point(i) for i in indices["right"]]

        def ear(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            if C == 0:
                return 0.0
            return (A + B) / (2.0 * C)

        return (ear(left) + ear(right)) / 2.0

    def estimate_head_pose(self, landmarks):
        if landmarks is None:
            return "Center"
        nose = landmarks.landmark[1]
        if nose.x < 0.42:
            return "Left"
        if nose.x > 0.58:
            return "Right"
        return "Center"

    def estimate_attention(self, face_region, head_pose="Center"):
        if self.face_mesh is None:
            if head_pose != "Center":
                return "Distracted"
            return "Focused"

        landmarks = self._get_face_mesh(face_region)
        if landmarks is None:
            return "Focused" if head_pose == "Center" else "Distracted"

        ear = self._compute_ear(landmarks, face_region.shape[1], face_region.shape[0])
        if ear < 0.22:
            return "Drowsy"
        if head_pose != "Center":
            return "Distracted"
        return "Focused"

    def estimate_stress(self, emotion):
        if emotion in ["Angry", "Fear", "Disgust"]:
            return "High"
        if emotion == "Sad":
            return "Moderate"
        return "Low"
