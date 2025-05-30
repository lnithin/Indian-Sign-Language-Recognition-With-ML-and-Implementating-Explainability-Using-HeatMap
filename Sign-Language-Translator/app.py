import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import customtkinter as ctk
import cv2
import csv
import itertools
import copy
import numpy as np
from collections import deque, Counter
from PIL import Image
import mediapipe as mp
from model import KeyPointClassifier

BUFFER_LENGTH = 15
CONFIDENCE_THRESHOLD = 0.85
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.7
HEATMAP_RADIUS = 25
HEATMAP_OPACITY = 0.6
OFFSET = 20
WHITE_BG_SIZE = (400, 400)


def calc_landmark_list(image, landmarks):
    if not landmarks:
        return []
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [min(int(landmark.x * image_width), image_width - 1),
         min(int(landmark.y * image_height), image_height - 1)]
        for landmark in landmarks.landmark
    ]

def pre_process_landmark(landmark_list):
    if not landmark_list:
        return []
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for point in temp_landmark_list:
        point[0] -= base_x
        point[1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list)) or 1
    return [x / max_value for x in temp_landmark_list]

def draw_heatmap(frame, landmarks):
    if not landmarks:
        return frame
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for point in landmarks:
        x, y = point
        if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
            cv2.circle(heatmap, (x, y), HEATMAP_RADIUS, 255, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - HEATMAP_OPACITY, heatmap, HEATMAP_OPACITY, 0)

keypoint_classifier = KeyPointClassifier()
with open('D:\\new sign\\Sign-Language-Translator\\model\\keypoint_classifier\\label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.setup_ui()
        self.setup_camera()
        self.setup_variables()

    def setup_ui(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.window.geometry('1200x800')
        self.window.title("Two-Hand Sign Language Detection")

        title_font = ctk.CTkFont(family='Consolas', weight='bold', size=28)
        letter_font = ctk.CTkFont(family='Consolas', weight='bold', size=100)
        text_font = ctk.CTkFont(family='Consolas', size=24)

        self.header = ctk.CTkLabel(self.window, text='TWO-HAND SIGN DETECTION', fg_color='steelblue', text_color='white', height=50, font=title_font, corner_radius=8)
        self.header.pack(side=ctk.TOP, fill=ctk.X, pady=(10, 5), padx=10)

        self.video_label = ctk.CTkLabel(self.window, text='')
        self.video_label.pack(padx=10, pady=10)

        self.letter_label = ctk.CTkLabel(self.window, font=letter_font, fg_color='#2B2B2B', text='')
        self.letter_label.pack(padx=10, pady=10)

        self.sentence_display = ctk.CTkTextbox(self.window, font=text_font)
        self.sentence_display.pack(fill=ctk.X, padx=10, pady=10)

        self.start_button = ctk.CTkButton(self.window, text='START', command=self.start_camera)
        self.start_button.pack(pady=10)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def setup_variables(self):
        self.buffer = deque(maxlen=BUFFER_LENGTH)
        self.prev_sign = ""
        self.sentence = ""
        self.running = False

    def start_camera(self):
        if not self.running:
            self.running = True
            self.process_camera_feed()

    def process_camera_feed(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        all_landmarks = []
        current_signs = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = calc_landmark_list(frame, hand_landmarks)
                all_landmarks.extend(lm_list)

                preprocessed = pre_process_landmark(lm_list)
                if preprocessed:
                    sign_id = keypoint_classifier(preprocessed)
                    sign = keypoint_classifier_labels[sign_id]
                    current_signs.append(sign)

        if current_signs:
            self.buffer.extend(current_signs)
            most_common = Counter(self.buffer).most_common(1)[0][0]
            if most_common != self.prev_sign:
                self.prev_sign = most_common
                self.letter_label.configure(text=most_common)
                self.sentence += most_common
                self.sentence_display.insert("end", most_common)

        frame = draw_heatmap(frame, all_landmarks)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        ctk_img = ctk.CTkImage(dark_image=img, size=(800, 600))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img
        self.window.after(10, self.process_camera_feed)

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    app_window = ctk.CTk()
    app = SignLanguageApp(app_window)
    app_window.protocol("WM_DELETE_WINDOW", app.on_closing)
    app_window.mainloop()
