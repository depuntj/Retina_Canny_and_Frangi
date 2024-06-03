import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
import os
import pickle
import sqlite3
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QApplication
from PyQt5.QtCore import Qt

# UI for the retina image matcher
class RetinaMatcherApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Retina Image Matcher')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.matching_image_label = QLabel(self)
        self.matching_image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.matching_image_label)

        self.result_label = QLabel(self)
        layout.addWidget(self.result_label)

        self.upload_btn = QPushButton('Upload Image', self)
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        self.match_btn = QPushButton('Match Retina Image', self)
        self.match_btn.clicked.connect(self.match_image)
        layout.addWidget(self.match_btn)

        self.setLayout(layout)

    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Retina Image", "", "Images (*.png *.xpm *.jpg *.ppm);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QtGui.QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, aspectRatioMode=Qt.KeepAspectRatio))

    def match_image(self):
        if not hasattr(self, 'image_path'):
            self.result_label.setText("Please upload an image first.")
            return

        # Load the input image and compute SIFT features
        input_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        input_keypoints, input_descriptors = sift.detectAndCompute(input_image, None)

        # Load the database
        conn = sqlite3.connect('retina_partitions.db')
        cursor = conn.cursor()

        cursor.execute("SELECT path, keypoints, descriptors FROM images")
        rows = cursor.fetchall()

        bf = cv2.BFMatcher()
        best_match = None
        best_distance = float('inf')

        for row in rows:
            db_path, db_keypoints_blob, db_descriptors_blob = row
            db_keypoints = pickle.loads(db_keypoints_blob)
            db_descriptors = pickle.loads(db_descriptors_blob)

            matches = bf.knnMatch(input_descriptors, db_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            distance = sum([m.distance for m in good_matches])

            if distance < best_distance:
                best_distance = distance
                best_match = db_path

        conn.close()

        if best_match:
            self.result_label.setText(f"Matched with: {best_match}\nDistance: {best_distance:.2f}")
            matched_pixmap = QtGui.QPixmap(best_match)
            self.matching_image_label.setPixmap(matched_pixmap.scaled(400, 400, aspectRatioMode=Qt.KeepAspectRatio))
        else:
            self.result_label.setText("No match found.")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    # Ensure database is created
    partition_dir = 'partitions'
    if not os.path.exists('retina_partitions.db'):
        create_database(partition_dir)

    ex = RetinaMatcherApp()
    ex.show()
    sys.exit(app.exec_())