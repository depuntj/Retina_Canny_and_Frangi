from joblib import load
import cv2
import sys
import numpy as np
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
import matplotlib.pyplot as plt

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)  # use highdpi icons

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.Image = None
        self.model_filename = 'knn_model.joblib'
        self.knn_model = load(self.model_filename)
        self.prediction_value = None  # Add a class attribute for prediction
        self.initUI()

    def initUI(self):
        loadUi('first.ui', self)
        self.pushButton.clicked.connect(self.load_second_ui)
        self.pushButton_2.clicked.connect(self.open_third_ui)

    def load_second_ui(self):
        loadUi('second2.ui', self)
        self.btn_loadimage.clicked.connect(self.load_image)
        self.actionKonversi.triggered.connect(self.konversi)

        self.actionMask_Ripe.triggered.connect(self.maskRipe)
        self.actionMask_Unripe.triggered.connect(self.maskUnripe)
        self.actionMask_Half_Ripe.triggered.connect(self.maskHalfripe)
        self.actionAll_Mask.triggered.connect(self.maskAll)

        self.actionMorfologi_Mask_Ripe.triggered.connect(self.morfoRipe)
        self.actionMorfologi_Mask_Unripe.triggered.connect(self.morfoUnripe)
        self.actionMorfologi_Mask_Half_Ripe.triggered.connect(self.morfoHalfripe)
        self.actionAll_Morfologi_Mask.triggered.connect(self.morfoAll)

        self.actionContour_Half_Ripe.triggered.connect(self.contourAll)

    def open_third_ui(self):
        self.third_window = QtWidgets.QMainWindow()
        loadUi('third.ui', self.third_window)
        self.third_window.show()


    def load_image(self):
        try:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.AnyFile)
            file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
            if file_dialog.exec_():
                file_names = file_dialog.selectedFiles()
                image_path = file_names[0]
                self.Image = cv2.imread(image_path)
                self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
                if self.Image is None:
                    print("Error: Unable to load image.")
                else:
                    self.Image = cv2.GaussianBlur(self.Image, (5, 5), 0)
                    # self.Image = self.apply_kmeans_segmentation(self.Image)
                    self.displayImage()
                    self.btn_prosescitra.clicked.connect(lambda: self.predict_image_category(image_path))
                    self.btn_lihatproses.clicked.connect(self.lihat_proses)
        except Exception as e:
            print("Error:", e)
    def displayImage(self):
        qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
        self.label.setPixmap(QPixmap.fromImage(img))
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)

    def preprocess_image(self, image_path):
        image_asli = cv2.imread(image_path)
        cv2.imwrite("images/0_Original.png", image_asli)
        image = cv2.cvtColor(image_asli, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cv2.imwrite("images/0_RGB.png", image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Convert to HSV
        cv2.imwrite("images/0_HSV.png", hsv_image)

        # mask_ripe = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
        # cv2.imwrite("images/1_ripe.png", mask_ripe)
        # mask_unripe = cv2.inRange(hsv_image, (35, 100, 100), (85, 255, 255))
        # cv2.imwrite("images/1_unripe.png", mask_unripe)
        # mask_half_ripe = cv2.inRange(hsv_image, (25, 100, 100), (35, 255, 255))
        # cv2.imwrite("images/1_halfripe.png", mask_half_ripe)

        # mask_ripe = cv2.inRange(hsv_image, (1, 75, 83), (178, 80, 84))
        # cv2.imwrite("images/1_ripe.png", mask_ripe)
        # mask_unripe = cv2.inRange(hsv_image, (93, 64, 62), (63, 65, 52))
        # cv2.imwrite("images/1_unripe.png", mask_unripe)
        # mask_half_ripe = cv2.inRange(hsv_image, (28, 63, 63), (17, 71, 84))
        # cv2.imwrite("images/1_halfripe.png", mask_half_ripe)

        # Unripe Mask
        lower_unripe = np.array([14, 100, 100], dtype=np.uint8)
        upper_unripe = np.array([47, 255, 255], dtype=np.uint8)
        mask_unripe = cv2.inRange(hsv_image, lower_unripe, upper_unripe)
        cv2.imwrite("images/1_unripe.png", mask_unripe)

        # Ripe Mask
        # lower_half_ripe = np.array([0, 100, 100], dtype=np.uint8)
        # upper_half_ripe = np.array([10, 255, 255], dtype=np.uint8)
        lower_half_ripe = np.array([0, 100, 100], dtype=np.uint8)
        upper_half_ripe = np.array([10, 255, 255], dtype=np.uint8)
        mask_ripe = cv2.inRange(hsv_image, lower_half_ripe, upper_half_ripe)
        cv2.imwrite("images/1_ripe.png", mask_ripe)

        # Half Ripe Mask
        # lower_ripe = np.array([25, 100, 100], dtype=np.uint8)
        # upper_ripe = np.array([35, 255, 255], dtype=np.uint8)
        lower_ripe = np.array([10, 100, 100], dtype=np.uint8)
        upper_ripe = np.array([14, 255, 255], dtype=np.uint8)
        mask_half_ripe = cv2.inRange(hsv_image, lower_ripe, upper_ripe)
        cv2.imwrite("images/1_halfripe.png", mask_half_ripe)

        mean_ripe = cv2.mean(hsv_image, mask=mask_ripe)[:3]
        mean_unripe = cv2.mean(hsv_image, mask=mask_unripe)[:3]
        mean_half_ripe = cv2.mean(hsv_image, mask=mask_half_ripe)[:3]

        features = np.concatenate([mean_ripe, mean_unripe, mean_half_ripe])

        # Morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        mask_ripe = cv2.morphologyEx(mask_ripe, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("images/2_ripe.png", mask_ripe)
        mask_unripe = cv2.morphologyEx(mask_unripe, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("images/2_unripe.png", mask_unripe)
        mask_half_ripe = cv2.morphologyEx(mask_half_ripe, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("images/2_halfripe.png", mask_half_ripe)

        # Find contours
        contours_ripe, _ = cv2.findContours(mask_ripe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_unripe, _ = cv2.findContours(mask_unripe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_half_ripe, _ = cv2.findContours(mask_half_ripe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw contours on images
        # contours_ripe_img = np.zeros_like(image)
        # contours_unripe_img = np.zeros_like(image)
        # contours_half_ripe_img = np.zeros_like(image)

        # Create copies of the original image
        ripe_image = image.copy()
        unripe_image = image.copy()
        half_ripe_image = image.copy()

        # Draw contours on the copies
        cv2.drawContours(ripe_image, contours_ripe, -1, (0, 0, 255), thickness=2)
        cv2.drawContours(unripe_image, contours_unripe, -1, (0, 0, 255), thickness=2)
        cv2.drawContours(half_ripe_image, contours_half_ripe, -1, (0, 0, 255), thickness=2)

        # Save the images
        cv2.imwrite("images/3_ripe.png", ripe_image)
        cv2.imwrite("images/3_unripe.png", unripe_image)
        cv2.imwrite("images/3_halfripe.png", half_ripe_image)

        # Return image and contours
        return features


    def predict_image_category(self, image_path):
        features = self.preprocess_image(image_path)
        feature = features.reshape(1, -1)
        prediction = self.knn_model.predict(feature)
        print(prediction)

        if prediction == 0:
            category = 'Unripe'
        elif prediction == 1:
            category = 'Ripe'
        elif prediction == 2:
            category = 'Half-Ripe'
        else:
            category = 'Unknown'

        self.label_prediction.setText(f'Prediction: {category}')
        self.prediction_value = prediction  # Store the prediction in an attribute
        return prediction

    def konversi(self):
        try:
            # Load and display saved images
            img_RGB = cv2.imread('images/0_RGB.png')
            img_HSV = cv2.imread('images/0_HSV.png')

            if img_RGB is None or img_HSV is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(img_RGB, cmap='gray', interpolation='bicubic'), plt.title('Image RGB')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img_HSV, cmap='gray', interpolation='bicubic'), plt.title('Image HSV')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def maskRipe(self):
        try:
            # Load and display saved images
            image_asli = cv2.imread('images/0_RGB.png')
            mask_ripe = cv2.imread('images/1_ripe.png')

            if mask_ripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(image_asli, cmap='gray', interpolation='bicubic'), plt.title('Image Asli')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(mask_ripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def maskUnripe(self):
        try:
            # Load and display saved images
            image_asli = cv2.imread('images/0_RGB.png')
            mask_unripe = cv2.imread('images/1_unripe.png')

            if mask_unripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(image_asli, cmap='gray', interpolation='bicubic'), plt.title('Image Asli')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(mask_unripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Unripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def maskHalfripe(self):
        try:
            # Load and display saved images
            image_asli = cv2.imread('images/0_RGB.png')
            mask_halfripe = cv2.imread('images/1_halfripe.png')

            if mask_halfripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(image_asli, cmap='gray', interpolation='bicubic'), plt.title('Image Asli')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(mask_halfripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Half Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def maskAll(self):
        try:
            # Load and display saved images
            img_asli = cv2.imread('images/0_RGB.png')
            mask_ripe = cv2.imread('images/1_ripe.png')
            mask_unripe = cv2.imread('images/1_unripe.png')
            mask_halfripe = cv2.imread('images/1_halfripe.png')

            if img_asli is None or mask_halfripe is None or mask_unripe is None or mask_ripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(221), plt.imshow(img_asli, cmap='gray', interpolation='bicubic'), plt.title('ORIGINAL')
            plt.xticks([]), plt.yticks([])
            plt.subplot(222), plt.imshow(mask_ripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Ripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(223), plt.imshow(mask_unripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Unripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(224), plt.imshow(mask_halfripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Half Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def morfoRipe(self):
        try:
            # Load and display saved images
            mask_ripe = cv2.imread('images/1_ripe.png')
            morfo_ripe = cv2.imread('images/2_ripe.png')

            if morfo_ripe is None or mask_ripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(mask_ripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Ripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(morfo_ripe, cmap='gray', interpolation='bicubic'), plt.title('Morfologi Mask Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def morfoUnripe(self):
        try:
            # Load and display saved images
            mask_unripe = cv2.imread('images/1_unripe.png')
            morfo_unripe = cv2.imread('images/2_unripe.png')

            if mask_unripe is None or morfo_unripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(mask_unripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Unripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(morfo_unripe, cmap='gray', interpolation='bicubic'), plt.title('Morfologi Mask Unripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def morfoHalfripe(self):
        try:
            # Load and display saved images
            mask_halfripe = cv2.imread('images/1_halfripe.png')
            morfo_halfripe = cv2.imread('images/2_halfripe.png')

            if mask_halfripe is None or morfo_halfripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(121), plt.imshow(mask_halfripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Half Ripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(morfo_halfripe, cmap='gray', interpolation='bicubic'), plt.title('Morfologi Mask Half Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def morfoAll(self):
        try:
            # Load and display saved images
            img_asli = cv2.imread('images/0_RGB.png')
            mask_ripe = cv2.imread('images/2_ripe.png')
            mask_unripe = cv2.imread('images/2_unripe.png')
            mask_halfripe = cv2.imread('images/2_halfripe.png')

            if img_asli is None or mask_halfripe is None or mask_unripe is None or mask_ripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(221), plt.imshow(img_asli, cmap='gray', interpolation='bicubic'), plt.title('ORIGINAL')
            plt.xticks([]), plt.yticks([])
            plt.subplot(222), plt.imshow(mask_ripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Ripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(223), plt.imshow(mask_unripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Unripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(224), plt.imshow(mask_halfripe, cmap='gray', interpolation='bicubic'), plt.title('Mask Half Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def contourAll(self):
        try:
            # Load and display saved images
            image_asli = cv2.imread('images/0_RGB.png')
            c_ripe = cv2.imread('images/3_ripe.png')
            c_unripe = cv2.imread('images/3_unripe.png')
            c_halfripe = cv2.imread('images/3_halfripe.png')

            if c_halfripe is None or c_unripe is None or c_ripe is None:
                raise FileNotFoundError("One or both images not found")

            plt.subplot(221), plt.imshow(image_asli, cmap='gray', interpolation='bicubic'), plt.title('ORIGINAL')
            plt.xticks([]), plt.yticks([])
            plt.subplot(222), plt.imshow(c_ripe, cmap='gray', interpolation='bicubic'), plt.title('Contour Ripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(223), plt.imshow(c_unripe, cmap='gray', interpolation='bicubic'), plt.title('Contour Unripe')
            plt.xticks([]), plt.yticks([])
            plt.subplot(224), plt.imshow(c_halfripe, cmap='gray', interpolation='bicubic'), plt.title('Contour Half Ripe')
            plt.xticks([]), plt.yticks([])
            plt.show()

        except Exception as e:
            print("An error occurred:", e)

    def lihat_proses(self):
        prediction = self.prediction_value  # Use the stored prediction value
        image_asli = cv2.imread('images/0_RGB.png')
        if prediction == 1:
            mask = cv2.imread('images/1_ripe.png')
            morfo = cv2.imread('images/2_ripe.png')
            contour = cv2.imread('images/3_ripe.png')
        if prediction == 0:
            mask= cv2.imread('images/1_unripe.png')
            morfo = cv2.imread('images/2_unripe.png')
            contour = cv2.imread('images/3_unripe.png')
        if prediction == 2:
            mask = cv2.imread('images/1_halfripe.png')
            morfo = cv2.imread('images/2_halfripe.png')
            contour = cv2.imread('images/3_halfripe.png')

        plt.subplot(221), plt.imshow(image_asli, cmap='gray', interpolation='bicubic'), plt.title('ORIGINAL')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(mask, cmap='gray', interpolation='bicubic'), plt.title('Masking')
        plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(morfo, cmap='gray', interpolation='bicubic'), plt.title('Morphology')
        plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(contour, cmap='gray', interpolation='bicubic'), plt.title('Contour')
        plt.xticks([]), plt.yticks([])
        plt.show()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.setWindowTitle('KELOMPOK C2')
window.show()
sys.exit(app.exec_())
