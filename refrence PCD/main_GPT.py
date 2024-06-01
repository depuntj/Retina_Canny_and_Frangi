import cv2
import numpy as np

def preprocess_image(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        print("Gagal membaca gambar.")
        return None
    
    
    scale_percent = 30 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    edges = cv2.Canny(blurred_image, 15, 35)
    
    return edges

def main():
    image_path = 'F:\PCD project akhir RETINA\img (05).jpg' # <----ubah ke path foto 
    processed_image = preprocess_image(image_path)
    
    if processed_image is not None:
        original_image = cv2.imread(image_path)
        scale_percent = 30 
        width = int(original_image.shape[1] * scale_percent / 100)
        height = int(original_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        original_image_resized = cv2.resize(original_image, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow('Original Image', original_image_resized)
        cv2.imshow('Processed Image - Edges', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
