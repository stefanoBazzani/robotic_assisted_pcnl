""" This code add black borders to US images acquired by the ultrasound probe """
import os
import cv2

def add_black_borders(input_folder, output_folder):
    
    # Create output dir if it not already exists
    os.makedirs(output_folder, exist_ok=True)

    # Iteration over all the images within the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        try:
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"File not supported: {filename}")
                continue

            # Aggiungi i bordi neri
            img[:, 0:300] = 0           # Left border
            img[:, 1720:1920] = 0       # Right border
            img[0:100,:] = 0            # Upper border
            img[1040:1080,310:400] = 0  # Bottom border

            # Save image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"Errore con il file {filename}: {e}")

input_folder = "/home/ars_control/Desktop/pcnl/us_scans/us_scan_10cm/original_images"   # Folder with original US images
output_folder = "/home/ars_control/Desktop/pcnl/us_scans/us_scan_10cm/masked_images"    # Folder for masked US images

add_black_borders(input_folder, output_folder)