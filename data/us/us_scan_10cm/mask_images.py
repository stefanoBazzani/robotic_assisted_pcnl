import os
import cv2

def add_black_borders(input_folder, output_folder):
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Itera su tutti i file nella cartella di input
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Controlla se il file è un'immagine supportata
        try:
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"File non supportato o non è un'immagine: {filename}")
                continue

            # Aggiungi i bordi neri
            img[:, 0:300] = 0  # Rettangolo sinistro
            img[:, 1720:1920] = 0  # Rettangolo destro
            img[0:100,:] = 0 # Rettangolo superiore
            img[1040:1080,310:400] = 0 # Rettangolo superiore

            # Salva l'immagine nella cartella di output
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"Errore con il file {filename}: {e}")

# Configura le cartelle
input_folder = "/home/ars_control/Desktop/pcnl/us_scans/us_scan_10cm/original_images"  # Cartella con le immagini originali
output_folder = "/home/ars_control/Desktop/pcnl/us_scans/us_scan_10cm/masked_images"  # Cartella per salvare le immagini modificate

# Esegui la funzione
add_black_borders(input_folder, output_folder)

print("Operazione completata!")
