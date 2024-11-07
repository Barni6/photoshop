import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photoshop")
        self.root.attributes('-fullscreen', True)
        self.panel = None
        self.canvas = None

        self.panel = tk.Label(root)
        self.panel.grid(row=3, column=0, columnspan=5, padx=10, pady=10)

        # Kép betöltése gomb
        self.load_button = tk.Button(root, text="Kép betöltése", command=self.load_image, bg="lightblue", fg="black")
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

    # Kép alaphelyzetbe állítása gomb
        self.reset_button = tk.Button(root, text="Kép alaphelyzetbe állítása", command=self.reset_image, bg="lightblue", fg="black")
        self.reset_button.grid(row=0, column=1, padx=10, pady=10)

    # Gamma transzformáció gomb
        self.gamma_button = tk.Button(root, text="Gamma transzformáció", command=self.apply_gamma_transform, bg="beige", fg="black")
        self.gamma_button.grid(row=0, column=2, padx=10, pady=10)

    # Gamma label és entry
        self.gamma_label = tk.Label(root, text="Gamma érték:")
        self.gamma_label.grid(row=0, column=3, padx=10, pady=10)

        self.gamma_entry = tk.Entry(root)
        self.gamma_entry.grid(row=0, column=4, padx=10, pady=10)

    # Negálás gomb
        self.negate_button = tk.Button(root, text="Negálás", command=self.negate_image, bg="beige", fg="black")
        self.negate_button.grid(row=1, column=0, padx=10, pady=10)

    # Logaritmikus transzformáció gomb
        self.log_transform_button = tk.Button(root, text="Logaritmikus transzformáció", command=self.apply_log_transform, bg="beige", fg="black")
        self.log_transform_button.grid(row=1, column=1, padx=10, pady=10)

    # Szürkítés gomb
        self.grayscale_button = tk.Button(root, text="Szürkítés", command=self.apply_grayscale, bg="beige", fg="black")
        self.grayscale_button.grid(row=1, column=2, padx=10, pady=10)

    # Hisztogram készítés gomb
        self.histogram_button = tk.Button(root, text="Hisztogram készítés", command=self.create_histogram, bg="beige", fg="black")
        self.histogram_button.grid(row=1, column=3, padx=10, pady=10)

    # Hisztogram kiegyenlítés gomb
        self.equalize_histogram_button = tk.Button(root, text="Hisztogram kiegyenlítés", command=self.equalize_histogram, bg="beige", fg="black")
        self.equalize_histogram_button.grid(row=1, column=4, padx=10, pady=10)

    # Átlagoló szűrő gomb
        self.average_filter_button = tk.Button(root, text="Átlagoló szűrő", command=self.apply_average_filter, bg="beige", fg="black")
        self.average_filter_button.grid(row=2, column=0, padx=10, pady=10)

    # Gauss szűrő gomb
        self.gaussian_blur_button = tk.Button(root, text="Gauss szűrő", command=self.apply_gaussian_blur, bg="beige", fg="black")
        self.gaussian_blur_button.grid(row=2, column=1, padx=10, pady=10)

    # Sobel éldetektor gomb
        self.sobel_edge_detector_button = tk.Button(root, text="Sobel éldetektor", command=self.apply_sobel_edge_detector, bg="beige", fg="black")
        self.sobel_edge_detector_button.grid(row=2, column=2, padx=10, pady=10)

    # Laplace éldetektor gomb
        self.laplace_edge_detector_button = tk.Button(root, text="Laplace éldetektor", command=self.apply_laplace_edge_detector, bg="beige", fg="black")
        self.laplace_edge_detector_button.grid(row=2, column=3, padx=10, pady=10)

    # Jellemzőpontok detektálása gomb
        self.detect_keypoints_button = tk.Button(root, text="Jellemzőpontok detektálása", command=self.detect_keypoints, bg="beige", fg="black")
        self.detect_keypoints_button.grid(row=2, column=4, padx=10, pady=10)

    # Kilépés gomb hozzáadása
        self.exit_button = tk.Button(root, text="Kilépés", command=self.root.destroy, bg="#FF6464", fg="black")
        self.exit_button.grid(row=0, column=5, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.image = self.original_image.copy()
            self.show_image()

    def show_image(self,  processing_time=None):
        if self.image is not None:
            # Átméretezzük a self.image képet a megfelelő méretre
            resized_image = cv2.resize(self.image, (1280, 720))
            # Konvertáljuk az 8 bites mélységű képpé
            resized_image = cv2.convertScaleAbs(resized_image)
            # Konvertáljuk RGB formátumba
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            # Kép megjelenítése
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image=image_pil)
            self.panel.config(image=image_tk)
            self.panel.image = image_tk

            # Kiírjuk a függvény futási idejét a kép alá
            if processing_time is not None:
                time_label = tk.Label(self.root, text=f"Futási idő: {processing_time:.4f} másodperc")
                time_label.grid(row=1, column=5, padx=10, pady=10, sticky='w')

    def reset_image(self):
        # Visszaállítjuk az eredeti képet
        self.image = self.original_image.copy()
        # Megjelenítjük az eredeti képet
        self.show_image()

    def negate_image(self):
        if self.image is not None:
            start_time = time.time()
            negated_image = self.negate(self.image)
            self.image = np.uint8(negated_image)
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def negate(self, image):
        return 255 - image

    def apply_gamma_transform(self):
        if self.image is not None:
            start_time = time.time()
            gamma = float(self.gamma_entry.get())
            gamma_corrected = self.gamma_transform(self.image, gamma)
            self.image = np.uint8(gamma_corrected)
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def gamma_transform(self, image, gamma):
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image, gamma_table)

    def apply_log_transform(self):
        if self.image is not None:
            start_time = time.time()
            log_transformed = self.log_transform(self.image)
            self.image = np.uint8(log_transformed)
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def log_transform(self, image):
        c = 255 / np.log(1 + np.max(image))
        log_transformed = c * (np.log(image + 1))
        return log_transformed

    def apply_grayscale(self):
        if self.image is not None:
            start_time = time.time()
            grayscale_image = self.rgb_to_grayscale(self.image)
            self.image = cv2.merge([grayscale_image, grayscale_image, grayscale_image])
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def rgb_to_grayscale(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def create_histogram(self):
        if self.image is not None:
            start_time = time.time()
            histogram = self.calculate_histogram(self.image)
            fig, ax = plt.subplots()
            ax.plot(range(256), histogram, color='r')
            ax.set_title("Image Histogram")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            # Mentjük a hisztogramot képként
            fig.savefig("histogram.png")
            plt.close(fig)
            end_time = time.time()
            processing_time = end_time - start_time
            self.display_histogram(fig, processing_time)

    def calculate_histogram(self, image):
        histogram = np.zeros(256)
        rows, cols, _ = image.shape
        for i in range(rows):
            for j in range(cols):
                pixel_value = int(image[i, j, 0])
                histogram[pixel_value] += 1
        return histogram

    def display_histogram(self, fig, processing_time=None):
        if fig is not None:
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=3, column=5, columnspan=5, padx=10, pady=10)
        # Kiírjuk a függvény futási idejét a kép alá
        if processing_time is not None:
            time_label = tk.Label(self.root, text=f"Futási idő: {processing_time:.4f} másodperc")
            time_label.grid(row=1, column=5, padx=10, pady=10, sticky='w')

    def equalize_histogram(self):
        if self.image is not None:
            start_time = time.time()
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = self.equalize_histogram_manually(gray_image)
            self.image = cv2.merge([equalized_image, equalized_image, equalized_image])
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def equalize_histogram_manually(self, image):
        histogram, _ = np.histogram(image.flatten(), 256, [0, 256])
        cdf = histogram.cumsum()
        cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
        equalized_image = cdf_normalized[image]
        return equalized_image.astype(np.uint8)

    def apply_average_filter(self):
        if self.image is not None:
            start_time = time.time()
            kernel = np.ones((5, 5), np.float32) / 25
            filtered_image = self.apply_filter(self.image, kernel)
            self.image = np.uint8(filtered_image)
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def apply_filter(self, image, kernel):
        return cv2.filter2D(image, -1, kernel)

    def apply_gaussian_blur(self):
        if self.image is not None:
            start_time = time.time()
            kernel = self.generate_gaussian_kernel(5, 1.0)
            blurred_image = self.apply_filter(self.image, kernel)
            self.image = np.uint8(blurred_image)
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def generate_gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size))
        return kernel / np.sum(kernel)

    def apply_sobel_edge_detector(self):
        if self.image is not None:
            start_time = time.time()
            gray_image = self.rgb_to_grayscale(self.image)
            sobelx, sobely = self.calculate_sobel_gradients(gray_image)
            edge_image = self.calculate_edge_magnitude(sobelx, sobely)
            self.image = cv2.merge([edge_image, edge_image, edge_image])
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def calculate_sobel_gradients(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return sobelx, sobely

    def calculate_edge_magnitude(self, sobelx, sobely):
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_magnitude *= 255.0 / edge_magnitude.max()
        return edge_magnitude

    def apply_laplace_edge_detector(self):
        if self.image is not None:
            start_time = time.time()
            gray_image = self.rgb_to_grayscale(self.image)
            laplacian = self.calculate_laplacian(gray_image)
            self.image = cv2.merge([laplacian, laplacian, laplacian])
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def calculate_laplacian(self, image):
        rows, cols = image.shape
        laplacian_image = np.zeros_like(image, dtype=np.float32)
        laplacian_operator = np.array([[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]])
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                laplacian_value = np.sum(image[i - 1:i + 2, j - 1:j + 2] * laplacian_operator)
                laplacian_image[i, j] = laplacian_value
        return laplacian_image

    def detect_keypoints(self):
        if self.image is not None:
            start_time = time.time()
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)
            keypoint_image = self.draw_keypoints(self.image, keypoints)
            self.image = keypoint_image
            end_time = time.time()
            processing_time = end_time - start_time
            self.show_image(processing_time)

    def detect_orb_keypoints(self, image):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def draw_keypoints(self, image, keypoints):
        keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return keypoint_image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

