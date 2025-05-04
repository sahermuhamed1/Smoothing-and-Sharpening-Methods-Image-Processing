import cv2
import numpy as np

class SmoothingFilters:
    @staticmethod
    def mean_filter(image, kernel_size=3):
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def median_filter(image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def gaussian_filter(image, kernel_size=3, sigma=0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def box_filter(image, kernel_size=3):
        return cv2.boxFilter(image, -1, (kernel_size, kernel_size))

    @staticmethod
    def motion_blur(image, kernel_size=15):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def anisotropic_diffusion(image, iterations=10):
        img_float = image.astype('float32')
        for _ in range(iterations):
            laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
            img_float += 0.1 * laplacian
        return np.clip(img_float, 0, 255).astype('uint8')

    @staticmethod
    def nlm_filter(image, h=10, template_window=7, search_window=21):
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window, search_window)

class SharpeningFilters:
    @staticmethod
    def laplacian_filter(image):
        return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)

    @staticmethod
    def high_boost_filter(image, alpha=1.5):
        gaussian = cv2.GaussianBlur(image, (3, 3), 0)
        return cv2.addWeighted(image, alpha, gaussian, -0.5, 0)

    @staticmethod
    def sobel_filter(image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sobelx, sobely).astype(np.uint8)

    @staticmethod
    def prewitt_filter(image):
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(image, -1, kernelx)
        img_prewitty = cv2.filter2D(image, -1, kernely)
        return cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

    @staticmethod
    def unsharp_masking(image, sigma=1.0, strength=1.5):
        gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
        return cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)

    @staticmethod
    def roberts_cross(image):
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        return np.abs(cv2.filter2D(image, -1, kernel_x)) + np.abs(cv2.filter2D(image, -1, kernel_y))

    @staticmethod
    def scharr_filter(image):
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        return cv2.magnitude(scharrx, scharry).astype(np.uint8)

    @staticmethod
    def dog_filter(image, sigma1=1, sigma2=2):
        g1 = cv2.GaussianBlur(image, (0, 0), sigma1)
        g2 = cv2.GaussianBlur(image, (0, 0), sigma2)
        return cv2.subtract(g1, g2)

    @staticmethod
    def emboss_filter(image):
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        return cv2.filter2D(image, -1, kernel) + 128

    @staticmethod
    def kirsch_compass_filter(image):
        kirsch = np.zeros_like(image)
        for angle in range(8):
            k = np.zeros((3, 3), dtype=np.float32)
            k.fill(-3)
            k[1, 1] = 0
            for i in range(3):
                k[i, 2] = 5
            k = np.rot90(k, angle)
            kirsch = np.maximum(kirsch, cv2.filter2D(image, -1, k))
        return kirsch.astype(np.uint8)
