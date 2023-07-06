import cv2
import mahotas

class Pre_processing:
    def __init__(self, image_path: str, scale_percent: int = 20):
        self.img = cv2.imread(image_path)
        self.scale_percent = scale_percent

    def operate(self, resize_only: bool = False):
        self.resize_input()
        if resize_only:
            self.results = [self.resized_img]
        else:
            self.filters_step()
            self.results = [self.adap_thresh, self.bin_thresh, self.canny, self.othresh, self.sobel]
        return self.results
        
    def resize_input(self):
        width = self.img.shape[1] * self.scale_percent // 100
        height = self.img.shape[0] * self.scale_percent // 100
        self.resized_img = cv2.resize(self.img, (width, height), interpolation = cv2.INTER_AREA)

    def filters_step(self, **kwargs) -> None:
        gray_img = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2GRAY)
        gaussian_ksize = kwargs.get("gaussian_ksize", 3)
        blur_img = cv2.GaussianBlur(gray_img, (gaussian_ksize,gaussian_ksize), cv2.BORDER_DEFAULT)
        
        # Apply Sobel filter
        sobel_ksize = kwargs.get("sobel_ksize", 3)
        grad_x = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize = sobel_ksize)
        grad_y = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize = sobel_ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        self.sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Apply Cany filter
        canny_low_th = kwargs.get("canny_lower", 50)
        canny_upper_th = kwargs.get("canny_upper", 150)
        self.canny = cv2.Canny(blur_img, canny_low_th, canny_upper_th)

        # Apply Binary Threshold
        bin_th_val = kwargs.get("bin_thresh", 127)
        self.bin_thresh = cv2.threshold(blur_img, bin_th_val, 255, cv2.THRESH_BINARY)[1]
        self.bin_thresh = cv2.bitwise_not(self.bin_thresh)

        # Apply Adaptative Gaussian Threshold
        self.adap_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11 ,2)
        self.adap_thresh = cv2.bitwise_not(self.adap_thresh)

        # othresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        T = mahotas.rc(blur_img)
        thresh = gray_img
        thresh[thresh < T] =0
        thresh[thresh > T] =255
        self.othresh = cv2.bitwise_not(thresh)

    def imshow(self):
        cv2.imshow("adaptive", self.adap_thresh)
        cv2.imshow("sobel", self.sobel)
        cv2.imshow("canny", self.canny)
        cv2.imshow("bin", self.bin_thresh)
        cv2.imshow("UWU", self.othresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()