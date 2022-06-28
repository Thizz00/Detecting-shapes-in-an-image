import cv2
import matplotlib.pyplot as plt
img=cv2.imread('shapes1.png')
class detectclass:
    def blur_image(self,img):
        blurred_img=cv2.GaussianBlur(img.copy(),(5,5),1)
        self.edges=cv2.Canny(blurred_img,100,160)

    def detect_Countours(self,img):
        contours,hierarchy=cv2.findContours(self.edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        imgcopy=img.copy()
        cv2.drawContours(imgcopy,contours,-1,(0,255,0),2)
        plt.figure(figsize=[10,10])
        plt.imshow(imgcopy)
        plt.axis("off")
        plt.show()

detect=detectclass()
detect.blur_image(img)
detect.detect_Countours(img)
cv2.destroyAllWindows()