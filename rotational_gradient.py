import numpy as np
import cv2

def main():
    img = cv2.imread('IMG_8693.JPG',0) 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded_img = np.pad(img, 1, 'constant', constant_values=0)
    img_shape = img.shape

    centre = [i/2 for i in padded_img.shape]

    filter_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    filter_y = np.transpose(filter_x)
    
    gradient_radial_image = np.zeros(img_shape)
    gradient_tangential_image = np.zeros(img_shape)

    for i in range(1,padded_img.shape[0]-1):
        print(i)
        for j in range(1,padded_img.shape[1]-1):
            theta = np.arctan2(i-centre[0], j-centre[1])

            radial_filter = -np.cos(theta)*filter_x + np.sin(theta)*filter_y
            tangential_filter = np.cos(theta)*filter_x + np.sin(theta)*filter_y
            
            gradient_radial_image[i-1,j-1] = np.sum(radial_filter*(padded_img[i-1:i+2, j-1:j+2]))
            gradient_tangential_image[i-1,j-1] = np.sum(tangential_filter*(padded_img[i-1:i+2, j-1:j+2]))
    print("Done calculations")
    gradient_radial_image = cv2.resize(gradient_radial_image, (800,1200))
    cv2.imshow("Radial Gradient",gradient_radial_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gradient_tangential_image = cv2.resize(gradient_tangential_image, (800,1200))
    cv2.imshow("Tangential Gradient",gradient_tangential_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()