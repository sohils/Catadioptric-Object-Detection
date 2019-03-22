import numpy as np
import cv2
from matplotlib import pyplot as plt

from scipy.interpolate import RectBivariateSpline

# def get_omni_gradient():
#     # Calculate gradient 
#     gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
#     gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

#     mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


#     im_shape = im.shape

#     centre = [i/2 for i in im_shape]

#     riemannian_inv = np.zeros(im.shape)

#     x = np.arange(im_shape[0])
#     y = np.arange(im_shape[1])
#     xv, yv = np.meshgrid(y, x)

#     xv = xv - centre[1]
#     yv = centre[0] - yv
#     print(xv[0], yv[0])

#     riemannian_inv = np.square((np.square(xv) + np.sqaure(yv) + 4))/16

def extract_omni_window(im,r1,r2,th1,th2):
    x_range = np.arange(0,im.shape[0])
    y_range = np.arange(0,im.shape[1])

    im_spline_r = RectBivariateSpline(x_range, y_range, im[:,:,0])
    im_spline_g = RectBivariateSpline(x_range, y_range, im[:,:,1])
    im_spline_b = RectBivariateSpline(x_range, y_range, im[:,:,2])

    im_rect = np.zeros((int(np.floor(np.abs(r1-r2))),5*int(np.floor(np.abs((r2-r1)*(th2-th1)/2)))))

    omni_indices_i = np.linspace(r2, r1, im_rect.shape[0])
    omni_indices_j = np.linspace(th2, th1, im_rect.shape[1])

    cos_omni_indices_j = np.cos(omni_indices_j)
    sin_omni_indices_j = np.sin(omni_indices_j)
    
    x_i_r = np.repeat(omni_indices_i, im_rect.shape[1])
    x_i = im.shape[0]/2 - x_i_r*(np.tile(sin_omni_indices_j, im_rect.shape[0]).flatten())
    y_i = im.shape[1]/2 + x_i_r*(np.tile(cos_omni_indices_j, im_rect.shape[0]).flatten())

    image = im_spline_r.ev(x_i,y_i).reshape(im_rect.shape)
    image=np.dstack((image,im_spline_g.ev(x_i,y_i).reshape(im_rect.shape)))
    image=np.dstack((image,im_spline_b.ev(x_i,y_i).reshape(im_rect.shape)))
    print(image.shape)
    plt.imshow(image)
    plt.show()
    # cv2.imshow("rect", im_rect)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image

if __name__ == "__main__":
    # Read image
    im = cv2.imread('../data/cctag/19_59_51_356.png')
    # im = cv2.imread('color-wheel.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(im.shape)
    im = np.float32(im) / 255.0
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    extract_omni_window(im,350,400,0,0.5)