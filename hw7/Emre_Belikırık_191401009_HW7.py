import numpy as np
import cv2 as cv

print("----------------- 1st Question -----------------")

def adaptive_mean_filter(image,gamma):
    img = image.copy()
    return_img = np.zeros(img.shape)
    img = np.single(img)
    img = cv.normalize(img, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    p = cv.copyMakeBorder(img, 2, 2, 2,2, cv.BORDER_REPLICATE, None, value = 0)
    m,n = img.shape
    for i in range(m):
        for j in range(n):
            part = p[i:i+4,j:j+4]
            mean_p = np.average(part)
            var = np.var(part)
            return_img[i][j] = p[i+2][j+2]-(gamma/var)*(p[i+2][j+2]-mean_p)
    return_img = cv.normalize(return_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return return_img.astype(np.uint8)

noisyImage_Gaussian = cv.imread("noisyImage_Gaussian.jpg", cv.IMREAD_GRAYSCALE)

normalized_noisyImage_Gaussian = np.single(noisyImage_Gaussian)
normalized_noisyImage_Gaussian = cv.normalize(normalized_noisyImage_Gaussian, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

b3 = None
b3 = cv.boxFilter(normalized_noisyImage_Gaussian, 0, (3,3), b3, (-1,-1), True, cv.BORDER_REPLICATE)
b3 = cv.normalize(b3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

b5 = None
b5 = cv.boxFilter(normalized_noisyImage_Gaussian, 0, (5,5), b5, (-1,-1), True, cv.BORDER_REPLICATE)
b5 = cv.normalize(b5, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g3 = cv.GaussianBlur(normalized_noisyImage_Gaussian,(3,3),0,borderType = cv.BORDER_REPLICATE)
g3 = cv.normalize(g3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g5 = cv.GaussianBlur(normalized_noisyImage_Gaussian,(5,5),0,borderType = cv.BORDER_REPLICATE)
g5 = cv.normalize(g5, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

adap_mean_filtered = adaptive_mean_filter(normalized_noisyImage_Gaussian,0.0042)

bilateralfiltered = cv.bilateralFilter(normalized_noisyImage_Gaussian, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE)
bilateralfiltered = cv.normalize(bilateralfiltered, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


b3psnr = cv.PSNR(b3, noisyImage_Gaussian)
print("3x3 Box Filter PSNR: ",b3psnr)
b5psnr = cv.PSNR(b5, noisyImage_Gaussian)
print("5x5 Box Filter PSNR:",b5psnr)

g3psnr = cv.PSNR(g3, noisyImage_Gaussian)
print("3x3 Gaussian PSNR:",g3psnr)
g5psnr = cv.PSNR(g5, noisyImage_Gaussian)
print("5x5 Gaussian PSNR:",g5psnr)

adap_mean_filtered_psnr = cv.PSNR(adap_mean_filtered, noisyImage_Gaussian)
print("Adaptive mean filtered PSNR:",adap_mean_filtered_psnr)

bilateralfiltered_psnr = cv.PSNR(bilateralfiltered, noisyImage_Gaussian)
print("Bilateral Filtered PSNR:",bilateralfiltered_psnr)

print("----------------- 2nd Question -----------------")

noisyImage_Gaussian_01 = cv.imread("noisyImage_Gaussian_01.jpg", cv.IMREAD_GRAYSCALE)

normalized_noisyImage_Gaussian_01 = np.single(noisyImage_Gaussian_01)
normalized_noisyImage_Gaussian_01 = cv.normalize(normalized_noisyImage_Gaussian_01, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

b3_01 = None
b3_01 = cv.boxFilter(normalized_noisyImage_Gaussian_01, 0, (3,3), b3_01, (-1,-1), True, cv.BORDER_REPLICATE)
b3_01 = cv.normalize(b3_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

b5_01 = None
b5_01 = cv.boxFilter(normalized_noisyImage_Gaussian_01, 0, (5,5), b5_01, (-1,-1), True, cv.BORDER_REPLICATE)
b5_01 = cv.normalize(b5_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g3_01 = cv.GaussianBlur(normalized_noisyImage_Gaussian_01,(3,3),0,borderType = cv.BORDER_REPLICATE)
g3_01 = cv.normalize(g3_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g5_01 = cv.GaussianBlur(normalized_noisyImage_Gaussian_01,(5,5),0,borderType = cv.BORDER_REPLICATE)
g5_01 = cv.normalize(g5_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

adap_mean_filtered_01 = adaptive_mean_filter(normalized_noisyImage_Gaussian_01,0.0009)

bilateralfiltered_01 = cv.bilateralFilter(normalized_noisyImage_Gaussian_01, 3, 0.1, 1, borderType = cv.BORDER_REPLICATE)
bilateralfiltered_01 = cv.normalize(bilateralfiltered_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


b3_01psnr = cv.PSNR(b3_01, noisyImage_Gaussian_01)
print("3x3 Box Filter PSNR: ",b3_01psnr)
b5_01psnr = cv.PSNR(b5_01, noisyImage_Gaussian_01)
print("5x5 Box Filter PSNR:",b5_01psnr)

g3_01psnr = cv.PSNR(g3_01, noisyImage_Gaussian_01)
print("3x3 Gaussian PSNR:",g3_01psnr)
g5_01psnr = cv.PSNR(g5_01, noisyImage_Gaussian_01)
print("5x5 Gaussian PSNR:",g5_01psnr)

adap_mean_filtered_01_psnr = cv.PSNR(adap_mean_filtered_01, noisyImage_Gaussian_01)
print("Adaptive mean filtered PSNR:",adap_mean_filtered_01_psnr)

bilateralfiltered_01_psnr = cv.PSNR(bilateralfiltered_01, noisyImage_Gaussian_01)
print("Bilateral Filtered PSNR:",bilateralfiltered_01_psnr)

print("----------------- 3rd Question -----------------")


def G_x(x,sigma):
    import math
    return math.exp(-(x**2)/(2*(sigma**2)))/(sigma*((2*math.pi)**0.5))

def kernel_calculation(part,sigma_r,sigma_s):
    sum = 0.0
    normalization = 0.0
    pi = part.shape[0]//2
    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            distance = ((i-pi)**2 + (j-pi)**2)**0.5
            intensity_difference = abs(part[i][j] - part[pi][pi])
            normalization+=(G_x(distance,sigma_s)*G_x(intensity_difference,sigma_r))
            sum += G_x(distance,sigma_s)*G_x(intensity_difference,sigma_r)*part[i,j]
    return sum/normalization


def bilateral_filter(image,kernel_size,sigma_s,sigma_r):
    border = kernel_size//2
    img = image.copy()
    return_img = np.zeros(img.shape)
    img = cv.copyMakeBorder(img, border, border, border,border, cv.BORDER_REPLICATE, None)

    for x in range(return_img.shape[0]):
        for y in range(return_img.shape[1]):
            part = img[x:x+2*border+1, y:y+2*border+1]
            return_img[x,y] = kernel_calculation(part,sigma_s,sigma_r)
    
    return return_img


noisyImage_Gaussian_01 = cv.imread("noisyImage_Gaussian_01.jpg", cv.IMREAD_GRAYSCALE)
normalized_noisyImage_Gaussian_01 = np.single(noisyImage_Gaussian_01)
normalized_noisyImage_Gaussian_01 = cv.normalize(normalized_noisyImage_Gaussian_01, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

noisyImage_Gaussian = cv.imread("noisyImage_Gaussian.jpg", cv.IMREAD_GRAYSCALE)
normalized_noisyImage_Gaussian = np.single(noisyImage_Gaussian)
normalized_noisyImage_Gaussian = cv.normalize(normalized_noisyImage_Gaussian, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)


cv1 = cv.bilateralFilter(normalized_noisyImage_Gaussian_01, 5, 0.1, 1, borderType = cv.BORDER_REPLICATE)
cv1 = cv.normalize(cv1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

cv2 = cv.bilateralFilter(normalized_noisyImage_Gaussian, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE)
cv2 = cv.normalize(cv2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

mine1 = bilateral_filter(normalized_noisyImage_Gaussian_01,5, 0.1, 1)
mine1 = cv.normalize(mine1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

mine2 = bilateral_filter(normalized_noisyImage_Gaussian,5, 3, 0.9)
mine2 = cv.normalize(mine2, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

cv.imshow("Bilateral filtered noisyImage_Gaussian_01 output(OPENCV)",mine1.astype(np.uint8))

cv.imshow("Bilateral filtered noisyImage_Gaussian_01 output(My Filter)",mine1.astype(np.uint8))

cv.imshow("Bilateral filtered noisyImage_Gaussian output(OPENCV)",mine1.astype(np.uint8))

cv.imshow("Bilateral filtered noisyImage_Gaussian output(My Filter)",mine1.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()

abs_img1 = abs(mine1.astype(float)-cv1.astype(float))
print("[noisyImage_Gaussian_01] Max pixel intensity difference:",round(abs_img1.max()))

abs_img2 = abs(mine2.astype(float)-cv2.astype(float))
print("[noisyImage_Gaussian] Max pixel intensity difference:",round(abs_img2.max()))