import numpy as np
import cv2 as cv

lena_grayscale_hq = cv.imread("lena_grayscale_hq.jpg", cv.IMREAD_GRAYSCALE)

print("------------------ 1st Question ------------------")
 
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


b3 = cv.blur(normalized_noisyImage_Gaussian, (3,3), borderType=cv.BORDER_REPLICATE)
b3 = cv.normalize(b3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

b5 = cv.blur(normalized_noisyImage_Gaussian, (5,5), borderType=cv.BORDER_REPLICATE)
b5 = cv.normalize(b5, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g3 = cv.GaussianBlur(normalized_noisyImage_Gaussian,(3,3),0,borderType = cv.BORDER_REPLICATE)
g3 = cv.normalize(g3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g5 = cv.GaussianBlur(normalized_noisyImage_Gaussian,(5,5),0,borderType = cv.BORDER_REPLICATE)
g5 = cv.normalize(g5, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

adap_mean_filtered = adaptive_mean_filter(normalized_noisyImage_Gaussian,0.0042)

bilateralfiltered = cv.bilateralFilter(normalized_noisyImage_Gaussian, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE)
bilateralfiltered = cv.normalize(bilateralfiltered, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


print("3x3 Box Filter PSNR: ", cv.PSNR(b3, lena_grayscale_hq))
print("5x5 Box Filter PSNR:",cv.PSNR(b5, lena_grayscale_hq))
print("3x3 Gaussian PSNR:",cv.PSNR(g3, lena_grayscale_hq))
print("5x5 Gaussian PSNR:",cv.PSNR(g5, lena_grayscale_hq))
print("Adaptive mean filtered PSNR:",cv.PSNR(adap_mean_filtered, lena_grayscale_hq))
print("Bilateral Filtered PSNR:",cv.PSNR(bilateralfiltered, lena_grayscale_hq))

#2. Soru
print("------------------ 2nd Question ------------------")

noisyImage_Gaussian_01 = cv.imread("noisyImage_Gaussian_01.jpg", cv.IMREAD_GRAYSCALE)

normalized_noisyImage_Gaussian_01 = np.single(noisyImage_Gaussian_01)
normalized_noisyImage_Gaussian_01 = cv.normalize(normalized_noisyImage_Gaussian_01, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

b3_01 = cv.blur(normalized_noisyImage_Gaussian_01, (3,3), borderType=cv.BORDER_REPLICATE)
b3_01 = cv.normalize(b3_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

b5_01 = cv.blur(normalized_noisyImage_Gaussian_01, (5,5), borderType=cv.BORDER_REPLICATE)
b5_01 = cv.normalize(b5_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g3_01 = cv.GaussianBlur(normalized_noisyImage_Gaussian_01,(3,3),0,borderType = cv.BORDER_REPLICATE)
g3_01 = cv.normalize(g3_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

g5_01 = cv.GaussianBlur(normalized_noisyImage_Gaussian_01,(5,5),0,borderType = cv.BORDER_REPLICATE)
g5_01 = cv.normalize(g5_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

adap_mean_filtered_01 = adaptive_mean_filter(normalized_noisyImage_Gaussian_01,0.0009)

bilateralfiltered_01 = cv.bilateralFilter(normalized_noisyImage_Gaussian_01, 3, 0.1, 1, borderType = cv.BORDER_REPLICATE)
bilateralfiltered_01 = cv.normalize(bilateralfiltered_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

print("3x3 Box Filter PSNR: ",cv.PSNR(b3_01, lena_grayscale_hq))
print("5x5 Box Filter PSNR:",cv.PSNR(b5_01, lena_grayscale_hq))
print("3x3 Gaussian PSNR:",cv.PSNR(g3_01, lena_grayscale_hq))
print("5x5 Gaussian PSNR:",cv.PSNR(g5_01, lena_grayscale_hq))
print("Adaptive mean filtered PSNR:",cv.PSNR(adap_mean_filtered_01, lena_grayscale_hq))
print("Bilateral Filtered PSNR:",cv.PSNR(bilateralfiltered_01, lena_grayscale_hq))


#3. Soru
print("------------------ 3rd Question ------------------")


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

cv_output_01 = cv.bilateralFilter(normalized_noisyImage_Gaussian_01, 5, 0.1, 1, borderType = cv.BORDER_REPLICATE)
cv_output_01 = cv.normalize(cv_output_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

cv_output = cv.bilateralFilter(normalized_noisyImage_Gaussian, 5, 3, 0.9, borderType = cv.BORDER_REPLICATE)
cv_output = cv.normalize(cv_output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

my_output_01 = bilateral_filter(normalized_noisyImage_Gaussian_01,5, 0.1, 1)
my_output_01 = cv.normalize(my_output_01, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

my_output = bilateral_filter(normalized_noisyImage_Gaussian,5, 3, 0.9)
my_output = cv.normalize(my_output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)


cv.imshow("Bilateral filtered noisyImage_Gaussian_01 output(OpenCV)",cv_output_01.astype(np.uint8))

cv.imshow("Bilateral filtered noisyImage_Gaussian_01 output(My Filter)",my_output_01.astype(np.uint8))

cv.imshow("Bilateral filtered noisyImage_Gaussian output(OpenCV)",cv_output.astype(np.uint8))

cv.imshow("Bilateral filtered noisyImage_Gaussian output(My Filter)",my_output.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()

abs_img_01 = abs(my_output_01.astype(float)-cv_output_01.astype(float))
print("[noisyImage_Gaussian_01] Max pixel intensity difference:",round(abs_img_01.max()))


abs_img = abs(my_output.astype(float)-cv_output.astype(float))
print("[noisyImage_Gaussian] Max pixel intensity difference:",round(abs_img.max()))


print("For (noisyImage_Gaussian) my bilateral filter PSNR:",cv.PSNR(my_output.astype(np.uint8), lena_grayscale_hq))
print("For (noisyImage_Gaussian) OpenCV PSNR:",cv.PSNR(cv_output.astype(np.uint8), lena_grayscale_hq))

print("For (noisyImage_Gaussian_01) my bilateral filter PSNR:",cv.PSNR(my_output_01.astype(np.uint8), lena_grayscale_hq))
print("For (noisyImage_Gaussian_01) OpenCV PSNR:",cv.PSNR(cv_output_01.astype(np.uint8), lena_grayscale_hq))