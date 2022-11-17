import numpy as np
import cv2 as cv

def stitch_2_img(extractor=0, matcher=0):
    """    Stitches 2 images which are given by user by input    """
    import numpy as np
    import cv2 as cv

    print("left img name: ",end="")
    left_img_name = input()
    print("\nright img name: ",end="")
    right_img_name = input()

    img1 = cv.imread(left_img_name, cv.IMREAD_GRAYSCALE)
    img1 = img1.astype(np.uint8)
    
    img2 = cv.imread(right_img_name, cv.IMREAD_GRAYSCALE)
    img2 = img2.astype(np.uint8)
    ############################################################################################################
    img1 = cv.equalizeHist(img1)
    img2 = cv.equalizeHist(img2)
    imgs = [img2, img1]
    ############################################################################################################
    if extractor == 0: # SIFT
        print("using SIFT...")
        key_point_feature_extractor1 = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
        kp1, des1 = key_point_feature_extractor1.detectAndCompute(imgs[0],None)

        key_point_feature_extractor2 = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
        kp2, des2 = key_point_feature_extractor2.detectAndCompute(imgs[1],None)

    if extractor == 1: # ORB
        print("using ORB...")
        key_point_feature_extractor1 = cv.ORB_create(nfeatures=1000, edgeThreshold=10, patchSize=10)
        kp1 = key_point_feature_extractor1.detect(imgs[0], None)
        (kp1, des1) = key_point_feature_extractor1.compute(imgs[0], kp1)
        des1 = des1.astype(np.float32)

        key_point_feature_extractor2 = cv.ORB_create(nfeatures=1000, edgeThreshold=10, patchSize=10)
        kp2 = key_point_feature_extractor2.detect(imgs[1], None)
        (kp2, des2) = key_point_feature_extractor2.compute(imgs[1], kp2)
        des2 = des2.astype(np.float32)

    if extractor == 2: # BRIEF
        print("using BRIEF...")
        star1 = cv.xfeatures2d.StarDetector_create()
        key_point_feature_extractor1 = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64, use_orientation = True)
        kp1 = star1.detect(imgs[0],None)
        kp1, des1 = key_point_feature_extractor1.compute(imgs[0], kp1)
        des1 = des1.astype(np.float32)
        
        star2 = cv.xfeatures2d.StarDetector_create()
        key_point_feature_extractor2 = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64, use_orientation = True)
        kp2 = star2.detect(imgs[1],None)
        kp2, des2 = key_point_feature_extractor2.compute(imgs[1], kp2)
        des2 = des2.astype(np.float32)


    if matcher == 0: #FLANN
        print("using FLANN...")
        FLANN_INDEX_KDTREE = 0
        index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
        search_params = {'checks': 50}
        key_point_matcher = cv.FlannBasedMatcher(index_params,search_params)

    if matcher == 1: # BRUTEFORCE
        print("using BRUTEFORCE...")
        key_point_matcher = cv.DescriptorMatcher_create(2)
    ############################################################################################################
    print("kp1: ",len(kp1),"    kp2: ",len(kp2), sep="")
    # finding 2 best match for each point, if there is a good amount of peek betweeen first and second match: it is a good match
    matches = key_point_matcher.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good), "good match finded")
    ############################################################################################################    
    if len(good)>4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found.")
        return
    ############################################################################################################  
    stitched_img = cv.warpPerspective(imgs[0], M, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))
    stitched_img[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = imgs[1]
    stitched_img = cv.equalizeHist(stitched_img)
    ############################################################################################################  
    zeros = np.zeros(imgs[0].shape, dtype=np.uint8)
    zeros[:,-1] = 1
    boundaries = cv.warpPerspective(zeros, M, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))
    stitch_cut = np.nonzero(boundaries)[1].min()
    stitched_img = stitched_img[:,:stitch_cut]
    ############################################################################################################  

    cv.namedWindow("left img", cv.WINDOW_NORMAL)
    cv.resizeWindow("left img", 1200, 720)
    cv.imshow("left img", imgs[1].astype(np.uint8))

    cv.namedWindow("right img", cv.WINDOW_NORMAL)
    cv.resizeWindow("right img", 1200, 720)
    cv.imshow("right img", imgs[0].astype(np.uint8))

    cv.namedWindow("stitched_img", cv.WINDOW_NORMAL)
    cv.resizeWindow("stitched_img", 1200, 720)
    cv.imshow("stitched_img", stitched_img.astype(np.uint8))

    cv.waitKey(0)
    cv.destroyAllWindows()


    return stitched_img
print("Stitching 2 given image:(press enter to continue)", end="")
input()

extractor = int(input("select extractor\n0:SIFT, 1:ORB, 2:BRIEF\n"))
matcher = int(input("select matcher\n0:FLANN, 1:BRUTEFORCE\n"))

stitched_img = stitch_2_img(extractor, matcher)
#----------######----------######----------######----------######----------######----------######----------######----------#
def stitch_2_given_img(img1, img2, extractor=0, matcher=0):
    """    Stitches 2 images that are given in order from left to right in parameters    """
    import numpy as np
    import cv2 as cv

    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    img1 = cv.equalizeHist(img1)
    img2 = cv.equalizeHist(img2)
    imgs = [img2, img1]
    ############################################################################################################
    if extractor == 0: # SIFT
        print("using SIFT...")
        key_point_feature_extractor1 = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
        kp1, des1 = key_point_feature_extractor1.detectAndCompute(imgs[0],None)

        key_point_feature_extractor2 = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
        kp2, des2 = key_point_feature_extractor2.detectAndCompute(imgs[1],None)

    if extractor == 1: # ORB
        print("using ORB...")
        key_point_feature_extractor1 = cv.ORB_create(nfeatures=1000, edgeThreshold=10, patchSize=10)
        kp1 = key_point_feature_extractor1.detect(imgs[0], None)
        (kp1, des1) = key_point_feature_extractor1.compute(imgs[0], kp1)
        des1 = des1.astype(np.float32)

        key_point_feature_extractor2 = cv.ORB_create(nfeatures=1000, edgeThreshold=10, patchSize=10)
        kp2 = key_point_feature_extractor2.detect(imgs[1], None)
        (kp2, des2) = key_point_feature_extractor2.compute(imgs[1], kp2)
        des2 = des2.astype(np.float32)

    if extractor == 2: # BRIEF
        print("using BRIEF...")
        star1 = cv.xfeatures2d.StarDetector_create()
        key_point_feature_extractor1 = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64, use_orientation = True)
        kp1 = star1.detect(imgs[0],None)
        kp1, des1 = key_point_feature_extractor1.compute(imgs[0], kp1)
        des1 = des1.astype(np.float32)
        
        star2 = cv.xfeatures2d.StarDetector_create()
        key_point_feature_extractor2 = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=64, use_orientation = True)
        kp2 = star2.detect(imgs[1],None)
        kp2, des2 = key_point_feature_extractor2.compute(imgs[1], kp2)
        des2 = des2.astype(np.float32)


    if matcher == 0: #FLANN
        print("using FLANN...")
        FLANN_INDEX_KDTREE = 0
        index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
        search_params = {'checks': 50}
        key_point_matcher = cv.FlannBasedMatcher(index_params,search_params)

    if matcher == 1: # BRUTEFORCE
        print("using BRUTEFORCE...")
        key_point_matcher = cv.DescriptorMatcher_create(2)
    ############################################################################################################
    print("kp1: ",len(kp1),"    kp2: ",len(kp2), sep="")
    # finding 2 best match for each point, if there is a good amount of peek betweeen first and second match: it is a good match
    matches = key_point_matcher.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good), "good match finded")
    ############################################################################################################    
    if len(good)>4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), 4) )
        return
    ############################################################################################################  
    stitched_img = cv.warpPerspective(imgs[0], M, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))
    stitched_img[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = imgs[1]
    stitched_img = cv.equalizeHist(stitched_img)
    ############################################################################################################  
    try:
        zeros = np.zeros(imgs[0].shape, dtype=np.uint8)
        zeros[:,-1] = 1
        boundaries = cv.warpPerspective(zeros, M, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))
        stitch_cut = np.nonzero(boundaries)[1].min()
        stitched_img = stitched_img[:,:stitch_cut]
    except:
        pass
    ############################################################################################################  
    return stitched_img
def stitch_more_than_2_img_left_to_right(imgs, extractor=0, matcher=0):
    """    Stitches list of images that are in order from left to right in parameters    """
    while len(imgs)>1:
        stitched_img = stitch_2_given_img(imgs[0], imgs[1], extractor, matcher)
        imgs[1] = stitched_img
        del imgs[0] 
    return stitched_img
print("Stitching more than 2 image from left to right:(press enter to continue)", end="")
input()
print("give image names left to right(q to quit)")

imgs = []
while True:
    img_name = input()
    if img_name == "q":
        break
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    img = img.astype(np.uint8)
    imgs.append(img)

extractor = int(input("select extractor\n0:SIFT, 1:ORB, 2:BRIEF\n"))
matcher = int(input("select matcher\n0:FLANN, 1:BRUTEFORCE\n"))

stitched_img = stitch_more_than_2_img_left_to_right(imgs, extractor, matcher)
cv.namedWindow("stitched_img", cv.WINDOW_NORMAL)
cv.resizeWindow("stitched_img", 1200, 720)
cv.imshow("stitched_img", stitched_img)
cv.waitKey(0)
cv.destroyAllWindows()
#----------######----------######----------######----------######----------######----------######----------######----------#
def sort_imgs_with_good_matches(imgs):
    """    Sortes randomly ordered images    """
    print("sorting images...")
    pairs = []
    for i in range(len(imgs)):
        for j in range(i+1,len(imgs)):    
            imgg1 = cv.equalizeHist(imgs[i])
            imgg2 = cv.equalizeHist(imgs[j])

            key_point_feature_extractor1L = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
            kp1L, des1L = key_point_feature_extractor1L.detectAndCompute(imgg1[:,:int(imgg1.shape[1]*0.75)],None)

            key_point_feature_extractor2R = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
            kp2R, des2R = key_point_feature_extractor2R.detectAndCompute(imgg2[:,int(imgg2.shape[1]*0.25):],None)
            
            key_point_feature_extractor1R = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
            kp1R, des1R = key_point_feature_extractor1R.detectAndCompute(imgg1[:,int(imgg1.shape[1]*0.25):],None)

            key_point_feature_extractor2L = cv.xfeatures2d.SIFT_create(contrastThreshold = 0.20, edgeThreshold = 25)
            kp2L, des2L = key_point_feature_extractor2L.detectAndCompute(imgg2[:,:int(imgg2.shape[1]*0.75)],None)

            FLANN_INDEX_KDTREE = 0
            index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
            search_params = {'checks': 50}
            key_point_matcher = cv.FlannBasedMatcher(index_params,search_params)
            ############################################################################################################
            matches = key_point_matcher.knnMatch(des1L, des2R, k=2)
            goodRL = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    goodRL.append(m)

            matches = key_point_matcher.knnMatch(des1R, des2L, k=2)
            goodLR = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    goodLR.append(m)

            if len(goodLR) > len(goodRL):
                pairs.append((i,j))
            else:
                pairs.append((j,i))
        print(i+1,"out of",len(imgs),"done")
        
    all_values = []
    for p in pairs:
        if p[0] in all_values:
            pass
        else:
            all_values.append(p[0])

        if p[1] in all_values:
            pass
        else:
            all_values.append(p[1])

    left_to_right_dict = {}
    for v in all_values:
        left_to_right_dict[v] = 0
    for pair in pairs:
        if pair[0] in left_to_right_dict.keys():
            left_to_right_dict[pair[0]] = left_to_right_dict[pair[0]] + 1

    sorted_tuples = sorted(left_to_right_dict.items(), key=lambda item: item[1])
    sorted_dict = {len(all_values)-1-v: k for k, v in sorted_tuples}


    sorted_imgs = imgs.copy()
    for k,v in zip(sorted_dict.keys(), sorted_dict.values()):
        sorted_imgs[k] = imgs[v]
    return sorted_imgs
def stitch_more_than_2_img_random(imgs, extractor, matcher):
    sorted_imgs = sort_imgs_with_good_matches(imgs)
    stitched_img = stitch_more_than_2_img_left_to_right(sorted_imgs, extractor, matcher)
    return stitched_img
print("Stitching more than 2 image in random order:(press enter to continue)", end="")
input()
print("give image names in random order(q to quit)")

imgs = []
while True:
    img_name = input()
    if img_name == "q":
        break
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    img = img.astype(np.uint8)
    imgs.append(img)

extractor = int(input("select extractor\n0:SIFT, 1:ORB, 2:BRIEF\n"))
matcher = int(input("select matcher\n0:FLANN, 1:BRUTEFORCE\n"))

stitched_img = stitch_more_than_2_img_random(imgs, extractor, matcher)
cv.namedWindow("stitched_img", cv.WINDOW_NORMAL)
cv.resizeWindow("stitched_img", 1200, 720)
cv.imshow("stitched_img", stitched_img)
cv.waitKey(0)
cv.destroyAllWindows()