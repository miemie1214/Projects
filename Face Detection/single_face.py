import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

def dist_point(center, point, sigma):
    return float(np.sqrt(((center[0] - point[0]) / sigma[0]) ** 2 + ((center[1] - point[1]) / sigma[1]) ** 2 + (
                (center[2] - point[2]) / sigma[2]) ** 2))

def yf2473_Yiyang_kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k):
    """
        parameters:imgPath: the path of the image folder. Please use relative pathimg
        Filename: the name of the image filesaved
        ImgPath: the path of the folder you will save the imagesaved
        ImgFilename: the name of the output image
        k: the number of clusters of the k-means function
        function: using k-means to segment the image and save the result to an image with  a bounding box
    """

    # Read Image
    img_file = os.path.join(imgPath, imgFilename)
    img1 = cv2.imread(img_file)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    sigma = np.array([np.std(img1_color[:, :, 0]), np.std(img1_color[:, :, 1]), np.std(img1_color[:, :, 2])])
    # Initialize centers
    np.random.seed(100)
    center_x = np.random.randint(0, img1_color.shape[0], size=k)
    center_y = np.random.randint(0, img1_color.shape[1], size=k)
    centers = np.zeros((k, 3))

    for i in range(0, k):
        p = img1_color[center_x[i]][center_y[i]].reshape(1, 3)
        centers[i] = p

    #print(centers)
    center_prev = np.zeros(centers.shape)
    result = np.zeros(img1_color[:, :, 0].shape)
    error = 1
    while error != 0:
        for i in range(0, img1_color.shape[0]):
            for j in range(0, img1_color.shape[1]):
                tmp = []
                for z in range(0, k):
                    tmp.append(dist_point(centers[z], img1_color[i][j], sigma))

                result[i][j] = np.argmin(tmp)
        center_prev = copy.deepcopy(centers)

        for z in range(0, k):
            temp = np.zeros((1, 3))
            for i in range(0, img1_color.shape[0]):
                for j in range(0, img1_color.shape[1]):
                    if result[i][j] == z:
                        temp = np.vstack((temp, img1_color[i][j]))

            temp = np.delete(temp, (0), axis=0)
            centers[z] = np.mean(temp, axis=0)

            error = 0

            for y in range(0, k):
                error = error + dist_point(centers[y], center_prev[y], sigma)

        result = result.astype(np.uint8)
    # threshold image
    ret, threshed_img = cv2.threshold(result, 3, 5, cv2.THRESH_BINARY)
    # find contours and get the external one
    threshed_img = cv2.convertScaleAbs(threshed_img)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a red rectangle to visualize the bounding rect of mean color pixel value > 120 and width > 20
        if np.mean(img1_color[y:y + h][x:x + w]) > 120 and w > 20:
            cv2.rectangle(img1_color, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            #print(x,y,w,h)
    #print(len(contours))
    plt.figure()
    plt.imshow(img1_color, cmap="gray")
    plt.show()

    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(savedImgPath, savedImgFilename), img1_color)
    cv2.waitKey(0)

if __name__ == "__main__":
    imgPath = "../5293HW1"
    imgFilename = "face_d2.jpg"
    savedImgPath = r'../5293HW1'
    savedImgFilename = "face_d2_face.jpg"
    k = 6
    yf2473_Yiyang_kmeans(imgPath, imgFilename, savedImgPath, savedImgFilename, k)

