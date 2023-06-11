import os
import math
import random
import sys

import cv2
import numpy as np


# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray


# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name, img):
    cv2.imshow(window_name, img)


# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_data(dname):
    imgs = []
    # for fname in reversed(os.listdir(dname)):
    for fname in os.listdir(dname):
        imgs.append(read_img(os.path.join(dname, fname)))

    return imgs


def draw_matches(matches):
    img = cv2.copyMakeBorder(
        dst_img, 0, src_img.shape[0] - dst_img.shape[0], 0, 0, 0)

    (hl, wl) = src_img.shape[:2]
    (hr, wr) = img.shape[:2]
    vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
    vis[0:hl, 0:wl] = src_img
    vis[0:hr, wl:] = img
    for (i, j) in matches:
        pos_l = src_kp[i].pt
        pos_l = (int(pos_l[0]), int(pos_l[1]))
        pos_r = dst_kp[j].pt[0] + wl, dst_kp[j].pt[1]
        pos_r = (int(pos_r[0]), int(pos_r[1]))
        cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
        cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
        cv2.line(vis, pos_l, pos_r, (255, 255, 0), 1)

    cv2.imshow("match", vis)
    im_show()


def feature_matching(src_des, dst_des):
    print("feature matching...")
    matches = []
    for i in range(src_des.shape[0]):
        l2 = []
        for j in range(dst_des.shape[0]):
            # compute l2 distance of the descriptor pair
            l2.append(np.linalg.norm(src_des[i] - dst_des[j]))
        # get the 2 nearest neighbors
        p1, p2 = np.argpartition(l2, 2)[:2]
        # Lowe's ratio test
        if l2[p1] < 0.75 * l2[p2]:
            matches.append((i, p1))

    print(f"{len(matches)}/{src_des.shape[0]} matches")
    return matches


def solve_homography(src_pts, dst_pts):
    # given 4 src and dst point pairs, we can compute the homography
    A = []
    for i in range(4):
        A.append([-src_pts[i, 0], -src_pts[i, 1], -1, 0, 0, 0, src_pts[i, 0]
                  * dst_pts[i, 0], src_pts[i, 1] * dst_pts[i, 0], dst_pts[i, 0],])
        A.append([0, 0, 0, -src_pts[i, 0], -src_pts[i, 1], -1, src_pts[i, 0]
                  * dst_pts[i, 1], src_pts[i, 1] * dst_pts[i, 1], dst_pts[i, 1],])

    # solve the linear system using SVD decomposition
    u, s, vt = np.linalg.svd(A)
    H = vt[-1].reshape((3, 3))
    H /= H[2, 2]
    return H


def ransac(matches):
    print("compute homography...")
    size = len(matches)
    src_pts = np.float32([src_kp[i].pt for i, _ in matches])
    src_pts = np.hstack([src_pts, np.ones((size, 1))])
    dst_pts = np.float32([dst_kp[j].pt for _, j in matches])
    dst_pts = np.hstack([dst_pts, np.ones((size, 1))])
    # RANSAC
    threshold = 5.0
    best_cnt = 0
    best_H = None
    n_iters = 20000  # sample 10000 times
    for iter in range(n_iters):
        # sample 4 src and dst pairs
        sample_idx = np.random.choice(size, 4, replace=False)
        # solve the homography using the 4 pairs
        H = solve_homography(src_pts[sample_idx], dst_pts[sample_idx])
        # transform src points to dst cooridates
        dstCoor = H @ src_pts.T
        # normalize
        dstCoor /= dstCoor[2]
        # compute l2 distance between the transformed src points and dst points
        d = np.linalg.norm(dstCoor.T - dst_pts, axis=1)
        # two points match if the distance < threshold
        cnt = np.count_nonzero(d < threshold)
        # H is the best estimation if the points lies within the threshold is larger than ever
        if cnt > best_cnt:
            best_cnt = cnt
            best_H = H
    return best_H


def trim(img):
    if len(img.shape) == 3:
        # trim the black part of the img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # find the largest contours in the image
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max(contours, key=cv2.contourArea)
    # the bounding rectangle of the largest contours is the final region we want
    x, y, w, h = cv2.boundingRect(max_area)
    return img[y: y + h, x: x + w]


def blend(img1, img2, mask):
    # get the left and right bound of the mask
    left = mask.shape[1] - 1
    right = 0
    for col in range(mask.shape[1]):
        if np.count_nonzero(mask[:, col]):
            left = min(left, col)
            right = max(right, col)

    # the blending range has constant width w
    w = 50

    # the middle line of the overlapped region
    mid = (left + right) / 2
    left = mid - w
    right = mid + w

    # linear blending
    res = np.zeros(img1.shape, dtype=np.uint8)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] and x > left and x < right:
                r = (x - left) / (right - left)
                res[y, x] = (1 - r) * img1[y, x] + r * img2[y, x]
            elif np.count_nonzero(img1[y, x]) and x < mid:
                res[y, x] = img1[y, x]
            elif np.count_nonzero(img2[y, x]) and x > mid:
                res[y, x] = img2[y, x]
            elif np.count_nonzero(img2[y, x]):
                res[y, x] = img2[y, x]
            else:
                res[y, x] = img1[y, x]
    return res


def bilinear_interplate(u, v, img):
    u_f = max(math.floor(u), 0)
    v_f = max(math.floor(v), 0)
    u_c = min(math.ceil(u), img.shape[1] - 1)
    v_c = min(math.ceil(v), img.shape[0] - 1)
    uw = u - u_f
    vw = v - v_f
    return (1 - vw) * ((1 - uw) * img[v_f, u_f] + uw * img[v_f, u_c]) \
        + vw * ((1 - uw) * img[v_c, u_f] + uw * img[v_c, u_c])


def stitch(dst_img, src_img, H):
    print("stitching...")
    dy = src_img.shape[0]
    dx = src_img.shape[1]
    # affine transform matrix, to make space for the src image
    A = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
    # inverse warping
    M = np.linalg.inv(A @ H)
    size = (dst_img.shape[1] + dx * 2, dst_img.shape[0] + dy * 2)
    # affine translation for dst image
    dst_warp = cv2.warpPerspective(dst_img, A, size)
    # use inverse warping for src image to avoid aliasing
    src_warp = np.zeros(dst_warp.shape, dtype=np.uint8)
    for y in range(src_warp.shape[0]):
        for x in range(src_warp.shape[1]):
            u, v, w = M @ np.array([x, y, 1])
            u, v = u / w, v / w
            if v >= 0 and u >= 0 and v < src_img.shape[0] and u < src_img.shape[1]:
                # u, v can be float number, do interpolation to get a more accurate value
                src_warp[y, x] = bilinear_interplate(u, v, src_img)

    # get the overlap mask
    mask1 = np.where(cv2.cvtColor(dst_warp, cv2.COLOR_BGR2GRAY), 1, 0)
    mask2 = np.where(cv2.cvtColor(src_warp, cv2.COLOR_BGR2GRAY), 1, 0)
    mask = cv2.bitwise_and(mask1, mask2)

    # blending
    res = blend(dst_warp, src_warp, mask)
    return res


if __name__ == "__main__":
    # folder = "./baseline"
    folder = "./bonus"
    imgs = read_data(folder)

    sift = cv2.SIFT_create()
    dst_img, dst_gray = imgs[0]
    for src_img, src_gray in imgs[1:]:
        dst_kp, dst_des = sift.detectAndCompute(dst_gray, None)
        src_kp, src_des = sift.detectAndCompute(src_gray, None)
        matches = feature_matching(src_des, dst_des)
        H = ransac(matches)
        res = stitch(dst_img, src_img, H)
        res = trim(res)
        dst_img = res
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("result", res)
    cv2.imwrite("result.jpg", res)
    im_show()
