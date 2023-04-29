import cv2 as cv
import numpy as np
import glob

# 标定板的大小，标定板内角点的个数
CHCKERBOARD = (9, 6)
# 角点优化，迭代的终止条件,一个是角点优化的最大迭代次数，另一个是角点的移动位移小于则终止
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

# 定义标定板在真实世界的坐标
# 创建一个向量来保存每张图片中角点的3D坐标
objpoints = []
# 创建一个向量来保存每张图片中角点的2D坐标
imgpoints = []
# 定义3D坐标: [row,col,z]
objp = np.zeros((1, CHCKERBOARD[0] * CHCKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHCKERBOARD[0], 0:CHCKERBOARD[1]].T.reshape(-1, 2)

# 提取不同角度拍摄的图片
images = glob.glob('C:/Users/25290/Videos/depth/eval/*.jpg')
images = sorted(images)
for i, fname in enumerate(images):
    img = cv.imread(fname)  # 读取图片
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # RGB转换成灰度图

# 计算标定板的角点的2D坐标
# 寻找角点坐标，如果找到ret返回True, corners:[col,row], 原点在左上角
ret, corners = cv.findChessboardCorners(gray, CHCKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH +
                                        cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
if ret == True:
    objpoints.append(objp)
    # 调用cornerSubpix对2D角点坐标位置进行优化
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # 绘制寻找到的角点，从红色开始绘制，紫色结束
    img = cv.drawChessboardCorners(img, CHCKERBOARD, corners2, ret)
    cv.imshow(fname + 'succeed', img)
else:
    print(f'第{i}张图，{fname}未发现足够角点')
    cv.imshow(fname + 'failed', img)
cv.waitKey(0)
cv.destroyAllWindows()

h, w = img.shape[:2]
# 相机标定
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape
[::-1], None, None)
print('相机内参： ')  # [[fx,0,cx],[0,fy,cy],[0,0,1]]
print(mtx, '\n')
print('畸变参数： ')  # k1,k2,p1,p2,k3
print(dist, '\n')
print('旋转矩阵： ')
print(rvecs, '\n')
print('平移矩阵： ')
print(tvecs, '\n')

# 去畸变
# 要先用getOptimalNewCameraMatrix来获取矫正后的有效像素面积，以及对参数进行优化
img = cv.imread('C:/Users/25290/Videos/depth/eval/0000000001.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 使用undistort矫正图像
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# 图片裁剪
x, y, w, h = roi
dst2 = dst[y:y + h, x:x + w]  # x宽，y高
print(f'ROI: x:{x},y:{y},w:{w},h:{h}')
cv.imshow('original', img)
cv.imshow('image_undistorted1', dst)
cv.imshow('image_undistorted1 ROI', dst2)
cv.waitKey(0)
cv.destroyAllWindows()

# 使用remapping
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h),
                                        5)
dst3 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
cv.imshow('remap_method', dst3)
cv.waitKey(0)
cv.destroyAllWindows()

# 计算重投影误差：
# 计算重投影误差来评估相机标定的结果，将3D坐标投影到成像平面
# 投影得到2D点后，便可以利用检测到的二维坐标角点计算他们之间的均方误差
mean_error = 0
for i in range(len(objpoints)):
    # 使用内外惨和畸变参数对点进行重投影
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx,
                                     dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    # L2范数，均方误差
    mean_error += error

mean_error /= len(objpoints)
print('total error: {}'.format(mean_error))