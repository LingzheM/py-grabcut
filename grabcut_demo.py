import cv2
import numpy as np

# 读取图片
img = cv2.imread('input.jpg')
if img is None:
    raise FileNotFoundError('input.jpg not found!')

# 创建一个初始的mask
mask = np.zeros(img.shape[:2], np.uint8)

# 创建背景模型和前景模型
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 这里rect是你选的区域：(x, y, width, height)
# 例：左上(50,50)，宽200，高300
rect = (300, 100, 600, 700)

# 调用grabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 将所有0和2（背景）变为0，1和3（前景）变为1
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

# 用mask提取前景
img_result = img * mask2[:, :, np.newaxis]

cv2.imwrite('output.png', img_result)
print("保存成功: output.png")
