import numpy as np
import cv2


a = np.zeros((1, 28, 28), dtype=np.uint8).transpose(2, 1, 0)


cv2.imshow("test", a)
cv2.imwrite("a.jpg", a)
cv2.waitKey(0)
cv2.destroyAllWindows()