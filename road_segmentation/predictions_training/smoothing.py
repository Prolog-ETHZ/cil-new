from skimage import io
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
PIXEL_DEPTH = 255
def convertToPNG(p_img):
    
    w = p_img.shape[0]
    h = p_img.shape[1]
    gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
    gt_img8 = img_float_to_uint8(p_img)          
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg
for idx in range(1,51):
	name = 'prediction_'+str(idx)+'.png'
	img = io.imread(name, as_grey=True) #608*608
	
	blur = cv2.medianBlur(convertToPNG(img),21)
	
	#blur = cv2.GaussianBlur(img,(41,41),0)
	#plt.imshow(blur, cmap='gray', interpolation='nearest');
	#blur = convertToPNG(blur)
	Image.fromarray(blur).save(name)
	
	


