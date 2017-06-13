from skimage import io
from PIL import Image
import numpy
PIXEL_DEPTH = 255
def convertToPNG(p_img):
    
    w = p_img.shape[0]
    h = p_img.shape[1]
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(p_img)          
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


for idx in range(1,51):
	name = 'prediction_'+str(idx)+'.png'
	img = io.imread(name, as_grey=True)
	print(img.shape)
	for i,row in enumerate(img):
		for j,col in enumerate(img[i]):
			img[i][j] = 1-img[i][j]
	img_3c = convertToPNG(img)
	Image.fromarray(img_3c).save(name)


