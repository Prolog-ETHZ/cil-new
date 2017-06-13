from PIL import Image, ImageFilter
import numpy as np
if __name__=="__main__":

    im = Image.open("prediction_48.png")
    kernelValues = [-2,-1,0,-1,1,1,0,1,2] #emboss
    kernel_edge_detect1 = np.array([[1.,0.,-1.],
                                [0.,0.,0.],
                                [-1.,0.,1.]])
    kernel_edge_detect2 = np.array([[0.,1.,0.],
                                [1.,-4.,1.],
                                [0.,1.,0.]])
    kernel_edge_detect3 = np.array([[-1.,-1.,-1.],
                                [-1.,8.,-1.],
                                [-1.,-1.,-1.]])
    kernel_sharpen = np.array([[0.,-1.,0.],
                           [-1.,5.,-1.],
                           [0.,-1.,0.]])
    kernel_sharpen2 = np.array([[-1.,-1.,-1.],
                           [-1.,9.,-1.],
                           [-1.,-1.,-1.]])
    kernel_blur = np.array([[1.,1.,1.],
                        [1.,1.,1.],
                        [1.,1.,1.]])
    
    kernel = ImageFilter.Kernel((3,3), kernel_blur.reshape(kernel_blur.shape[0]*kernel_blur.shape[1],-1))
    im2 = im.filter(kernel)
    im2.show()
    print('finish')