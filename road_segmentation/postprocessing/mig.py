from PIL import Image
import matplotlib.image as mpimg
import scipy.misc

predict_data_filename = '../test_set_images';
for i in range(1,50+1):
    name = predict_data_filename+'/test_'+str(i)+'/test_'+str(i)+'.png'
    img = mpimg.imread(name)
    print(img.shape)
    print(name)
    #exit(0)
    #Image.fromarray(img).save('./true_result/' + "prediction_" + str(i) + ".png")
    scipy.misc.imsave('./true_result/' + "prediction_" + str(i) + ".png", img)