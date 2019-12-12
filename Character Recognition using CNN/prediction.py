from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('models/CNN.h5')
classes = '0123456789'

for i in range(20):
    im1 = Image.open('test_images/' + str(i) + '.png').convert("L")
    im1 = im1.resize((28,28))
    im2arr = np.array(im1)
    im2arr = im2arr.reshape(1,1,28,28)
    #im1.show()
    #Testing on input images
    y = model.predict(im2arr)
    print(y,' Predicted digit : ', classes[np.argmax(y)])
    
