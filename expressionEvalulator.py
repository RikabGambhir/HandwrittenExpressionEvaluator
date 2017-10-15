import inference
from keras.models import load_model
from keras.models import Sequential


image = input('Image file: ')

model = Sequential()
model = load_model('cnn_mnist')

while(True):
    expression = inference.evalulate(model, image)
    print(expression)
    print(eval(expression))
    image = input('Image file: ')
    if (image == '0'):
        break;
