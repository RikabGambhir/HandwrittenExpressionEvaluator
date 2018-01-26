import inference
from keras.models import load_model
from keras.models import Sequential


image = "test7.png"


model = Sequential()
model = load_model('cnn_mnist')



while(True):
    expression = inference.evalulate(model, image)
    print(expression)
    try:
        print(eval(expression))
    except:
        print("Failed to evaluate")
    enter = input('Press enter to reload')
    if (image == '0'):
        break
