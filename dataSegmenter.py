import segmentation
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32')
        if img is not None:
            images.append(img)
    return images

def segmentData(input_directory, classes):

    symbols = load_images_from_folder(input_directory)

    for i in range(len(symbols)):
        images = segmentation.segment(symbols[i])

        for class in range(len(classes)):
            cv2.imwrite(classes[class] + '/' + classes[class] + '_' + str(i), images[class])
