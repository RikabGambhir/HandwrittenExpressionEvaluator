from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')


def generate(symbols):
    for j in range(len(symbols)):
        for i in range(23):
            if i == 0:
                i = i + 1
            img = load_img(str(j) + '/' + symbols[j] + '_' + str(i) + ".png")
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            k = 0
            for batch in datagen.flow(x, batch_size = 1, save_to_dir= str(j), save_prefix = str(j), save_format='png'):
                k += 1
                if k>499:
                    break





# for l in range(23):
#     if l == 0:
#         l = l + 1
#     img = load_img('10/+_' + str(l) + ".png")
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     k = 0
#     for batch in datagen.flow(x, batch_size = 1, save_to_dir= '10', save_prefix = '+', save_format='png'):
#         k += 1
#         if k>499:
#             break
#
# for m in range(23):
#     if m == 0:
#         m = m + 1
#     img = load_img('11/-_' + str(m) + ".png")
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     k = 0
#     for batch in datagen.flow(x, batch_size = 1, save_to_dir= '11', save_prefix = '-', save_format='png'):
#         k += 1
#         if k>499:
#             break

# for n in range(23):
#     if n == 0:
#         n = n + 1
#     img = load_img('12/*_' + str(n) + ".png")
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     k = 0
#     for batch in datagen.flow(x, batch_size = 1, save_to_dir= '12', save_prefix = '*', save_format='png'):
#         k += 1
#         if k>499:
#             break
#
# for o in range(23):
#     if o == 0:
#         o = o + 1
#     img = load_img('13/_' + str(o) + ".png")
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     k = 0
#     for batch in datagen.flow(x, batch_size = 1, save_to_dir= '13', save_prefix = 'div', save_format='png'):
#         k += 1
#         if k>499:
#             break

# for i in range(10):
#     if i == 0:
#         i = i + 1
#     img = load_img('decimal/decimal' + str(i) + ".png")
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     k = 0
#     for batch in datagen.flow(x, batch_size = 1, save_to_dir= '10', save_prefix = '+', save_format='png'):
#         k += 1
#         if k>499:
#             break
