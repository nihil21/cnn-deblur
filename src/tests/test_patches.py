import tensorflow as tf


def _extract_patches(img):
    img = tf.reshape(img, (1, 720, 1280, 3))
    # from the single image extract the 12 patches
    # with input shape 720x1280 each patch has shape 240x320
    patches = tf.image.extract_patches(
        images=img,
        sizes=[1, 240, 320, 1],
        strides=[1, 240, 320, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches = tf.reshape(patches, (12, 240, 320, 3))

    return patches


def _save_img(img, filename):
    img = tf.image.convert_image_dtype(img, tf.uint8)
    enc = tf.image.encode_png(img)
    tf.io.write_file(tf.constant(filename), enc)


def _load_img(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img


# load image
image = _load_img("/home/tree/Documents/keras-tutorial/test_patches/parrot.png")

patches = _extract_patches(image)

# save patches
#i = 0
#for p in patches:
#    i += 1
#    _save_img(p, "/home/tree/Documents/keras-tutorial/test_patches/patches/parrot" + str(i) + ".png")

# from patches to the original image
COLS = 4
ROWS = 3

restored_image = []
for i in range(0, ROWS):
    row = []
    for j in range(0, COLS):
        row.append(patches[i*4+j])

    restored_row = tf.concat(row, axis=1)
    restored_image.append(restored_row)

full_image = tf.concat(restored_image, axis=0)
_save_img(full_image, "/home/tree/Documents/keras-tutorial/test_patches/prova.png")