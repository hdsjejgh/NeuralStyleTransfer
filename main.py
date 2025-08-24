import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

VGG = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
VGG.trainable = False

ALPHA = 10**0
BETA = 10**2 * ALPHA

content_path = "images/Lunch_atop_a_Skyscraper_remastered_and_colored.png"
style_path = "styles/vaifjefpwe.jpg"

content = cv.imread(content_path)
content_shape = content.shape
content = cv.resize(content,(224,224))
content = preprocess_input(content)
content = tf.constant(content.reshape((1,224,224,3)))

style = cv.imread(style_path)
style_shape = style.shape
style = cv.resize(style,(224,224))
style = preprocess_input(style)
style = tf.constant(style.reshape((1,224,224,3)))



layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'block4_conv2']
outputs = [VGG.get_layer(layer).output for layer in layers]
activation_model = tf.keras.Model(inputs=VGG.inputs, outputs=outputs)

print(activation_model.summary())


gen = tf.Variable(content, dtype=tf.float32)


def content_loss(cont,inp):
    outp1,outp2 = activation_model(cont),activation_model(inp)

    m,nh,nw,nc = outp2[5].shape

    cont = tf.reshape(outp1[5],[m,-1,nc])
    inp = tf.reshape(outp2[5], [m,-1,nc])

    diff = tf.subtract(cont,inp)
    square = tf.square(diff)
    cost = tf.reduce_sum(square)

    return cost/(4*nh*nw*nc)

def gram(A):
    A = tf.reshape(A, [-1, A.shape[-1]])
    return tf.matmul(A, A, transpose_a=True)


def style_layer_loss(sty,inp):
    m, nh, nw, nc = sty.shape
    sb, ib = tf.reshape(sty, [-1, nc]), tf.reshape(inp, [-1, nc])
    gsb,gib = gram(sb),gram(ib)
    lc = tf.subtract(gsb, gib)
    lc = tf.square(lc)
    lc = tf.reduce_sum(lc)
    lc = lc / ((2 * nc * nh * nw) ** 2)

    return lc

def style_loss(sty,inp):
    outp1, outp2 = activation_model(sty), activation_model(inp)

    sb1,sb2,sb3,sb4,sb5,_ = outp1
    ib1, ib2, ib3, ib4, ib5, _ = outp2

    l1c,l2c,l3c,l4c,l5c = style_layer_loss(sb1,ib1),style_layer_loss(sb2,ib2),style_layer_loss(sb3,ib3),style_layer_loss(sb4,ib4),style_layer_loss(sb5,ib5)

    return (l1c + l2c + l3c + l4c + l5c)/5

def total_cost(sty,cont,inp):
    return ALPHA*content_loss(cont,inp) + BETA*style_loss(sty,inp)

print(content_shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        J = total_cost(style, content, gen)
    grad = tape.gradient(J, gen)
    optimizer.apply_gradients([(grad, gen)])
    gen.assign(tf.clip_by_value(gen, -128.0, 128.0))  # keep in VGG range
    return J

EPOCHS = 1200
for i in range(EPOCHS):
    loss = train_step()
    if i % 10 == 0:
        print(f"Epoch {i}: loss = {loss.numpy()}")

out = gen.numpy()[0]
out = out + [103.939, 116.779, 123.68]
out = np.clip(out, 0, 255).astype(np.uint8)
out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
out = cv.resize(out,(content_shape[1]//2,content_shape[0]//2))
while cv.waitKey(0):
    cv.imshow("display",out)
