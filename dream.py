'''
CSCI 631 Project
DEEP DREAM  CNN Layer Analysis
Link to dataset:https://drive.google.com/open?id=15BPrlXojfeqx9-ADoq0MqFgX_9cgamjn

Contributors:
Pallavi Chandanshive (pvc8661@rit.edu)
Trisha  Malhotra (tpm6421@rit.edu)
cite : Name :HVASS -Labs TensorFlow-Tutorials/14_DeepDream.ipynb , Chipgraner
This program creates a tensorflow session and implements deep dream .
'''

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import tensorflow as tf
import os
import random

'''
Helper for selection of the layer which is required to be enhanced
'''
def selectLayer(layers):
    set = [2, 39, 52,34, 45,53]
    index=random.choice(set)
    layerName = layers[index].split("/")
    print("Layer number selected : " , index)
    print("Selected Layer Name : ", layerName[1])
    return layerName[1]


'''
Helper for generating graph based on the prepreprocessed input
'''
def tensor(t_input,graph_def):
    mean = 117.0
    t_prepcossed = tf.expand_dims(t_input - mean, 0)
    tf.import_graph_def(graph_def, {'input': t_prepcossed})


'''
Helper for displaying the layers of the CNN model
'''
def displayLayers(layers):

    print("Name of Layers in CNN Network : ")
    for index in range(len(layers)):
        print("Layers ", index, layers[index])


'''
Helper for creating session of tensorflow
'''
def sessions(data_dir,model_fn):
    graph = tf.Graph()
    session = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')

    return graph,graph_def,t_input,session

'''
Main method 
'''
def main():
    # steps for using the pre-trained CNN
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data'
    # local_zip_file = '../inception5h'

    # defining tensor flow model
    model_fn = 'tensorflow_inception_graph.pb'

    # Step 2 Creating the tensorflow session and loading the model
    graph, graph_def, t_input,session=sessions(data_dir,model_fn)

    # define input tensor
    tensor(t_input,graph_def)

    #  layers
    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    displayLayers(layers)

    feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

    print('Number of layers : ', len(layers))
    print("Total number of feature channels : ", sum(feature_nums))


    # Step 3 - Pick layer to enhance our image
    print("Layer for enhancing : ")
    layer = selectLayer(layers)
    # layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139

    image0 = PIL.Image.open('/images/DEEPDREAM/FINAL/UNADJUSTEDNONRAW_thumb_b.jpg')
    image0 = np.float32(image0)

    # start with a gray image with a little noise
    img_noise = np.zeros(shape=(224, 224, 3)) + 100.0  # .random.uniform(size=(224, 224, 3)) + 100.0

    '''
    Helper for getting layer output tensor.
    '''
    def T(layer):
        return graph.get_tensor_by_name("import/%s:0" % layer)

    '''
    Helper that transforms TF-graph generating function into a regular one.
    '''
    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

            return wrapper

        return wrap

    '''
         Helper that helps for resizing the image.
    '''
    def resize(img, size):

        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    '''
    Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.
    '''
    def calc_grad_tiled(img, t_grad, tile_size=512):

        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = session.run(t_grad, {t_input: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    ''' 
    Method for ploting the resultant image 
    '''
    def showarray(a,iters):
        a = np.uint8(np.clip(a, 0, 1) * 255)
        plt.imshow(a)
        plt.xlabel('Number of iterations : ' + str(iters))
        plt.title('Layer :  ' + str(53) + " " + str(layer))
        plt.show()

    ''' 
       Method for applying the spliting the image into octave  
    '''
    def render_deepdream(t_obj, image0, iter_n=10, step=1.5, octave_n=4, octvae_scale=1.4):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = image0
        octaves = []
        for _ in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw) / octave_n))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))

        showarray(img / 255.0,iter_n)

    # Step 4 - Apply gradient ascent to that layer
    render_deepdream(T(layer)[:, :, :, 139], image0)


if __name__ == '__main__':
    main();
