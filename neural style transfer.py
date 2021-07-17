import tensorflow as tf
import tensorflow.compat.v1 as tfold
import tensorflow.python.ops.numpy_ops.np_config as np_config
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
np_config.enable_numpy_behavior()
tfold.disable_eager_execution()
tfold.reset_default_graph()

base_model = VGG19(include_top=False, weights="imagenet")

base_model.trainable = False
base_model.summary()

style_layers_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1","block5_conv1"]
content_layers_names = ["block3_conv3"]

style_layers, content_layers = [base_model.get_layer(name=layer) for layer in style_layers_names ], [base_model.get_layer(name=layer) for layer in content_layers_names]

#Defining intermediate layer
import re
def make_intermediate_model(base_model, new_model_name, *args):
     #style_outs, content_outs = [layer.output for layer in args[0]], [layer.output for layer in args[1]]
     
     style_outs = []
     content_outs = []
     stop_network = len(args[0] + args[1])
     new_model_input = base_model.input
     x = new_model_input
     for layer in base_model.layers[1:]:
       
       if len(style_outs + content_outs) == stop_network :
         break
       if re.match(".*pool", layer.name) :
         x = tf.keras.layers.AvgPool2D(pool_size=(2,2), strides=(2,2))(x)
       else:
         x = layer(x)

       if layer in args[0] :
         style_outs.append(x)
       elif layer in args[1]:
         content_outs.append(x)

     new_model_outputs = style_outs + content_outs
     new_model_outputs = [tfold.nn.relu(out) for out in new_model_outputs]
     new_model = tf.keras.models.Model(inputs=[new_model_input], outputs=new_model_outputs, name=new_model_name)

     return new_model


"""
def make_intermediate_model(base_model, new_model_name, *args):
     style_outs, content_outs = [layer.output for layer in args[0]], [layer.output for layer in args[1]]
     
     new_model_input = base_model.input
     new_model_outputs = style_outs + content_outs
     new_model_outputs = [tfold.nn.relu(out) for out in new_model_outputs]
     new_model = tf.keras.models.Model(inputs=[new_model_input], outputs=new_model_outputs, name=new_model_name)

     return new_model
"""

my_model = make_intermediate_model(base_model, "My_Model", style_layers, content_layers)
#setting all layers in new_model to non-trainable #Note input layer of keras can accept a np.array or tensor or variable as input and hence it doesn't have to be explicitly set up to be trainable

my_model.summary()

import matplotlib.pyplot as plt
import tensorflow_probability as tfp

def deprocess(img):
    # perform the inverse of the pre processing step
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # convert RGB to BGR
    img = img[:, :, ::-1]
  
    img = np.clip(img, 0, 255).astype('uint8')
    return img

  
def display_image(image):
    # remove one dimension if image has 4 dimension
    img = image
    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
  
    img = deprocess(img)
  
    #plt.grid(False)
    #plt.xticks([])
    #plt.yticks([])
    #plt.imshow(img)
    img = Image.fromarray(img)
    display(img)
    return

def total_variation_loss(xtrain):
  return tf.image.total_variation(xtrain)

def content_loss(target, pred):
  return tf.reduce_sum(abs(target-pred))

def gram_matrix(feature_maps):
  F = feature_maps
  channels = tf.shape(F)[-1]
  F = tf.reshape(F, (-1, channels))
  n = tf.cast(tf.shape(F)[0], tf.float32)
  G = tf.matmul(F,F, transpose_a=True)
  return G

def style_loss(target_feature_maps, pred_feature_maps):
     A, G = gram_matrix(target_feature_maps), gram_matrix(pred_feature_maps) 
     size = tf.shape(target_feature_maps)[0]* tf.shape(target_feature_maps)[1]
     channels = tf.shape(target_feature_maps)[2]
    
     return tf.reduce_sum(abs(A -G))

def train(xtrain, base_content_data, base_style_data, random_initial = False, learning_rate=0.05, epochs=10, slw=1e1, clw=1e3, tvlw=1e-6):
      global base_content_data_across_func
      global base_style_data_across_func

      base_content_data_across_func = base_content_data
      base_style_data_across_func = base_style_data
      
      if random_initial == False: 
        xtrain = tf.Variable(initial_value=xtrain)
      else:
        initial_input = (np.random.uniform(0,255,(base_content_data.shape[0], base_content_data.shape[1], base_content_data.shape[2], base_content_data.shape[3])) - 128).astype(np.float32)
        xtrain = tf.Variable(initial_value=initial_input)
        #with tfold.variable_scope("input_initiate", reuse=tfold.AUTO_REUSE) as scope:
         #xtrain = tfold.get_variable(name="xtrain", shape=(base_content_data.shape[0], base_content_data.shape[1], base_content_data.shape[2], base_content_data.shape[3]), )

      lossT, lossS, lossC, grads = get_grads(xtrain, base_content_data, base_style_data, slw, clw, tvlw)
      opt = tfold.train.AdamOptimizer(learning_rate= learning_rate).apply_gradients(zip([grads], [xtrain]))
      
      #norm_means = np.array([103.939, 116.779, 123.68])
      #min_vals = 0 #-norm_means
      #max_vals = 255 #- norm_means   
  
    
      
      
      init = tfold.global_variables_initializer()

      with tfold.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
          #print(xtrain)
          #xtrain = tfp.optimizer.lbfgs_minimize(value_and_grads, initial_position=xtrain, tolerance=1e-8, max_iterations=20).position
          _, loss_total, loss_style, loss_content, updated_image = sess.run([opt, lossT, lossS, lossC, xtrain])
          
          
          if epoch%10 == 0:
             print("Epoch : ", epoch, "Loss_total : ", loss_total[0], "Loss_style : ", loss_style, "Loss_content : ", loss_content)
             print("Updated Image : \n")
             display_image(updated_image)
             print("\n")
    
      
    
      return updated_image

      
def value_and_grads(xtrain):
  results = get_grads(xtrain, base_content_data_across_func, base_style_data_across_func)
  trail = tf.reshape(results[-1], (3, 700, 700, None))
  print(results[0], results[-1], trail)
  return results[0], results[-1]

def get_grads(xtrain, base_content_data, base_style_data, slw, clw, tvlw):    

     with tf.GradientTape(persistent=True) as tape:
      tape.watch(xtrain)

      pred_outs = my_model(xtrain)
      target_outs_cont = my_model(base_content_data)[-1]
      target_outs_style = my_model(base_style_data)[:5]
      target_outs = target_outs_style + [target_outs_cont]
     
      total_style_loss = 0
      total_content_loss = 0
      per_style_layer_weight = 0.2
      style_loss_weight = slw #1e1 #4.5 #Style loss in powers of 10^8 so bring style loss and content loss to almost same order but make sure that style loss has slight edge which is not evident by slw or clw though
      content_loss_weight = clw #1e3 # 0.02 #Content loss in powers of 10^5
      total_variation_loss_weight = tvlw# (content_loss_weight/100.)

      for out in range(len(target_outs)):
        if out <5:
          total_style_loss += per_style_layer_weight*style_loss(target_outs[out], pred_outs[out])
        else:
          total_content_loss += content_loss(target_outs[out], pred_outs[out])
     
      total_loss = style_loss_weight*total_style_loss + content_loss_weight*total_content_loss +total_variation_loss_weight *total_variation_loss(xtrain)
      #total_style_loss, total_content_loss,
     return total_loss, total_style_loss, total_content_loss, tape.gradient(total_loss, xtrain)

    


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

im_content = Image.open(r'/content/content_image_8.jpg')
im_content = im_content.resize((512, 512))
display(im_content)

im_style = Image.open(r'/content/style_image_10.jpg')
im_style = im_style.resize((512, 512))
display(im_style)

from numpy import asarray
xtrain = preprocess_input(asarray(im_content))
base_content_data = preprocess_input(asarray(im_content))
base_style_data = preprocess_input(asarray(im_style))

import numpy as np

xtrain = np.expand_dims(xtrain, axis=0)
base_content_data = np.expand_dims(base_content_data, axis=0)
base_style_data = np.expand_dims(base_style_data, axis=0)


final_image = train(xtrain, base_content_data, base_style_data, learning_rate=6., epochs=100)
#updated_image = Image.fromarray(updated_image[0])
#display(updated_image)

