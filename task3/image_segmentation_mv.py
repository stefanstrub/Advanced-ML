import pickle
import gzip
import tensorflow as tf
from pix2pix import pix2pix
import tensorflow_datasets as tfds
import numpy as np
import cv2

from keras.callbacks import TensorBoard
tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# from tensorflow_examples.models.pix2pix import pix2pix

import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

# load data
folder_path = 'task3/'
train_data = load_zipped_pickle(folder_path+"train.pkl")
test_data = load_zipped_pickle(folder_path+"test.pkl")
samples = load_zipped_pickle(folder_path+"sample.pkl")

train_data = train_data[46:]

train_roi = []
train_roi_label = []
for i in range(len(train_data)):
  train_roi.append([])
  train_roi_label.append([])
  start_row = -1
  end_row = -1
  length_row = 0
  for j in range(len(train_data[i]['box'])):
      if np.any(train_data[i]['box'][j]):
          length_row += 1
          if start_row == -1:
              start_row = j
          end_row = j
  start_col = -1
  end_col = -1
  length_col = 0
  for j in range(len(train_data[i]['box'][0])):
      if np.any(train_data[i]['box'][start_row:end_row,j]):
          length_col += 1
          if start_col == -1:
              start_col = j
          end_col = j
  for j in range(train_data[i]['video'].shape[2]):
      train_roi[-1].append(train_data[i]['video'][:,:,j][train_data[i]['box']].reshape(length_row,length_col))
      train_roi_label[-1].append(train_data[i]['label'][:,:,j][train_data[i]['box']].reshape(length_row,length_col))
  train_roi[-1] = np.array(train_roi[-1])
  train_roi_label[-1] = np.array(train_roi_label[-1])

train_roi = np.array(train_roi)

# train_roi = np.reshape((train_roi.shape[0], train_roi[0].shape[0], train_roi[0].shape[1], train_roi[0].shape[2]))
for i in range(len(train_data)):
  train_data[i]['roi'] = np.zeros((len(train_roi[i][0]),len(train_roi[i][0][0]),train_data[i]['video'].shape[2]))
  train_data[i]['roi_label'] = np.zeros((len(train_roi[i][0]),len(train_roi[i][0][0]),train_data[i]['video'].shape[2]))
  for j in range(train_data[i]['video'].shape[2]):
    train_data[i]['roi'][:,:,j] = train_roi[i][j]
    train_data[i]['roi_label'][:,:,j] = train_roi_label[i][j]

split = 0.95
validation_data = train_data[int(len(train_data)*split):]
train_data = train_data[:int(len(train_data)*split)]

# plt.figure()
# plt.imshow(train_data[i]['roi'][:,:,j])
# plt.figure()
# plt.imshow( train_roi[i][j])
# plt.show()
# plt.figure()
# plt.imshow( validation_data[i]['roi'][:,:,j])
# plt.show()

# plt.figure()
# plt.imshow(train_data2[i]['roi'][:,:,j])
# plt.figure()
# plt.imshow( train_roi[i][j])
# plt.show()
resolution = 128

image_of_interest = 'video'
label_of_interest = 'label'
def load_dataset(dataset):
  input_image = []
  input_label = []
  for i in range(len(dataset)):
    for j in dataset[i]['frames']:
      img = dataset[i][image_of_interest][:,:,j].reshape((dataset[i][image_of_interest][:,:,j].shape[0],dataset[i][image_of_interest][:,:,j].shape[1],1))
      # input_image.append(tf.image.resize(img, (resolution, resolution)))
      input_image.append(tf.image.grayscale_to_rgb(tf.image.resize(img, (resolution, resolution))))

      label = np.asarray(dataset[i][label_of_interest][:,:,j], dtype = 'uint8').reshape((dataset[i][label_of_interest][:,:,j].shape[0],dataset[i][label_of_interest][:,:,j].shape[1],1))
      input_label.append(tf.image.resize(label, (resolution, resolution)))
      input_label[-1] = tf.round(input_label[-1])
  input_dataset = tf.data.Dataset.from_tensor_slices({'image':input_image, 'label':input_label})
  return input_dataset

def load_testset(dataset):
  input_image = []
  for i in range(len(dataset)):
    for j in range(len(dataset[i][image_of_interest][0,0,:])):
      img = dataset[i][image_of_interest][:,:,j].reshape((dataset[i][image_of_interest][:,:,j].shape[0],dataset[i][image_of_interest][:,:,j].shape[1],1))
      # input_image.append(tf.repeat(tf.image.resize(img, (resolution, resolution)),3,-1))
      # input_image.append(tf.image.resize(img, (resolution, resolution)))
      input_image.append(tf.image.grayscale_to_rgb(tf.image.resize(img, (resolution, resolution))))
  input_dataset = tf.data.Dataset.from_tensor_slices({'image':input_image})
  return input_dataset

def normalize(train_dataset):
  input_image = tf.cast(train_dataset['image'], tf.float32) / 255.0
  return input_image, train_dataset['label']

def normalize_image(train_dataset):
  input_image = tf.cast(train_dataset['image'], tf.float32) / 255.0
  return input_image

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (resolution, resolution))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (resolution, resolution))
  # input_image = tf.image.resize(datapoint['image'], (resolution, resolution))
  # input_mask = tf.image.resize(datapoint['label'], (resolution, resolution))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

train_dataset = load_dataset(train_data)
train_images = train_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = load_dataset(validation_data)
test_images = test_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

image_of_interest = 'video'
label_of_interest = 'label'
task_dataset = load_testset(test_data)
task_images = task_dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
# task_images = task_images.map(lambda x, y: x)

# np.random.seed(42)
TRAIN_LENGTH = len(list(train_images))*5
TRAIN_LENGTH = 2**11
# TRAIN_LENGTH = info.splits['train'].num_examples
TEST_LENGTH = len(list(task_images))
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    # self.augment_inputs = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", seed=seed),tf.keras.layers.RandomRotation(0.3, seed=seed), tf.keras.layers.RandomTranslation(0.2,0.2, seed=seed)])
    # self.augment_labels = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", seed=seed),tf.keras.layers.RandomRotation(0.3, seed=seed), tf.keras.layers.RandomTranslation(0.2,0.2, seed=seed)])
    self.augment_inputs = tf.keras.Sequential([tf.keras.layers.RandomTranslation(0.2,0.2, seed=seed),tf.keras.layers.RandomRotation(1, seed=seed), tf.keras.layers.RandomZoom(0.2, seed=seed)])
    self.augment_labels = tf.keras.Sequential([tf.keras.layers.RandomTranslation(0.2,0.2, seed=seed),tf.keras.layers.RandomRotation(1, seed=seed), tf.keras.layers.RandomZoom(0.2, seed=seed)])
    # self.augment_inputs = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", seed=seed),tf.keras.layers.RandomRotation(1, seed=seed)])
    # self.augment_labels = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", seed=seed),tf.keras.layers.RandomRotation(1, seed=seed)])

    # self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    # self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    labels = tf.round(labels)
    return inputs, labels

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE).repeat()
task_batches = task_images.batch(TEST_LENGTH)
# len_batches = len(list(train_batches.as_numpy_iterator()))

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])

base_model = tf.keras.applications.MobileNetV2(input_shape=[resolution, resolution, 3], include_top=False)

# adding regularization
regularizer = tf.keras.regularizers.l2(0.01)

for layer in base_model.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
          setattr(layer, attr, regularizer)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[resolution, resolution, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



smooth = 0.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f**2) + tf.keras.backend.sum(y_pred_f**2) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# for image, mask in test_batches.take(1):
#     print(np.max(image[0]))
#     dice = dice_coef_loss(image[0], mask[0])
#     print(dice.numpy())

# ### U net mode
# inputs = tf.keras.layers.Input((128, 128, 1))
# conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
# pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

# conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
# conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
# pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

# conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
# conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
# pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

# conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
# conv4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
# pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

# conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
# conv5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

# up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
# conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
# conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

# up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
# conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
# conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

# up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
# conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
# conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

# up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
# conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
# conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

# conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='dense')(conv9)

# model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])

# model_checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

# model.compile(optimizer='adam',
#               loss=dice_coef_loss, metrics=[dice_coef])



# def conv_block(tensor, nfilters, size=1, padding='same', initializer="he_normal"):
#     x = tf.keras.layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation("relu")(x)
#     x = tf.keras.layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation("relu")(x)
#     return x


# def deconv_block(tensor, residual, nfilters, size=1, padding='same', strides=(2, 2)):
#     y = tf.keras.layers.Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
#     y = tf.keras.layers.concatenate([y, residual], axis=3)
#     y = conv_block(y, nfilters)
#     return y


# def Unet(img_height, img_width, nclasses=3, filters=64):
# # down
#     input_layer = tf.keras.layers.Input(shape=(img_height, img_width, 1), name='image_input')
#     conv1 = conv_block(input_layer, nfilters=filters)
#     conv1_out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = conv_block(conv1_out, nfilters=filters*2)
#     conv2_out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = conv_block(conv2_out, nfilters=filters*4)
#     conv3_out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = conv_block(conv3_out, nfilters=filters*8)
#     conv4_out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
#     conv4_out = tf.keras.layers.Dropout(0.5)(conv4_out)
#     conv5 = conv_block(conv4_out, nfilters=filters*16)
#     conv5 = tf.keras.layers.Dropout(0.5)(conv5)
# # up
#     deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
#     deconv6 = tf.keras.layers.Dropout(0.5)(deconv6)
#     deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
#     deconv7 = tf.keras.layers.Dropout(0.5)(deconv7) 
#     deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
#     deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# # output
#     output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
#     output_layer = tf.keras.layers.BatchNormalization()(output_layer)
#     output_layer = tf.keras.layers.Activation('sigmoid')(output_layer)

#     model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='Unet')
#     return model

# model = Unet(128, 128, nclasses=2, filters=16)
# print(model.output_shape)
# # unet.load_weights('models-dr/top_weights.h5')
# model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy'])
# tb = TensorBoard(log_dir='logs', write_graph=True)
# mc = tf.keras.callbacks.ModelCheckpoint(mode='max', filepath='models-dr/top_weights.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
# es = tf.keras.callbacks.EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
# callbacks = [tb, mc, es]

# def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
#     conv = tf.keras.layers.Conv2D(n_filters, 
#                   3,  # filter size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(inputs)
#     conv = tf.keras.layers.Conv2D(n_filters, 
#                   3,  # filter size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(conv)
  
#     conv = tf.keras.layers.BatchNormalization()(conv, training=False)    
#     if dropout_prob > 0:     
#         conv = tf.keras.layers.Dropout(dropout_prob)(conv)
#     if max_pooling:
#         next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
#     else:
#         next_layer = conv
#     skip_connection = conv    
#     return next_layer, skip_connection

# def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
#     up = tf.keras.layers.Conv2DTranspose(
#                  n_filters,
#                  (3,3),
#                  strides=(2,2),
#                  padding='same')(prev_layer_input)    
#     merge = tf.keras.layers.concatenate([up, skip_layer_input], axis=3)  
#     conv = tf.keras.layers.Conv2D(n_filters, 
#                  3,  
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='HeNormal')(merge)
#     conv = tf.keras.layers.Conv2D(n_filters,
#                  3, 
#                  activation='relu',
#                  padding='same',
#                  kernel_initializer='HeNormal')(conv)
#     return conv 

# predictions = model(train_images).numpy()
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# tf.keras.utils.plot_model(model, show_shapes=True)

def create_masks(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  # pred_mask = pred_mask>0.5
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()
EPOCHS = 30
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches)

# model_history = model.fit(train_batches, epochs=EPOCHS,
#                           steps_per_epoch=STEPS_PER_EPOCH,
#                           validation_steps=VALIDATION_STEPS,
#                           validation_data=test_batches,
#                           )

show_predictions()

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_batches, 1)

list_task = []
for element in task_dataset.as_numpy_iterator(): 
  list_task.append(element) 

for image, mask in train_batches.take(1):
  prediction = model.predict(image)

for image in task_batches.take(TEST_LENGTH+1):
  pred_mask = model.predict(image)
  prediction_tf = create_masks(pred_mask)
  prediction_np = prediction_tf.numpy()


prediction_resized = []
num = 0
for i in range(len(test_data)):
  for j in range(len(test_data[i]['video'][0,0,:])):
    prediction_reshaped = np.reshape(prediction_np[num], (prediction_np[i+j].shape[0],prediction_np[i+j].shape[1]))
    prediction_reshaped = np.asarray(prediction_reshaped, dtype='uint8')
    prediction_resized.append(cv2.resize(prediction_reshaped, (test_data[i]['video'].shape[1],test_data[i]['video'].shape[0])))
    num += 1

for i in range(len(prediction_resized)):
    prediction_resized[i] = np.asarray(prediction_resized[i], dtype='bool')

i = 1
plt.figure()
plt.imshow(test_data[i]['video'][:,:,0])
plt.imshow(prediction_resized[103], alpha=0.5)
plt.show()

predictions = []
num = 0
for i, d in enumerate(test_data):
  prediction_video = []
  prediction = np.array(np.zeros_like(d['video']), dtype=np.bool)
  for j in range(len(test_data[i]['video'][0,0,:])):
    prediction[:,:,j] = prediction_resized[num]
    num += 1
  # DATA Strucure
  predictions.append({
      'name': d['name'],
      'prediction': prediction
      }
  )

i = 1
j = 6
plt.figure()
plt.imshow(test_data[i]['video'][:,:,j])
plt.imshow(predictions[i]['prediction'][:,:,j], alpha=0.5)

i = 2
j = 6
plt.figure()
plt.imshow(test_data[i]['video'][:,:,j])
plt.imshow(predictions[i]['prediction'][:,:,j], alpha=0.5)

i = 3
j = 30
plt.figure()
plt.imshow(test_data[i]['video'][:,:,j])
plt.imshow(predictions[i]['prediction'][:,:,j], alpha=0.5)

# save in correct format
save_zipped_pickle(predictions, folder_path+'my_predictions.pkl')

show_predictions(test_batches, 2)

print('end')