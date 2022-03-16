# import libraries
from modules.setup_file import *

# import hyperparameters
with open('configs/config_6.yaml', 'r') as file:
    hyper_params = yaml.safe_load(file)
try:
  sampling_rate = hyper_params['sampling_rate']
except:
  sampling_rate = None
print(hyper_params['config_name'])

# identify current working directory and set up subdirectories
working_directory = os.getcwd()
charts_dir = os.path.join(working_directory, 'charts/')
log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S") 
checkpoint_path = "log_{}/cp.ckpt".format(hyper_params['config_name'])
checkpoint_dir = os.path.dirname(checkpoint_path)

# set dataset directory paths
PATH = os.path.dirname(hyper_params['directory'])
train_dir = os.path.join(PATH, hyper_params['training_folder'])
test_dir = os.path.join(PATH, hyper_params['test_folder'])

# set model hyperparameters
IMG_SIZE = tuple(hyper_params['img_size'])
BATCH_SIZE = hyper_params['batch_size']

# open a sample image and check it's dimensions
sample_img_directory = next(os.scandir(train_dir)).path
sample_image_file = next(os.scandir(sample_img_directory)).path
sample_image = Image.open(sample_image_file)
width, height = sample_image.size
print(f'Images found with width {width} and height {height}.')

# create image import pipeline
from modules.import_pipeline import import_pipeline
train_generator, validation_generator, test_generator = import_pipeline(train_dir,
                                                                        test_dir,
                                                                        IMG_SIZE,
                                                                        BATCH_SIZE,
                                                                        target_class = None,      
                                                                        sampling_rate = 0.1,
                                                                        target_class_weight = None)

# display a sample image and print it's label and dimensions
plt.imshow(next(iter(train_generator))[0][1])
print(f'Class label {next(iter(train_generator.class_indices))}')
print(f'Pre-processed image shape is {next(iter(train_generator))[0][1].shape}.')

# load VGG16 pre-trained on ImageNet
base_model = tf.keras.applications.VGG16(include_top=True,
                                         weights='imagenet',
                                         classifier_activation='softmax')

# print the model summary incl. number of trainable parameters in the model
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))
print("Number of trainable variables is: ", len(base_model.trainable_variables))

# define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hyper_params['patience'], mode='auto'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='auto',
                                      save_freq='epoch',
                                      options=None),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.5,
                                        patience=3,
                                        verbose=0,
                                        mode='auto'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  ]

# compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['base_learning_rate'])
loss = tf.keras.losses.CategoricalCrossentropy()
base_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

# evaluate the raw model
loss0, accuracy0 = base_model.evaluate(test_generator)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# fit the base model on the ImageNet dataset
history = base_model.fit(train_generator,
                          epochs=hyper_params['epochs'],
                          batch_size=hyper_params['batch_size'],
                          validation_data=validation_generator,
                          verbose=1,
                          callbacks=callbacks)

# graph the history for accuracy
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig(charts_dir + '{}_accuracy_{}.png'.format(dt.datetime.today().date(), hyper_params['config_name']))
plt.show()

# graph the history for loss
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.savefig(charts_dir + '{}_loss_{}.png'.format(dt.datetime.today().date(), hyper_params['config_name']))
plt.show()

# evaluate the retrained model
loss0, accuracy0 = base_model.evaluate(test_generator)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Retrieve a batch of images from the test set
image_batch, label_batch = next(iter(train_generator))
predictions = base_model.predict_on_batch(image_batch)

# convert the predictions to class labels
from tensorflow.keras.applications.vgg16 import decode_predictions
prediction_results = decode_predictions(predictions)

# display 9 sample images and corresponding predictions
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  im_rgb = image_batch[i].astype("uint8")[:, :, [2, 1, 0]]
  plt.imshow(im_rgb)
  title_obj = plt.title(prediction_results[i][0][1] + " or " + prediction_results[i][1][1])
  plt.getp(title_obj)
  plt.getp(title_obj, 'text')
  plt.setp(title_obj, color='w')
  plt.axis("off")