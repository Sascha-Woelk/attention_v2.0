# import libraries
from modules.setup_file import *
from modules.custom_dense import custom_dense
from imagenet1k_class2labels import class_dictionary

# # load Tensorboard extension
# %load_ext tensorboard
# # clear any logs from previous runs
# !rm -rf ./logs/

# import hyperparameters
with open('config_6.yaml', 'r') as file:
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

# find the numeric labels of all classes containing the target class keyword
target_classes = []
target_string = re.compile(r'\bcat\b', re.IGNORECASE)
for index, (key, value) in enumerate(class_dictionary.items()):
  string_match = target_string.search(value)
  if string_match:
    target_classes.append(key)
print("Focusing attention on the following classes:")    
for match in target_classes:
  print(class_dictionary[match])

# create image import pipeline
from modules.import_pipeline import import_pipeline
train_generator, validation_generator, test_generator = import_pipeline(train_dir,
                                                                        test_dir,
                                                                        IMG_SIZE,
                                                                        BATCH_SIZE,
                                                                        target_class = target_classes,
                                                                        sampling_rate = hyper_params['sampling_rate'],
                                                                        target_class_weight = 0.1)

# load VGG16 pre-trained on ImageNet
base_model = tf.keras.applications.VGG16(include_top=True,
                                         weights='imagenet',
                                         classifier_activation='softmax')

# Load the retrained weights
base_model.load_weights('log_config_6/cp.ckpt')

# compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['base_learning_rate'])
loss = tf.keras.losses.CategoricalCrossentropy()
base_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

# evaluate the retrained model
loss1, accuracy1 = base_model.evaluate(test_generator)
print("retrained base model loss: {:.2f}".format(loss1))
print("retrained base model accuracy: {:.2f}".format(accuracy1))

# construct the new model with the additional layer
base_model.trainable = False

IMG_SHAPE = IMG_SIZE + (3,)
inputs = tf.keras.Input(shape=IMG_SHAPE)
new_layer_at = 15
initializer = tf.keras.initializers.Ones()
constraint = tf.keras.constraints.NonNeg()

for layer in range(1,len(base_model.layers) + 1):
  if layer == 1:
    x = base_model.layers[layer](inputs)
  elif layer < new_layer_at:
    x = base_model.layers[layer](x)
  elif layer == new_layer_at:
    x = custom_dense(512,
                      use_bias=False,
                      kernel_initializer=initializer,
                      kernel_constraint=constraint,
                      name='attention_layer')(x)
  else:
    x = base_model.layers[layer-1](x)
outputs = x

attention_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='attention_model')
attention_model.summary()
print("Number of layers in the attention model: ", len(attention_model.layers))
print("Number of trainable variables is: ", len(attention_model.trainable_variables))  
print("Attention layer set to trainable: ", attention_model.get_layer('attention_layer').trainable)

# compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['base_learning_rate'])
loss = tf.keras.losses.CategoricalCrossentropy()
attention_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

# evaluate the untrained attention model
loss2, accuracy2 = attention_model.evaluate(test_generator)
print("untrained attention model loss: {:.2f}".format(loss2))
print("untrained attention model accuracy: {:.2f}".format(accuracy2))

# define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hyper_params['patience'], mode='auto'),
    # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                   monitor='val_loss',
    #                                   verbose=1,
    #                                   save_best_only=True,
    #                                   save_weights_only=True,
    #                                   mode='auto',
    #                                   save_freq='epoch',
    #                                   options=None),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.5,
                                        patience=3,
                                        verbose=0,
                                        mode='auto'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  ]

# fit the base model on the ImageNet dataset
history = attention_model.fit(train_generator,
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
loss3, accuracy3 = attention_model.evaluate(test_generator)
print("trained attention model loss: {:.2f}".format(loss3))
print("trained attention model accuracy: {:.2f}".format(accuracy3))