# import libraries
from modules.setup_file import *
from files.imagenet1k_class2labels import class_dictionary
from modules.import_pipeline import import_pipeline
from modules.custom_initializer import custom_onezero_initializer
from modules.custom_dense import custom_dense

# import hyperparameters
with open('configs/config_9.yaml', 'r') as file:
    hyper_params = yaml.safe_load(file)
try:
  sampling_rate = hyper_params['sampling_rate']
except:
  sampling_rate = None
print(hyper_params['config_name'])

# import 20 target classes based on recall quantile split
with open('files/class_recall_quantiles.pickle', 'rb') as file:
  samples_recall_q1, samples_recall_q2, samples_recall_q3, samples_recall_q4 = pickle.load(file)
target_classes = np.hstack((samples_recall_q1, samples_recall_q2, samples_recall_q3, samples_recall_q4))

# define attention intensity levels to test
attention_levels = [0.9, 0.7, 0.5, 0.3, 0.1]

# loop over all target classes x attention intensity levels and to train the attention layer and evaluate
for target_class in target_classes:
  for intensity in attention_levels:

    # identify current working directory and set up subdirectories
    working_directory = os.getcwd()
    charts_dir = os.path.join(working_directory, 'charts/')
    log_dir = "logs/fit/class{}_intensity{}/".format(target_class, int(intensity*1000)) + dt.datetime.now().strftime("%Y%m%d-%H%M%S") 
    checkpoint_path = "checkpoints/log_class{}_intensity{}/cp.ckpt".format(target_class, int(intensity*1000))
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # set dataset directory paths
    PATH = os.path.dirname(hyper_params['directory'])
    train_dir = os.path.join(PATH, hyper_params['training_folder'])
    test_dir = os.path.join(PATH, hyper_params['test_folder'])

    # set model hyperparameters
    IMG_SIZE = tuple(hyper_params['img_size'])
    BATCH_SIZE = hyper_params['batch_size']

    # create image import pipeline
    train_generator, validation_generator, test_generator = import_pipeline(train_dir,
                                                                            test_dir,
                                                                            IMG_SIZE,
                                                                            BATCH_SIZE,
                                                                            target_class = target_class,
                                                                            sampling_rate = hyper_params['sampling_rate'],
                                                                            target_class_weight = intensity,
                                                                            )

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
    base_loss, base_accuracy = base_model.evaluate(test_generator)
    print("retrained base model loss: {:.2f}".format(base_loss))
    print("retrained base model accuracy: {:.2f}".format(base_accuracy))

    ## construct the new model with the additional layer
    # build a test model only up to and including the attention layer
    base_model.trainable = False
    IMG_SHAPE = IMG_SIZE + (3,)
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    new_layer_at = 15
    constraint = tf.keras.constraints.NonNeg()
    initializer = custom_onezero_initializer()

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
        pass
    outputs = x

    test_attention_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='attention_model')

    # compile the test attention model
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['base_learning_rate'])
    loss = tf.keras.losses.CategoricalCrossentropy()
    test_attention_model.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=['accuracy'])

    ## confirm the attention layer is working as expected
    # extract attention weights and confirm every second weight is set to zero
    attention_weight_extracted = test_attention_model.get_layer('attention_layer').get_weights()[0][0,:]
    print("Every second weight set to zero: {}".format(((attention_weight_extracted[1::2] == 0)).any()))
    # run one batch and select only the filter dimension from the prediction
    prediction_sample = test_attention_model.predict(next(iter(train_generator))[0])[3][0,0,:]
    print("All corresponding final layer outputs equal to zero: {}".format((prediction_sample[1::2] == 0).all()))

    # build the actual attention model
    initializer = tf.keras.initializers.Ones()

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
                  metrics=['accuracy',
                          tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy', dtype=None)])

    # evaluate the untrained attention model
    untrained_attention_loss, untrained_attention_accuracy, untrained_attention_top5accuracy = attention_model.evaluate(test_generator)
    print("untrained attention model loss: {:.2f}".format(untrained_attention_loss))
    print("untrained attention model accuracy: {:.2f}".format(untrained_attention_accuracy))
    print("untrained attention model top5accuracy: {:.2f}".format(untrained_attention_top5accuracy))

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
                                            factor=hyper_params['reduceLROP_factor'],
                                            patience=hyper_params['patience'],
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
    plt.savefig(charts_dir + '{}_accuracy_class{}_intensity{}.png'.format(dt.datetime.today().date(), target_class, int(intensity*1000)))
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
    plt.savefig(charts_dir + '{}_loss__class{}_intensity{}.png'.format(dt.datetime.today().date(), target_class, int(intensity*1000)))
    plt.show()

    # evaluate the retrained model
    trained_attention_loss, trained_attention_accuracy, trained_attention_top5accuracy = attention_model.evaluate(test_generator)
    print("trained attention model loss: {:.2f}".format(trained_attention_loss))
    print("trained attention model accuracy: {:.2f}".format(trained_attention_accuracy))
    print("trained attention model top5accuracy: {:.2f}".format(trained_attention_top5accuracy))
    
    # create confusion matrix
    test_predictions = attention_model.predict(test_generator)
    Y_pred = tf.argmax(test_predictions, axis=-1)
    Y_true = []
    for i in range(len(test_generator.index_array)):
        loc_index = test_generator.index_array[i]
        Y_true.append(test_generator.labels[loc_index])  
    confusion_matrix = tf.math.confusion_matrix(Y_true, Y_pred)
    
    # save the confusion matrix for each run
    with open('files/confusion_matrices/class{}_intensity{}.pickle'.format(target_class, int(intensity*1000)), 'wb') as file:
      pickle.dump(confusion_matrix, file)