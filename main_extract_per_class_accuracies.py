# import libraries
from modules.setup_file import *
from modules.import_pipeline import import_pipeline

# import hyperparameters
with open('configs/config_6.yaml', 'r') as file:
    hyper_params = yaml.safe_load(file)

# identify current working directory and set up subdirectories
working_directory = os.getcwd()

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
                                                                        target_class=None,
                                                                        sampling_rate=None,
                                                                        target_class_weight=None)

# load VGG16 pre-trained on ImageNet
base_model = tf.keras.applications.VGG16(include_top=True,
                                        weights='imagenet',
                                        classifier_activation='softmax')

# load the retrained weights
base_model.load_weights('log_config_6/cp.ckpt')

# compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['base_learning_rate'])
loss = tf.keras.losses.CategoricalCrossentropy()
base_model.compile(optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])

# create confusion matrix
test_predictions = base_model.predict(test_generator)
Y_pred = tf.argmax(test_predictions, axis=-1)
Y_true = []
for i in range(len(test_generator.index_array)):
    loc_index = test_generator.index_array[i]
    Y_true.append(test_generator.labels[loc_index])  
confusion_matrix = tf.math.confusion_matrix(Y_true, Y_pred)

# calculate signal detection metrics
stimuli_per_class = confusion_matrix.numpy().sum(axis=1) 
predictions_per_class = confusion_matrix.numpy().sum(axis=0)

true_positives = np.diag(confusion_matrix)
false_positives = predictions_per_class - true_positives
false_negatives = stimuli_per_class - true_positives
true_negatives = confusion_matrix.numpy().sum() - (true_positives + false_positives + false_negatives)

hit_rate = true_positives/(true_positives + false_negatives)
false_alarm_rate = false_positives/(false_positives + true_negatives)

per_class_accuracy = (true_positives + true_negatives) / (false_positives + false_negatives + true_positives + true_negatives)
per_class_precision = true_positives / (true_positives + false_positives)
per_class_recall = true_positives / (true_positives + false_negatives)

# split classes into 4 quantiles, based on recall
low, mid, high = np.quantile(per_class_recall, [0.25, 0.5, 0.75])
quantiles = np.where(per_class_recall<=low, 'q1',
                     np.where((per_class_recall>low) & (per_class_recall<=mid), 'q2',
                              np.where((per_class_recall>mid) & (per_class_recall<=high), 'q3',
                                       np.where(per_class_recall>high, 'q4', ''))))

# save class indices for each quantiles to variables
recall_q1 = np.where(quantiles == 'q1')[0]
recall_q2 = np.where(quantiles == 'q2')[0]
recall_q3 = np.where(quantiles == 'q3')[0]
recall_q4 = np.where(quantiles == 'q4')[0]

# select 5 classes from each recall quantile
np.random.seed(1)
samples_recall_q1 = np.random.choice(recall_q1, size=5)
samples_recall_q2 = np.random.choice(recall_q2, size=5)
samples_recall_q3 = np.random.choice(recall_q3, size=5)
samples_recall_q4 = np.random.choice(recall_q3, size=5)

with open('files/class_recall_quantiles.pickle', 'wb') as file:
    pickle.dump([samples_recall_q1, samples_recall_q2, samples_recall_q3, samples_recall_q4], file)