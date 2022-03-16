from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modules.custom_flow_from_directory import flow_from_directory_independent
from tensorflow.keras.applications.vgg16 import preprocess_input

def import_pipeline(train_dir, test_dir, IMG_SIZE, BATCH_SIZE, shuffle=True, validation_split=0.2, target_class=None, sampling_rate=None, target_class_weight=None):
    
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                       validation_split=validation_split)
    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    train_generator = flow_from_directory_independent(train_datagen,
                                                  train_dir,
                                                  subset='training',
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=shuffle,
                                                  target_class=target_class,
                                                  sampling_rate=sampling_rate,
                                                  target_class_weight=target_class_weight)

    validation_generator = flow_from_directory_independent(train_datagen,
                                                  train_dir,
                                                  subset='validation',
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=shuffle,
                                                  target_class=target_class,
                                                  sampling_rate=sampling_rate,
                                                  target_class_weight=target_class_weight)

    test_generator = flow_from_directory_independent(test_datagen,
                                                   test_dir,
                                                   target_size=IMG_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical',
                                                   shuffle=shuffle,
                                                   target_class=target_class,
                                                   sampling_rate=sampling_rate,
                                                   target_class_weight=target_class_weight)
    
    return train_generator, validation_generator, test_generator