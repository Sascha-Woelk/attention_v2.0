import os
import numpy as np
import multiprocessing.pool
from keras_preprocessing import image
# from keras_preprocessing.image.iterator import BatchFromFilesMixin, Iterator
from keras_preprocessing.image.utils import _list_valid_filenames_in_directory
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.custom_iterator import BatchFromFilesMixin, CustomIterator

class CustomDirectoryIterator(BatchFromFilesMixin, CustomIterator):
    
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __new__(cls, *args, **kwargs):
        try:
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
        except ImportError:
            pass
        return super(CustomDirectoryIterator, cls).__new__(cls)
    
    def __init__(self, directory, image_data_generator,
                target_size=(256, 256),
                color_mode='rgb',
                classes=None,
                class_mode='categorical',
                batch_size=32,
                shuffle=True,
                seed=None,
                data_format=None,
                save_to_dir=None,
                save_prefix='',
                save_format='png',
                follow_links=False,
                subset=None,
                interpolation='nearest',
                dtype=None,
                target_class=None,
                sampling_rate=None,
                target_class_weight=None):
        super(CustomDirectoryIterator, self).set_processing_attrs(image_data_generator,
                                                            target_size,
                                                            color_mode,
                                                            data_format,
                                                            save_to_dir,
                                                            save_prefix,
                                                            save_format,
                                                            subset,
                                                            interpolation)
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes)))) #SW maps folder names to number indices

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        
        self.target_class = target_class #SW
        self.sampling_rate = sampling_rate #SW
        self.target_class_weight = target_class_weight #SW
        
        #SW the next line calls the Iterator, passing it samples as n
        super(CustomDirectoryIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed,
                                                self.classes,
                                                self.target_class,
                                                self.sampling_rate,
                                                self.target_class_weight)
    
    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

              
def flow_from_directory_independent(InstanceofImageDataGenerator,
                        directory,
                        target_size=(256, 256),
                        color_mode='rgb',
                        classes=None,
                        class_mode='categorical',
                        batch_size=32,
                        shuffle=True,
                        seed=None,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='png',
                        follow_links=False,
                        subset=None,
                        interpolation='nearest',
                        target_class=None,
                        sampling_rate=None,
                        target_class_weight=None):

    return CustomDirectoryIterator(
        directory,
        InstanceofImageDataGenerator,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        data_format=InstanceofImageDataGenerator.data_format,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        follow_links=follow_links,
        subset=subset,
        interpolation=interpolation,
        target_class=target_class,
        sampling_rate=sampling_rate,
        target_class_weight=target_class_weight)