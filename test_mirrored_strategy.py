from modules.setup_file import *

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:3", "/gpu:4"])

BATCH_SIZE = 64

GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync

print(f'Number of devices: {strategy.num_replicas_in_sync}')

ds = tf.data.Dataset.from_generator(test_generator,('float64', 'float64'))

with strategy.scope():
