# import libraries
from modules.setup_file import *
from modules.custom_dense import custom_dense
from sklearn.decomposition import PCA
from scipy.linalg import svd

# import hyperparameters
with open('configs/config_9.yaml', 'r') as file:
    hyper_params = yaml.safe_load(file)
try:
  sampling_rate = hyper_params['sampling_rate']
except:
  sampling_rate = None
print(hyper_params['config_name'])

# set model hyperparameters
IMG_SIZE = tuple(hyper_params['img_size'])
BATCH_SIZE = hyper_params['batch_size']


# load VGG16 pre-trained on ImageNet
base_model = tf.keras.applications.VGG16(include_top=True,
                                        weights='imagenet',
                                        classifier_activation='softmax')

# Load the retrained weights
base_model.load_weights('log_config_6/cp.ckpt')

# Select the kernel of weights of the convolutional layer of convolutional block 4
kernel = base_model.get_layer('block4_conv3').kernel
flattened_kernel = tf.reshape(kernel,[-1, 512]).numpy()

# Singular-value decomposition
U, s, VT = svd(flattened_kernel)

# select top-10 principal components to create step-up matrix
step_up_matrix = tf.convert_to_tensor(VT[:10,:])
step_up_matrix = tf.dtypes.cast(step_up_matrix, tf.double)

# define attention seeds
attention_seeds=np.ones(10)
attention_seeds = tf.convert_to_tensor(attention_seeds)
attention_seeds = tf.reshape(attention_seeds,[1,10])
attention_seeds = tf.dtypes.cast(attention_seeds, tf.double)
attention_weights = tf.tensordot(attention_seeds,step_up_matrix,axes=[1,0])

# build attention model with latent space attention seeds
base_model.trainable = False
IMG_SHAPE = IMG_SIZE + (3,)
inputs = tf.keras.Input(shape=IMG_SHAPE)
secondary_input = tf.keras.Input(shape=None, tensor= tf.reshape(tf.ones(10),[10,1]))
new_layer_at = 15
constraint = tf.keras.constraints.NonNeg()
initializer = tf.keras.initializers.Ones()

for layer in range(1,len(base_model.layers) + 1):
    if layer == 1:
        x = base_model.layers[layer](inputs)
    elif layer < new_layer_at:
        x = base_model.layers[layer](x)
    elif layer == new_layer_at:
        attention_seed_layer_output = custom_dense(10,
                                                    use_bias=False,
                                                    kernel_initializer=initializer,
                                                    kernel_constraint=constraint,
                                                    name='attention_seed_layer')(secondary_input)
        x = tf.keras.layers.Dot(axes=(1,10))([attention_seed_layer_output,step_up_matrix])    
        # x = custom_dense(512,
        #                 use_bias=False,
        #                 kernel_initializer=initializer,
        #                 kernel_constraint=constraint,
        #                 name='attention_layer')(x)
    else:
        x = base_model.layers[layer-1](x)
outputs = x

attention_seed_model = tf.keras.Model(inputs=[inputs, secondary_input], outputs=outputs, name='attention_seed_model')
attention_seed_model.summary()
print("Number of layers in the attention seed model: ", len(attention_seed_model.layers))
print("Number of trainable variables is: ", len(attention_seed_model.trainable_variables))  
print("Attention seed layer set to trainable: ", attention_seed_model.get_layer('attention_seed_layer').trainable)