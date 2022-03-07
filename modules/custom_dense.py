
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.ops import core as core_ops

from tensorflow.python.eager import context
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow import math


class custom_dense(Dense):
    
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[1, self.units], #SW creating a kernel with shape 1 * number of filters, so it can be broadcast automatically when calling the layer
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
            self.built = True
    
    
    # def call(self, inputs):
        # return core_ops.dense(
        #     inputs,
        #     self.kernel,
        #     self.bias,
        #     self.activation,
        #     dtype=self._compute_dtype_object)
        
    def call(self, inputs):
        """Densely connected NN layer op.

        Arguments:
            inputs: `tf.Tensor` or `tf.SparseTensor`. Inputs to operation.
            kernel: `tf.Variable`. Matrix kernel.
            bias: (Optional) `tf.Variable`. Bias to add to outputs.
            activation: (Optional) 1-argument callable. Activation function to apply to
            outputs.
            dtype: (Optional) `tf.DType`. Dtype to cast `inputs` to.

        Returns:
            `tf.Tensor`. Output of dense connection.
        """
        kernel=self.kernel
        bias=self.bias
        activation=self.activation
        dtype=self._compute_dtype_object
        
        if dtype:
            if inputs.dtype.base_dtype != dtype.base_dtype:
                inputs = math_ops.cast(inputs, dtype=dtype)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            if isinstance(inputs, sparse_tensor.SparseTensor):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, kernel)
        # Broadcast kernel to inputs.
        else:
            # outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
            outputs = math.multiply(inputs, kernel)
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if bias is not None:
            outputs = nn_ops.bias_add(outputs, bias)

        if activation is not None:
            outputs = activation(outputs)

        return outputs
