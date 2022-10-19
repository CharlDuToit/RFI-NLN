import tensorflow as tf
#import numpy as np

from .freeze_model import freeze_model

"""
Comment on 22 Aug 2022 by albertmundu: https://github.com/tensorflow/tensorflow/issues/32809
Comment referenced this source https://github.com/wandb/wandb/blob/latest/wandb/integration/keras/keras.py#L1025-L1073
"""

def get_flops(model=None, frozen_func=None, model_inputs=None) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """

    if model is None and frozen_func is None:
        raise ValueError(
            "model or frozen_func must not be None "
        )

    frozen_func = freeze_model(model, model_inputs) if frozen_func is None else frozen_func

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            #tf.compat.v1.profiler.ProfileOptionBuilder().time_and_memory()
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
    )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    #return (profiling.total_float_ops / 1e9) / 2
    return flops.total_float_ops // 2


#if __name__ == "__main__":
    #image_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None)
#    image_model = dummy_model()
#    x = tf.constant(np.random.randn(1, 100, 100, 1))
 #   print(get_flops(image_model, [x]))