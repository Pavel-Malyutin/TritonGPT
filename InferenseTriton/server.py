import logging

import numpy as np
import tensorflow as tf  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.mlp_random_tensorflow2.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

MODEL = tf.keras.models.load_model('model')


@batch
def _infer_gpt(inputs):
    output1_batch, _ = MODEL.predict(inputs)
    return [output1_batch]


with Triton() as triton:
    logger.info("Loading GPT model.")
    triton.bind(
        model_name="GPT",
        infer_func=_infer_gpt,
        inputs=[
            Tensor(name="inputs", dtype=np.int64, shape=(80,)),
        ],
        outputs=[
            Tensor(name="output", dtype=np.float32, shape=(1, 80, 20000)),
        ],
        config=ModelConfig(max_batch_size=16),
    )
    logger.info("Serving inference")
    triton.serve()
