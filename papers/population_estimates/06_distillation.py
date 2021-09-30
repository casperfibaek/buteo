yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import time
import os

sys.path.append(yellow_follow)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
)
from buteo.utils import timing
from buteo.machine_learning.ml_utils import create_step_decay, tpe
from model_trio_down import model_trio_down
from tensorflow_addons.activations import mish

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions / self.temperature,
                student_predictions / self.temperature,
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"
model_name = "distilled_01"

x_train = [
    np.concatenate(
        [
            np.load(folder + f"{place}/all_nonoise_RGBN.npy"),
            # np.load(folder + f"{place}/teacher2_RGBN.npy"),
        ]
    ),
    np.concatenate(
        [
            np.load(folder + f"{place}/all_nonoise_SAR.npy"),
            # np.load(folder + f"{place}/teacher2_SAR.npy"),
        ]
    ),
    np.concatenate(
        [
            np.load(folder + f"{place}/all_nonoise_RESWIR.npy"),
            # np.load(folder + f"{place}/teacher2_RESWIR.npy"),
        ]
    ),
]

y_train = np.concatenate(
    [
        np.load(folder + f"{place}/all_nonoise_label_area.npy"),
        # np.load(folder + f"{place}/teacher2_label_area.npy"),
    ]
)

shuffle_mask = np.random.permutation(y_train.shape[0])

for idx in range(len(x_train)):
    x_train[idx] = x_train[idx][shuffle_mask]

y_train = y_train[shuffle_mask]

x_test = [
    np.load(folder + f"{place}/test_RGBN.npy"),
    np.load(folder + f"{place}/test_SAR.npy"),
    np.load(folder + f"{place}/test_RESWIR.npy"),
]

y_test = np.load(folder + f"{place}/test_label_area.npy")

inception_blocks = 2
activation = mish
initializer = "glorot_normal"

student = model_trio_down(
    x_train[0].shape[1:],
    x_train[1].shape[1:],
    x_train[2].shape[1:],
    kernel_initializer=initializer,
    activation=activation,
    inception_blocks=inception_blocks,
    name=f"{model_name.lower()}",
)

teacher_path = "student_02_00"
teacher = tf.keras.models.load_model(teacher_path, custom_objects={"tpe": tpe})

distiller = Distiller(student=student, teacher=teacher)

lr = 0.0001
min_delta = 0.005

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

distiller.compile(
    alpha=0.1,
    temperature=10,
    optimizer=optimizer,
    metrics=["mse", "mae", tpe],  # tpe
    student_loss_fn=tf.losses.log_cosh,
    distillation_loss_fn=tf.losses.log_cosh,
)

epochs = [20, 10, 5]
bs = [256, 128, 64]

start = time.time()

for phase in range(len(bs)):
    use_epoch = np.cumsum(epochs)[phase]
    use_bs = bs[phase]
    initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

    distiller.fit(
        x=x_train,
        y=y_train,
        # validation_split=0.1,
        validation_data=(x_test, y_test),
        shuffle=True,
        epochs=use_epoch,
        initial_epoch=initial_epoch,
        verbose=1,
        batch_size=use_bs,
        use_multiprocessing=True,
        workers=0,
        callbacks=[
            LearningRateScheduler(
                create_step_decay(
                    learning_rate=lr,
                    drop_rate=0.75,
                    epochs_per_drop=3,
                )
            ),
            ModelCheckpoint(
                filepath=f"{outdir}{model_name.lower()}_" + "{epoch:02d}",
                save_best_only=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                min_delta=min_delta,
                restore_best_weights=True,
            ),
        ],
    )

distiller.save(f"{outdir}{model_name.lower()}")

timing(start)
