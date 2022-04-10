import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error


def tpe(y_true, y_pred):
    epsilon = 1e-7
    pred_sum = tf.math.reduce_sum(y_pred)
    true_sum = tf.math.reduce_sum(y_true)
    ratio = tf.math.divide(pred_sum, true_sum + epsilon)

    return ratio


def tpe_target(y_true, y_pred):
    epsilon = 1e-7
    pred_sum = tf.math.reduce_sum(y_pred)
    true_sum = tf.math.reduce_sum(y_true)
    ratio = tf.math.divide(pred_sum, true_sum + epsilon)

    return tf.math.abs(1 - ratio)


def mse_mae_mix_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return tf.math.multiply(mse, mae)


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


def load_mish():
    get_custom_objects().update({"Mish": Mish(mish)})


class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"

# From user2646087 @ GitHub
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric="val_loss", this_max=False, initial_weights=None):
        self.save_best_metric = save_best_metric
        self.max = this_max

        if initial_weights is not None:
            self.best_weights = initial_weights

        if this_max:
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, _epoch, logs=None):
        metric_value = abs(logs[self.save_best_metric])
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor=["loss", "val_loss"]):
        self.time_started = None
        self.time_finished = None
        self.monitor = monitor
        
    def on_train_begin(self, logs=None):
        self.time_started = datetime.now()
        print(f'\nTraining started: {self.time_started}\n')
        
    def on_train_end(self, logs=None):
        self.time_finished = datetime.now()
        train_duration = str(self.time_finished - self.time_started)
        print(f'\nTraining finished: {self.time_finished}, duration: {train_duration}')
        
        metrics = [] 
        for metric in self.monitor:
            str_val = str(logs[metric])
            before_dot = len(str_val.split(".")[0])

            spaces = 16 - (len(metric) + before_dot)
            if spaces <= 0:
                spaces = 1

            pstr = f"{metric}:{' ' * spaces}{logs[metric]:.4f}"
            metrics.append(pstr)

        print('\n'.join(metrics))


class OverfitProtection(tf.keras.callbacks.Callback):
    def __init__(self, difference=0.1, patience=3, offset_start=3, verbose=True):
        self.difference = difference
        self.patience = patience
        self.offset_start = offset_start
        self.verbose = verbose
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        
        if epoch < self.offset_start:
            return

        epsilon = 1e-7
        ratio = loss / (val_loss + epsilon)

        if (1.0 - ratio) > self.difference:
            self.count += 1

            if self.verbose:
                print(f"Overfitting.. Patience: {self.count}/{self.patience}")

        elif self.count != 0:
            self.count -= 1
        
        if self.count >= self.patience:
            self.model.stop_training = True

            if self.verbose:
                print(f"Training stopped to prevent overfitting. Difference: {ratio}, Patience: {self.count}/{self.patience}")


class LearningRateAdjuster(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch, decay_rate=0.95, decay_rate_epoch=10, set_at_end="end_rate", step_wise=True, verbose=True):
        self.start_epoch = start_epoch
        self.decay_rate = decay_rate
        self.decay_rate_epoch = decay_rate_epoch
        self.set_at_end = set_at_end
        self.step_wise = step_wise
        self.initial_epoch = 0
        self.decay_count = 0
        self.verbose = verbose
        self.initial_lr = None

    def on_epoch_end(self, epoch, logs=None):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * self.decay_rate ^ ((epoch - self.start_epoch) / self.decay_rate_epoch)

        if self.step_wise and epoch == (self.start_epoch + (self.decay_rate_epoch * (self.decay_count + 1))):
            if self.verbose:
                print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

            self.model.optimizer.lr.assign(new_lr)
            self.decay_count += 1

        elif not self.step_wise:
            if self.verbose:
                print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

            self.model.optimizer.lr.assign(new_lr)
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_lr is None:
            self.initial_lr = self.model.optimizer.lr.read_value()

    def on_train_end(self, logs=None):
        if self.set_at_end == "end_rate":
            pass
        elif isinstance(self.set_at_end, (int, float)):
            self.model.optimizer.lr.assign(self.set_at_end)
        elif self.set_at_end == "initial":
            self.model.optimizer.lr.assign(self.initial_lr)
