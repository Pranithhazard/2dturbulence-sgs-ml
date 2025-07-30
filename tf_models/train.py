# tf_models/train.py
import datetime
import tensorflow as tf
from .config import LOG_DIR, BATCH_SIZE, EPOCHS
from .data_loader import get_train_fw16_sig, get_train_fw32_sig, get_train_source_fw16, ...
from .model import create_sig_model, create_source_model

def get_tensorboard_cb(name):
    log_path = str(LOG_DIR / name / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

def train_sig_fw16():
    X, y = get_train_fw16_sig()
    model = create_sig_model(neurons=50)
    history = model.fit(
        X, y,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[ get_tensorboard_cb("fw16_sig") ]
    )
    model.save("saved_models/fw16_sig")
    return history

def main():
    hist1 = train_sig_fw16()
    hist2 = train_sig_fw32()
    hist3 = train_source_fw16()
    hist4 = train_source_fw32()
    # optionally pickle your histories here...
    
if __name__=="__main__":
    main()
