# tf_models/evaluate.py
import pickle, matplotlib.pyplot as plt

def plot_history(history, title):
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title(title); plt.ylabel("loss"); plt.xlabel("epoch")
    plt.legend(); plt.show()

# usage: load pickled histories and call plot_history()
