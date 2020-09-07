import argparse
from pathlib import Path
import numpy as np
import tables
import pdb
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from generator import FaceGenerator, ValGenerator
from model import get_model
from model_2 import mae


# Additional
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# ---

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--appa_dir", type=str, required=True,
                        help="path to the APPA-REAL dataset")
    parser.add_argument("--utk_dir", type=str, default=None,
                        help="path to the UTK face dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="model name: 'ResNet50' or 'InceptionResNetV2'")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

# The __call__ method enables Python programmers to write classes where the instances behave like functions and can be called like a function. 
# When the instance is called as a function; if this method is defined, x(arg1, arg2, ...) is a shorthand for x.__call__(arg1, arg2, ...).

# This controls the learning rate based on the number of epochs
    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    appa_dir = args.appa_dir
    utk_dir = args.utk_dir
    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    y_true = []
    if model_name == "ResNet50":
        image_size = 224
    elif model_name == "InceptionResNetV2":
        image_size = 299

    elif model_name == "Custom":
        image_size = 224

    train_gen = FaceGenerator(appa_dir, utk_dir=utk_dir, batch_size=batch_size, image_size=image_size)
    val_gen = ValGenerator(appa_dir, batch_size=batch_size, image_size=image_size)
    model = get_model(model_name=model_name)
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss="mae", metrics=[mae, 'mse'])
    output_dir = Path(__file__).resolve().parent.joinpath(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                 ModelCheckpoint(str(output_dir) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_mae:.3f}.hdf5",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

    hist = model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)

    # Additional
    

    pdb.set_trace()


    Y_pred = model.predict_generator(val_gen, 23711 // 32+1)
    y_pred = np.argmax(Y_pred, axis=1)

    print('Confusion Matrix')  
    from conf_mat import keras_plot_confusion_mat

    keras_plot_confusion_mat(y_pred, y_true)
    pdb.set_trace()

    # matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print('Classification Report')
    target_names = list(range[1,101])
    print(classification_report(val_gen.classes, y_pred, target_names=target_names))
    

    plt.plot(history.history['mae'])
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    np.savez(str(output_dir.joinpath("history.npz")), history=hist.history)

    

    #Confution Matrix and Classification Report


    # -----

if __name__ == '__main__':
    main()
