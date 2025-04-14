"""Main script for the solution."""

import numpy as np
import pandas as pd
import argparse, matplotlib.pyplot as plt


import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


if __name__ == '__main__':
    args = _get_args()
    X, y = npnn.load_mnist(args.dataset)

    # TODO
    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]

    train = npnn.Dataset(X[:50000], y[:50000], batch_size=32)
    val = npnn.Dataset(X[50000:], y[50000:], batch_size=32)
    
    
    
     # Select optimizer
    if args.opt == "SGD":
        optimizer = npnn.SGD(args.lr)
    elif args.opt == "Adam":
        optimizer = npnn.Adam(args.lr)
    else:
        raise ValueError("Unsupported optimizer")
    
    # Create model (see npnn/model.py)
    # Train for args.epochs
    # Create the model

    modules=[
        npnn.Flatten(),  
        npnn.Dense(784, 256),  
        npnn.ELU(),  
        npnn.Dense(256,64),
        npnn.ELU(),
        npnn.Dense(64,10),
        
    ]
    model = npnn.Sequential(modules, optimizer=optimizer, loss=  npnn.SoftmaxCrossEntropy() )
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # train for the given epochs
    for i in range(args.epochs):
        
        m_loss_t, m_acc_t= model.train(train)
        m_loss_val,  m_acc_val= model.test(val)
        train_losses.append(m_loss_t)
        train_accuracies.append(m_acc_t)
        val_losses.append(m_loss_val)
        val_accuracies.append(m_acc_val)

if __name__ == '__main__':
    print(f"Val accuracies: {val_accuracies}")
    # plot the curves
    plt.ylabel('Train loss and accuracy')
    plt.xlabel('Epoch')
    plt.plot(range(1,args.epochs+ 1), train_losses, label='Training loss')
    plt.plot(range(1,args.epochs+ 1), train_accuracies, label='Training accuracy')
    plt.legend()
    plt.savefig('training_loss_accuracy.png')
    plt.clf()
    plt.ylabel('Validation loss and accuracy')
    plt.xlabel('Epoch')
    plt.plot(range(1,args.epochs + 1), val_losses, label='validation loss')
    plt.plot(range(1,args.epochs + 1), val_accuracies, label='Validation accuracies')
    plt.legend()
    plt.savefig('validation_loss_accuracy.png')

    # Read the test data set
    X_test, y = npnn.load_mnist(args.test_dataset)
    if len(X_test.shape) > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)
    preds= np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
    np.save('mnist_test_pred.npy', preds)

    learning_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
    best_val_losses = []
    
    for lr in learning_rates:
        lr_val_losses =[]
        optimizer = npnn.SGD(lr)
        modules=[
            npnn.Flatten(),  
            npnn.Dense(784, 256),  
            npnn.ELU(),  
            npnn.Dense(256,64),
            npnn.ELU(),
            npnn.Dense(64,10),
        
        ]
        model = npnn.Sequential(modules, optimizer=optimizer, loss=  npnn.SoftmaxCrossEntropy() )
        for i in range(args.epochs):   
            m_loss_val,  _ = model.test(val)
            lr_val_losses.append(m_loss_val) 
               
        
        best_val_losses.append(min(lr_val_losses))
    
    # Plot learning rate comparison
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, best_val_losses, 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Validation Loss')
    plt.title('Learning Rates and Validation Loss')
    plt.xticks(learning_rates, [str(lr) for lr in learning_rates])
    plt.grid(True)
    plt.savefig('learning_rate_comparison.png')
    # Print best learning rate
    best_lr_idx = np.argmin(best_val_losses)
    print(f"Best learning rate: {learning_rates[best_lr_idx]}  validation loss: {best_val_losses[best_lr_idx]:.4f}")


    


    stats = pd.DataFrame()

    # Save statistics to file.
    # We recommend that you save your results to a file, then plot them
    # separately, though you can also place your plotting code here.
    if args.save_stats:
        stats.to_csv("data/{}_{}.csv".format(args.opt, 0.01))

    # Save predictions.
    if args.save_pred:
        X_test, _ = npnn.load_mnist("mnist_test.npz")
        y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)
