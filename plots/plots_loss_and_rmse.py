import matplotlib.pyplot as plt


def single_batch_loss_plots(train_losses, test_losses, train_rmses, test_rmses):

    """
    Plots the loss and the RMSE for a single batch


    """

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(test_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.savefig("graph_log_batch_mean_loss_30_0_0001_loss_MSE.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation RMSE")
    plt.plot(test_rmses, label="val")
    plt.plot(train_rmses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("RMSE")
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.savefig("graph_log_batch_mean_RMSE_30_0_0001_loss_MSE.png")
    plt.close()

def loss_plots(train_losses, test_losses, train_rmses, test_rmses):


    #plots the loss and RMSE for all the epochs
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(test_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("graph_mean_170_epochs_20_hits_5_DNN_loss_30_0_001_loss_MSE_batch_256_simpler_mlp_knn_7_2files.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation RMSE")
    plt.plot(test_rmses, label="val")
    plt.plot(train_rmses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("graph_mean_170_epochs_20_hits_5_DNN_RMSE_30_0_001_loss_MSE_batch_256_simpler_mlp_knn_7_2files.png")
    plt.close()

