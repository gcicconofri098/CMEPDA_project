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
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/neutrinos_icecube/plots/test_graph_mean_180_epochs_20_hits_5_DNN_loss_40_0_001_loss_MSE_batch_256_dropout_simpler_mlp_knn_7_2files.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation RMSE")
    plt.plot(test_rmses, label="val")
    plt.plot(train_rmses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/neutrinos_icecube/plots/test_graph_mean_180_epochs_20_hits_5_DNN_RMSE_40_0_001_loss_MSE_batch_256_dropout_simpler_mlp_knn_7_2files.png")
    plt.close()

