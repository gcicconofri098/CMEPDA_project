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
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/neutrinos_icecube/plots/graph_mean_180_epochs_RLR_40_hits_5_DNN_loss_50_0_0005_loss_MAE_batch_512_dropout_simpler_mlp_knn_10_1file.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation RMSE")
    plt.plot(test_rmses, label="val")
    plt.plot(train_rmses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/neutrinos_icecube/plots/graph_mean_180_epochs_RLR_40_hits_5_DNN_RMSE_50_0_0005_loss_MAE_batch_512_dropout_simpler_mlp_knn_10_1file.png")
    plt.close()

