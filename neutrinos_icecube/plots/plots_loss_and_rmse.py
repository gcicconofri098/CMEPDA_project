import matplotlib.pyplot as plt
import parameters
import hyperparameters

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

def loss_plots(train_losses, test_losses, train_rmses, test_rmses, learning_rate): #aggiungi come argomento il miglior lr

    sliced_test_losses = test_losses[1:]
    sliced_train_losses = train_losses[1:]
    sliced_test_rmses = test_rmses[1:]
    sliced_train_rmses = train_rmses[1:]


    #plots the loss and RMSE for all the epochs
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(sliced_test_losses, label="val")
    plt.plot(sliced_train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/neutrinos_icecube/plots/graph_mean_loss_" + str(hyperparameters.number_epochs) + "epochs_RLR_lr_" + str(learning_rate) +"_" +str(parameters.n_hits)+"_hits_"+str(hyperparameters.N_layers) +"_DNN_ "+str(hyperparameters.N_features) +"_loss_MAE_batch_ "+str(hyperparameters.batch_size) + "_dropout_simpler_mlp_knn_" + str(hyperparameters.n_neighbors) +".png")
    plt.close()

#model_lr_' + best_lr_str + 'early_stop_RLR_'+ str(parameters.n_hits)+'_hits_'+str(hyperparameters.N_layers) +'DNN_ '+str(hyperparameters.N_features) +'_loss_MAE_batch_ '+str(hyperparameters.batch_size) + '_dropout_simpler_mlp_knn_' + str(hyperparameters.n_neighbors) +'.pth

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation RMSE")
    plt.plot(sliced_test_rmses, label="val")
    plt.plot(sliced_train_rmses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/neutrinos_icecube/plots/graph_mean_RMSE_" + str(hyperparameters.number_epochs) + "epochs_RLR_lr_" + str(learning_rate) +"_"+ str(parameters.n_hits)+"_hits_"+str(hyperparameters.N_layers) +"_DNN_ "+str(hyperparameters.N_features) +"_loss_MAE_batch_ "+str(hyperparameters.batch_size) + "_dropout_simpler_mlp_knn_" + str(hyperparameters.n_neighbors) + ".png")
    plt.close()

