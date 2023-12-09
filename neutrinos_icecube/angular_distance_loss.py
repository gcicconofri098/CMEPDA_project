import sys
import torch
import numpy as np
import logging

logging.basicConfig(filename='logs/custom_loss.log', level= logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def angular_dist_score(y_true, y_pred):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Args:
    y_true: target tensor
    y_pred: output tensor from the neural network

    Returns:
        _type_: _description_
    """


    logging.debug(y_true.shape)
    logging.debug(y_pred.shape)
    az_pred = y_pred[:, 0].clone().detach().numpy() #using torch.clone() and then torch.detach() to have 
    az_true = y_true[:, 0].numpy()
    zen_true = y_true[:, 1].numpy()
    zen_pred = y_pred[:, 1].clone().detach().numpy()

    logging.debug(az_pred.shape)
    logging.debug(az_true.shape)
    logging.debug(zen_pred.shape)
    logging.debug(zen_true.shape)


    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return torch.tensor(np.average(np.abs(np.arccos(scalar_prod))), requires_grad=True)
