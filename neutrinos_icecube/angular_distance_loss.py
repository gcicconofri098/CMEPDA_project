import torch


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

    # print(y_true.shape)
    # print(y_pred.shape)
    az_pred = y_pred[:, 0]
    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]
    zen_pred = y_pred[:, 1]

    # print(az_pred.shape)
    # print(az_true.shape)
    # print(zen_pred.shape)
    # print(zen_true.shape)

    # Combine non-finite masks for all input tensors
    non_finite_mask = (
        ~torch.isfinite(az_true)
        | ~torch.isfinite(zen_true)
        | ~torch.isfinite(az_pred)
        | ~torch.isfinite(zen_pred)
    )

    # Apply the mask to filter out non-finite values
    az_true = az_true[~non_finite_mask]
    zen_true = zen_true[~non_finite_mask]
    az_pred = az_pred[~non_finite_mask]
    zen_pred = zen_pred[~non_finite_mask]

    # Pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)
    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # Scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # Scalar product of two unit vectors is always between -1 and 1
    # Clip to avoid numerical instability
    scalar_prod = torch.clamp(scalar_prod, -1.0, 1.0)

    # Convert back to an angle (in radians)
    return (torch.abs(torch.acos(scalar_prod)))

