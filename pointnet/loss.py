import torch


def orthogonal_normalizer(matrix):
    # Expected shape: (batch_size, n, n)
    assert matrix.size()[1] == matrix.size()[2]
    matrix_size = matrix.size()[1]
    batch_size = matrix.size()[0]

    identity = torch.eye(n=matrix_size, device=matrix.device)
    identity = identity.expand(batch_size, matrix_size, matrix_size)

    # p='fro' (Frobenius-Norm) is broken on Cuda in torch 1.4.0. According to the discussion in the bug report,
    # p=2 has the same effect and works on Cuda https://github.com/pytorch/pytorch/issues/30704
    loss = torch.mean(torch.norm(torch.bmm(matrix.transpose(2, 1), matrix) - identity, p=2, dim=(1, 2)))
    return loss
