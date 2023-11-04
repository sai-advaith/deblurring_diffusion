# import torch
# from torch import nn


# def corr2d(X, K):

#     # Convolution in deep learning is a misnomer.
#     # In fact, it is cross-correlation.
#     # https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html
#     # This is equivalent as Conv2D that that input_channel == output_channel == 1 and stride == 1.

#     assert X.dim() == 2 and K.dim() == 2

#     h, w = K.shape
#     Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             Y[i, j] = (X[i:i + h, j:j + w] * K).sum()

#     return Y


# def get_sparse_kernel_matrix(K, h_X, w_X):

#     # Assuming no channels and stride == 1.
#     # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
#     # This is a little bit brain-twisting.

#     h_K, w_K = K.shape

#     h_Y, w_Y = h_X - h_K + 1, w_X - w_K + 1

#     W = torch.zeros((h_Y * w_Y, h_X * w_X))
#     for i in range(h_Y):
#         for j in range(w_Y):
#             for ii in range(h_K):
#                 for jj in range(w_K):
#                     W[i * w_Y + j, i * w_X + j + ii * w_X + jj] = K[ii, jj]

#     return W


# def conv2d_as_matrix_mul(X, K):

#     # Assuming no channels and stride == 1.
#     # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
#     # This is a little bit brain-twisting.

#     h_K, w_K = K.shape
#     h_X, w_X = X.shape

#     h_Y, w_Y = h_X - h_K + 1, w_X - w_K + 1

#     W = get_sparse_kernel_matrix(K=K, h_X=h_X, w_X=w_X)

#     Y = torch.matmul(W, X.reshape(-1)).reshape(h_Y, w_Y)

#     return Y


# def conv_transposed_2d_as_matrix_mul(X, K):

#     # Assuming no channels and stride == 1.
#     # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
#     # This is a little bit brain-twisting.

#     h_K, w_K = K.shape
#     h_X, w_X = X.shape

#     h_Y, w_Y = h_X + h_K - 1, w_X + w_K - 1

#     # It's like the kernel were applied on the output tensor.
#     W = get_sparse_kernel_matrix(K=K, h_X=h_Y, w_X=w_Y)

#     # Weight matrix tranposed.
#     Y = torch.matmul(W.T, X.reshape(-1)).reshape(h_Y, w_Y)

#     return Y


# def main():

#     X = torch.arange(81).reshape(9, 9).float()
#     K = torch.arange(25).reshape(5, 5).float()

#     K = K.view(1, K.shape[0], K.shape[0])
#     K = K.repeat(1, 1, 1, 1)
#     print(K.size())

#     print("X:")
#     print(X)
#     print("K:")
#     print(K)
#     print("Cross-Correlation:")
#     # Y = corr2d(X=X, K=K)
#     # print(Y)

#     conv = nn.Conv2d(in_channels=3,
#                      out_channels=3,
#                      groups=1,
#                      kernel_size=K.shape[2:],
#                      padding=0,
#                      stride=1,
#                      bias=False)
#     conv.weight.data = K#.unsqueeze(0).unsqueeze(0)
#     Z1 = conv(X.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach()
#     print("Convolution:")
#     print(Z1)
#     # assert torch.equal(Y, Z1)

#     print("Convolution as Matrix Multiplication:")
#     print(X.size(), K.squeeze(0).squeeze(0))
#     print(K.size())
#     Z2 = conv2d_as_matrix_mul(X=X, K=K.squeeze(0).squeeze(0))
#     print(Z2)
#     # assert torch.equal(Y, Z2)
#     assert torch.equal(Z1, Z2)

#     conv_transposed = nn.ConvTranspose2d(in_channels=1,
#                                          out_channels=1,
#                                          groups=1,
#                                          kernel_size=K.shape[2:],
#                                          padding=0,
#                                          stride=1,
#                                          bias=False)
#     conv_transposed.weight.data = K#.unsqueeze(0).unsqueeze(0)
#     Z3 = conv_transposed(Z1.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach()
#     print("Transposed Convolution:")
#     print(Z3)
#     # The shape will "go back".
#     assert Z3.shape == X.shape

#     print("Transposed Convolution as Matrix Multiplication:")
#     Z4 = conv_transposed_2d_as_matrix_mul(X=Z1, K=K.squeeze(0).squeeze(0))
#     print(Z4)
#     assert torch.equal(Z3, Z4)
#     assert Z4.shape == X.shape

#     return


# if __name__ == "__main__":

#     main()

def iteration_scheduler(t):
    max_iterations = 200  # The maximum number of iterations
    min_iterations = 50   # The minimum number of iterations
    t_start = 1300        # The initial timestep

    # Linearly increase the number of iterations as t decreases
    if t > t_start:
        iterations = max_iterations
    else:
        iterations = min_iterations + (max_iterations - min_iterations) * ((t_start - t) / t_start)
    
    return int(iterations)

# Example usage:
for t in range(1300, -1, -25):
    num_iterations = iteration_scheduler(t)
    print(f"At timestep t={t}, use {num_iterations} iterations to optimize A.")