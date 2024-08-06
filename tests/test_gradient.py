import torch

def load_tensor(file_name, map_location, set_grad=True):
    test_tensor = torch.load(file_name, map_location=map_location).cuda()
    test_tensor.requires_grad = set_grad
    return test_tensor

def convolve(blur_kernel, image, set_gradient=False):
    channels, kernel_size = image.size() [1], blur_kernel.size() [2]

    # Define the blur convolution function
    blur = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                        kernel_size=(kernel_size, kernel_size),
                        groups=channels, bias=False, padding=0,
                        stride=1, device=image.device)

    blur.weight.data = blur_kernel
    blur.weight.requires_grad = set_gradient

    # Convolve blur_kernel over image
    output_image = blur(image)

    return output_image

def transpose_convolve(kernel, image, set_gradient=False):
    channels, kernel_size = image.size() [1], kernel.size() [2]

    # Define the transpose convolution function
    conv_transposed = torch.nn.ConvTranspose2d(in_channels=channels,
                                            out_channels=channels,
                                            groups=channels,
                                            kernel_size=(kernel_size, kernel_size),
                                            padding=0, stride=1,
                                            bias=False, device=image.device)

    conv_transposed.weight.data = kernel
    conv_transposed.requires_grad = set_gradient

    # Convolve transpose over the image
    output_image = conv_transposed(image)
    return output_image


if torch.cuda.is_available():
    map_location = 'cuda:0'
else:
    map_location = 'cpu'

# Alpha values
sqrt_alphas_cumprod_tensor = load_tensor('gradient_logs/sqrt_alphas_cumprod_15.pt', map_location)
sqrt_one_minus_alphas_cumprod_tensor = load_tensor('gradient_logs/sqrt_one_minus_alphas_cumprod_15.pt', map_location)

# Rest
eps = load_tensor('gradient_logs/eps_15.pt', map_location)
kernel = load_tensor('gradient_logs/kernel_15.pt', map_location)
print(kernel)
mean = load_tensor('gradient_logs/mean_15.pt', map_location)
y = load_tensor('gradient_logs/y_15.pt', map_location)

channels, kernel_size = eps.size() [1], kernel.size() [2]

blur_kernel_A = torch.nn.Conv2d(in_channels=channels,out_channels=channels,
                                kernel_size=(kernel_size, kernel_size), groups=channels, bias=False,
                                padding=0, stride=1, device=kernel.device)

transpose_kernel = torch.nn.ConvTranspose2d(in_channels=channels,
                                         out_channels=channels,
                                         groups=channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         padding=0, stride=1,
                                         bias=False, device=kernel.device)

blur_kernel_A.weight.data = kernel.clone()
blur_kernel_A.weight.requires_grad = True

transpose_kernel.weight.data = kernel.clone()
transpose_kernel.weight.requires_grad = True

# print(alphas_cumprod_tensor.requires_grad, one_minus_alphas_cumprod_tensor.requires_grad)
# print(eps.requires_grad, kernel.requires_grad, mean.requires_grad, y.requires_grad)

dummy_optimizer = torch.optim.Adam(list(blur_kernel_A.parameters()), lr=1e-3)

A_eps = blur_kernel_A(eps)
y_t = sqrt_alphas_cumprod_tensor * y + sqrt_one_minus_alphas_cumprod_tensor * A_eps

A_mu = blur_kernel_A(mean)

output = - torch.norm(A_mu - y_t) ** 2

gradient = torch.autograd.grad(output, mean) [0]

print(gradient.size())
# print("pytorch gradient")
# print(gradient)

with torch.no_grad():
    A_eps_c = convolve(kernel, mean)
    y_t_c = sqrt_alphas_cumprod_tensor * y + sqrt_one_minus_alphas_cumprod_tensor * A_eps_c
    kernel_mu_convolve = convolve(kernel, mean)
    kernel_mu_convolve = kernel_mu_convolve - y_t
    analytical_gradient = -2 * (transpose_convolve(kernel, kernel_mu_convolve))
    # print("analytical gradient")
    # print(analytical_gradient)
    mse_diff = (analytical_gradient - gradient)**2
    diff_sum, max_diff, min_diff = torch.sum(mse_diff), torch.max(mse_diff), torch.min(mse_diff)
    print(f"difference gradient sum: {diff_sum}, max diff: {max_diff}, min diff: {min_diff}")
