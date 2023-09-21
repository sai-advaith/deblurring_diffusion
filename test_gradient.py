import torch

def load_tensor(file_name, map_location, set_grad=True):
    test_tensor = torch.load(file_name, map_location=map_location).cuda()
    test_tensor.requires_grad = set_grad
    return test_tensor

def convolve(blur_kernel, image, set_gradient=False):
    channels, kernel_size = image.size() [1], blur_kernel.size() [2]

    # Define the blur convolution function
    blur = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                           kernel_size=kernel_size, groups=channels, bias=False,
                           padding=kernel_size // 2, device=blur_kernel.device)
    blur.weight.data = blur_kernel
    blur.weight.requires_grad = set_gradient

    # Convolve blur_kernel over image
    output_image = blur(image)
    
    return output_image


if torch.cuda.is_available():
    map_location = 'cuda:0'
else:
    map_location = 'cpu'

# Alpha values
alphas_cumprod_tensor = load_tensor('gradient_logs/alphas_cumprod.pt', map_location)
one_minus_alphas_cumprod_tensor = load_tensor('gradient_logs/one_minus_alphas_cumprod.pt', map_location)

# Rest
eps = load_tensor('gradient_logs/eps.pt', map_location)
kernel = load_tensor('gradient_logs/blur_kernel.pt', map_location)
mean = load_tensor('gradient_logs/mean.pt', map_location)
y = load_tensor('gradient_logs/y.pt', map_location)

# print(alphas_cumprod_tensor.requires_grad, one_minus_alphas_cumprod_tensor.requires_grad)
# print(eps.requires_grad, kernel.requires_grad, mean.requires_grad, y.requires_grad)

print(kernel.size(), eps.size())
A_eps = convolve(kernel, eps, set_gradient=True)
y_t = alphas_cumprod_tensor * y + one_minus_alphas_cumprod_tensor * A_eps

A_mu = convolve(kernel, mean, set_gradient=True)

loss = torch.nn.MSELoss()
output = loss(A_mu, y_t)

gradient = torch.autograd.grad(output, mean)
# print(gradient)
with torch.no_grad():
    kernel_mu_convolve = convolve(kernel, mean)
    kernel_mu_convolve = kernel_mu_convolve - y_t
    kernel_tranpose = kernel.transpose(2, 3)

    analytical_gradient = gradient = -2 * convolve(kernel_tranpose, kernel_mu_convolve)
    print((analytical_gradient - gradient)**2)
