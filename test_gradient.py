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

channels, kernel_size = eps.size() [1], kernel.size() [2]

blur_kernel_A = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=kernel_size // 2, device=kernel.device)

blur_kernel_A.weight.data = kernel.clone()
blur_kernel_A.weight.requires_grad = True


# print(alphas_cumprod_tensor.requires_grad, one_minus_alphas_cumprod_tensor.requires_grad)
# print(eps.requires_grad, kernel.requires_grad, mean.requires_grad, y.requires_grad)

dummy_optimizer = torch.optim.Adam(list(blur_kernel_A.parameters()), lr=1e-3)

A_eps = blur_kernel_A(eps)
y_t = alphas_cumprod_tensor * y + one_minus_alphas_cumprod_tensor * A_eps

A_mu = blur_kernel_A(mean)

output = - torch.norm(A_mu - y_t) ** 2

gradient = torch.autograd.grad(output, mean) [0]

print(gradient.size())
print("pytorch gradient")
print(gradient)

with torch.no_grad():
    kernel_mu_convolve = convolve(kernel, mean)
    kernel_mu_convolve = kernel_mu_convolve - y_t
    kernel_tranpose = kernel.transpose(2, 3)

    analytical_gradient = -2 * convolve(kernel_tranpose, kernel_mu_convolve)
    print("analytical gradient")
    print(analytical_gradient)
    diff_sum = torch.sum((analytical_gradient - gradient)**2)
    print(f"difference gradient sum: {diff_sum}")
