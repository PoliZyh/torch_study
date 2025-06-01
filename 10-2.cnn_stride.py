import torch

input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
input = torch.tensor(input, dtype=torch.float32).view(1, 1, 5, 5) # B C W H

conv_layer = torch.nn.Conv2d(1, 1,  stride=2,kernel_size=3, bias=False)

kernel = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32).view(1, 1, 3, 3) # O I W H
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)