import numpy as np
import torch

if __name__ == '__main__':
    input = np.arange(24).reshape(2,3,4)
    input = torch.from_numpy(input)
    fang = input.view(2,-1,6,1)
    # fang = input.transpose(1,2)
    print(fang)