from ImageGenerator import ImageGenerator
from kornia.geometry import translate
import torch

def shiftImage(input, shiftMatrix, isBatch=True):
    if input.size != 3 or input.size != 4: 
        raise Exception("Size of image tensor must be 3 or 4. Supported structure of tensor must be (C, H, W) or (B, C, H, W)")

    if not isBatch:
        input = torch.unsqueeze(input, 0)
        shiftMatrix = torch.unsqueeze(shiftMatrix, 0) 
        B, _, H, W = input.shape
        output = torch.zeros_like(input)
        print(shiftMatrix.shape)
        
        """
        for i in range(B):
            # singleImage = torch.unsqueeze(torch.clone(input[i]),0)
            # singleShift = torch.clone(shifts[i])
            for j in range(H):
                singleRow = torch.unsqueeze_copy(input[i, :, j], 0)
                singleVector = torch.clone()
                for k in range(W):
                    singleShift = torch.clone(shiftMatrix[i, j, k])
                    output[i, :, j, :k] = translate(singleRow[:, :, j, :k],
                                               torch.unsqueeze(singleShift[k], 0),
                                               padding_mode="zeros",
                                               align_corners=False
        """
        return output           

    

