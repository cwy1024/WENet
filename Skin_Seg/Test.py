import torch

A = torch.ones(2, 3, 3, 3)  # 形状 [8,24,128,128]
B = torch.ones(2, 1, 3, 3)    # 形状 [8,1,128,128]
C = A + B  # 自动广播B为 [8,24,128,128]，再相加
print(A)
print('=========================')
print(B)
print('=========================')
print(C)  # 输出: torch.Size([8, 24, 128, 128])