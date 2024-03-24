import time
import torch
from networkAlignmentAnalysis import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# B, D, S, C = 1024, 25, 784, 32
B, D, S, C = 1024, 800, 196, 64
input = torch.normal(0, 1, (B, D, S)).to(DEVICE)
weight = torch.normal(0, 1, (C, D)).to(DEVICE)


def get_align_usual(input, weight):
    var_stride = torch.mean(torch.var(input, dim=1), dim=0)
    align_stride = torch.stack([utils.alignment(input[:, :, i], weight) for i in range(S)], dim=1)
    return utils.weighted_average(align_stride, var_stride.view(1, -1), 1, ignore_nan=True)


def get_align_new(input, weight):
    var_stride = torch.mean(torch.var(input, dim=1), dim=0)
    cc = utils.batch_cov(input.transpose(0, 2))
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=2) / torch.sum(weight * weight, axis=1)
    prq = rq / torch.diagonal(cc, dim1=1, dim2=2).sum(1, keepdim=True)
    return utils.weighted_average(prq, var_stride.view(-1, 1), 0, ignore_nan=True)


au = get_align_usual(input, weight)
an = get_align_new(input, weight)
print("align_usual and align_new produce same result?", torch.allclose(au, an))

num_tests = 100

t = time.time()
for _ in range(num_tests):
    _ = get_align_usual(input, weight)
print("get_align_usual : time per test=", (time.time() - t) / num_tests)

t = time.time()
for _ in range(num_tests):
    _ = get_align_new(input, weight)
print("get_align_new : time per test=", (time.time() - t) / num_tests)
