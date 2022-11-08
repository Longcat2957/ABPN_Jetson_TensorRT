import torch.backends.cudnn as cudnn
from libs.utils import benchmark
cudnn.benchmark = True

if __name__ == "__main__":
    benchmark("edgeSR_qat_jit.pt")