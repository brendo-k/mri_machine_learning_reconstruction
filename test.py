from Models import varnet
import torch
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    x = torch.rand((1, 1, 640, 320), dtype=torch.complex64)
    mask = torch.ones((1, 1, 640, 320))
    net = varnet.VarNet(2, 2)
    params = sum(param.numel() for param in net.parameters())
    print(params)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            net(x, mask)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    prof.export_stacks('C:/Users/brend/AppData/Local/Temp/stack.txt', 'self_cpu_time_total')

