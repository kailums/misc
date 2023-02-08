import torch
import torch.distributed as dist
import onnx
import time
import psutil
import argparse
import os
import mpi4py

class TestModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.linear(x)
        out = dist.all_reduce(out)
       
        return out

class FusionModule(torch.nn.Module):
    def __init__(self, in_features, out_features, num_shards):
        super().__init__()
        assert out_features % num_shards == 0
        shard_out_features = out_features // num_shards
        m_list = []
        for i in range(num_shards):
            l = torch.nn.Linear(in_features, shard_out_features)
            m_list.append(l)

        self.m=torch.nn.ModuleList(m_list)

    def forward(self, x):
        res = []
        sync = []
        for l in self.m:
            out = l(x)
            s = dist.all_reduce(out, async_op=True)
            sync.append(s)
            res.append(out)
        torch.cuda.synchronize()
        #for s in sync:
        #    s.wait()
        return torch.concat(res, dim=1)

def run_torch(args, model, inputs, banner, local_rank):
    loop_cnt = args.loop_cnt
    log_interval = args.interval

    def run():
        with torch.autograd.profiler.emit_nvtx(True):
            with torch.no_grad():
                out = model(*inputs)
                return out
    # warmup
    out = run()
   
    end = time.time()
    for i in range(loop_cnt):
        out = run()
       
        if local_rank == 0 and i % log_interval == 0:
            cost = time.time() - end
            print(f'[{banner}] iter[{i}] cost: {cost} avg: {cost / log_interval}')
            end = time.time()
    return out

def init_dist():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK',0))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:9876', world_size=world_size, rank=rank)
    device = torch.device(local_rank)
    return device, local_rank


def main(args):
    batch = args.batch
    in_features = args.in_features
    out_features = args.out_features
    num_shards = args.shards

    #device = torch.device('cuda:0')
    device, local_rank = init_dist()
    torch.cuda.set_device(device)

    model = TestModule(in_features, out_features)
    fuse_model = FusionModule(in_features, out_features, num_shards)

    model.to(device)
    model.eval()
    fuse_model.to(device)
    fuse_model.eval()
    x = torch.randn(batch, in_features, device=device)

    if args.fp16:
        model.half()
        fuse_model.half()
        x = x.to(torch.float16)

    out1 = run_torch(args,model, [x], 'model', local_rank)

    out2 = run_torch(args, fuse_model, [x], 'fusion', local_rank)

    #print(f'out1 shape: {out1.shape}, out2 shape: {out2.shape}')

def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--loop-cnt', type=int, default=500)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--in-features', type=int)
    parser.add_argument('--out-features', type=int)
    parser.add_argument('--shards', type=int)
    parser.add_argument('--fp16', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arges()
    main(args)
