import argparse
import multiprocessing
import random
import time

import torch.optim as optim
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from lirpa_integration import SemanticTransformation
from transformations.twisting import TwistingZ
from relaxations.interval import Interval

from data_processing import datasets

from auto_LiRPA import BoundedModule, BoundedTensor
from pointnet.model import PointNet
from util.argparse import parse_theta

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="MNIST", choices=["MNIST", "CIFAR"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=0.01, help='Target training epsilon')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="resnet", help='model name (mlp_3layer, cnn_4layer, cnn_6layer, cnn_7layer, resnet)')
parser.add_argument("--num_epochs", type=int, default=100, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=3,length=60", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')
parser.add_argument("--conv_mode", type=str, choices=["matrix", "patches"], default="matrix")
parser.add_argument("--save_model", type=str, default='')
parser.add_argument("--num_points", type=int, default=64)
parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg'], help="The pooling function to use")

args = parser.parse_args()


def Train(model, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust'):
    num_class = 40
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    for i, (data, _, labels) in enumerate(loader):
        start = time.time()
        data = data.float()
        labels = labels.squeeze()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-20:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        # bound input for Linf norm used only
        data_ub = data + eps
        data_lb = data - eps

        if list(model.parameters())[0].is_cuda:
            data, labels, c = data.cuda(), labels.cuda(), c.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=data_lb, x_U=data_ub)

        x = BoundedTensor(data, ptb)

        output = model(x)
        regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
        meter.update('CE', regular_ce.item(), x.size(0))
        meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0), x.size(0))

        if batch_method == "robust":
            if bound_type == "IBP":
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
            elif bound_type == "CROWN":
                lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
            elif bound_type == "CROWN-IBP":
                # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                    lb = clb * factor + ilb * (1 - factor)
            elif bound_type == "CROWN-FAST":
                # model.compute_bounds(IBP=True, C=c, method=None)
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
                lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

            # Pad zero at the beginning for each example, and use fake label "0" for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
        if batch_method == "robust":
            loss = robust_ce
        elif batch_method == "natural":
            loss = regular_ce
        if train:
            loss.backward()
            eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))
        if batch_method != "natural":
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
            # If any margin is < 0 this example is counted as an error
            meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Step 1: Initial original model as usual, see model details in models/example_feedforward.py and models/example_resnet.py
    model_ori = PointNet(
        number_points=args.num_points,
        num_classes=40,
        pool_function=args.pooling
    )

    if args.load:
        state_dict = torch.load(args.load)
        model_ori.load_state_dict(state_dict)
        print(state_dict)

    ## Step 2: Prepare dataset as usual

    train_data = datasets.modelnet40(num_points=args.num_points, split='train', rotate='z')
    test_data = datasets.modelnet40(num_points=args.num_points, split='test', rotate='none')

    train_data = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    test_data = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    dummy_input = torch.randn(2, args.num_points, 3)

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu': args.bound_opts, 'conv_mode': args.conv_mode}, device=args.device)

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    opt = optim.Adam(model.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    print("Model structure: \n", str(model_ori))

    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            Train(model, 1, test_data, eps_scheduler, norm, False, None, args.bound_type)
    else:
        timer = 0.0
        for t in range(1, args.num_epochs + 1):
            if eps_scheduler.reached_max_eps():
                # Only decay learning rate after reaching the maximum eps
                lr_scheduler.step()
            print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
            start_time = time.time()
            Train(model, t, train_data, eps_scheduler, norm, True, opt, args.bound_type)
            epoch_time = time.time() - start_time
            timer += epoch_time
            print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            print("Evaluating...")
            with torch.no_grad():
                Train(model, t, test_data, eps_scheduler, norm, False, None, args.bound_type)
            torch.save(model.state_dict(), args.save_model if args.save_model != "" else args.model)


if __name__ == "__main__":
    main(args)
