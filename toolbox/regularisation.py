import torch
from toolbox.tools import find_device
from hypernet.mnet.cnn import CNN_tgt


# Function to calculate the l2 regularization
def l2reg(W_star, W):
    l2 = (W_star - W).pow(2).sum()
    return l2


# Regularization used in continual learning
def output_reg(hnet, hnet_star, embeddings, n_tasks, occupancy_tracker, gpu):
    reg = torch.tensor(0.0, requires_grad=True).to(find_device(gpu))
    for task in range(n_tasks):
        # Skip regularization if task is unlearnt
        if occupancy_tracker[task] == 0:
            continue
        e = embeddings[task]
        # hnet_star.eval()
        with torch.no_grad():
            w_star, b_star, nw_star, nb_star, dw_star, db_star = hnet_star(e)
        w, b, nw, nb, dw, db = hnet(e)
        param_star = torch.cat(
            [
                w_star.view(-1),
                b_star.view(-1),
                nw_star.view(-1),
                nb_star.view(-1),
                dw_star.view(-1),
                db_star.view(-1),
            ]
        )
        param = torch.cat(
            [
                w.view(-1),
                b.view(-1),
                nw.view(-1),
                nb.view(-1),
                dw.view(-1),
                db.view(-1),
            ]
        )
        reg += l2reg(param_star, param)
    return reg / n_tasks


# Regularization terms used in unlearning -------------------------------------


# Remembrance regularization term
def remember_reg(
    hnet, hnet_star, embeddings, forget_task, n_tasks, occupancy_tracker, gpu
):
    reg = torch.tensor(0.0, requires_grad=True).to(find_device(gpu))
    for task in range(n_tasks):
        if task == forget_task:
            continue
        if occupancy_tracker[task] == 0:
            continue
        e = embeddings[task]
        # hnet_star.eval()
        with torch.no_grad():
            w_star, b_star, nw_star, nb_star, dw_star, db_star = hnet_star(e)
        w, b, nw, nb, dw, db = hnet(e)
        param_star = torch.cat(
            [
                w_star.view(-1),
                b_star.view(-1),
                nw_star.view(-1),
                nb_star.view(-1),
                dw_star.view(-1),
                db_star.view(-1),
            ]
        )
        param = torch.cat(
            [
                w.view(-1),
                b.view(-1),
                nw.view(-1),
                nb.view(-1),
                dw.view(-1),
                db.view(-1),
            ]
        )
        reg += l2reg(param_star, param)
    n_remember_tasks = occupancy_tracker.sum() - 1
    return reg / n_remember_tasks


# Forgetting regularization term
def noisy_forget(hnet, forget_task, embeddings, n_sampling, partial=False):
    e = embeddings[forget_task]
    w, b, nw, nb, dw, db = hnet(e)
    if partial:
        lw = w[-10 * 2048 :]
        lb = b[-10:]
        param = torch.cat([lw.view(-1), lb.view(-1)])
    else:
        param = torch.cat(
            [
                w.view(-1),
                b.view(-1),
                nw.view(-1),
                nb.view(-1),
                dw.view(-1),
                db.view(-1),
            ]
        )
    # noise = torch.randn_like(param)
    error = torch.zeros(1, requires_grad=True, device=param.device)
    for sample in range(n_sampling):
        with torch.no_grad():
            noise = torch.randn_like(param)
        error = error + (param - noise).pow(2).sum()
    return error / n_sampling


def point_forget(hnet, forget_task, embeddings, point, partial=False):
    e = embeddings[forget_task]
    w, b, nw, nb, dw, db = hnet(e)
    if partial:
        lw = w[-10 * 2048 :]
        lb = b[-10:]
        param = torch.cat([lw.view(-1), lb.view(-1)])
    else:
        param = torch.cat(
            [
                w.view(-1),
                b.view(-1),
                nw.view(-1),
                nb.view(-1),
                dw.view(-1),
                db.view(-1),
            ]
        )
    return (param - point).pow(2).sum()


def norm_forget(hnet, forget_task, embeddings, partial=False):
    e = embeddings[forget_task]
    w, b, nw, nb, dw, db = hnet(e)
    param = torch.cat(
        [
            w.view(-1),
            b.view(-1),
            nw.view(-1),
            nb.view(-1),
            dw.view(-1),
            db.view(-1),
        ]
    )
    return param.norm()


def norm_partial(hnet, forget_task, embeddings):
    e = embeddings[forget_task]
    w, b, _, _, _, _ = hnet(e)
    tgt_w = w[: 7 * 7 * 64 * 3]
    tgt_b = b[:64]
    param = torch.cat([tgt_w.view(-1), tgt_b.view(-1)])
    return param.norm()


def cnn_formulate(arch, gpu):
    in_dim = arch["in_dim"]
    kernels = arch["kernels"]
    linear_layers = arch["linear_layers"]
    return CNN_tgt(in_dim, kernels, linear_layers).to(find_device(gpu))


# Kaiming forgetting
def kaiming_forget(hnet, forget_task, embeddings, mnet_arch, gpu):
    e = embeddings[forget_task]
    W = hnet(e)
    forget_net = cnn_formulate(mnet_arch, gpu)
    forget_net_params = [p for p in forget_net.parameters()]
    tgt = torch.cat([p.view(-1) for p in forget_net_params])
    print(W.shape)
    print(tgt.shape)
    for name, param in forget_net.named_parameters():
        print(name, param.shape)
    return (W - tgt).pow(2).sum()
