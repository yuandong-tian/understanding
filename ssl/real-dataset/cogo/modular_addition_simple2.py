from collections import defaultdict
import json
import os
from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from modular_addition_load import process_one
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup

import hydra
import itertools

import math
import logging
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

import common_utils
from muon_opt import MuonEnhanced

def compute_diag_off_diag_avg(kernel):
    diagonal_avg = kernel.abs().diag().mean().item()
    off_diag_avg = (kernel.abs().sum() - kernel.abs().diag().sum()) / (kernel.shape[0] * (kernel.shape[0] - 1))
    return diagonal_avg, off_diag_avg

def fit_diag_11(kernel):
    # fit the diagonal to be 1 and the off-diagonal to be 1/10
    # For complex kernels, work with real part
    if kernel.is_complex():
        kernel = kernel.real
    diagonal_mean = kernel.diag().mean().item()
    off_diag_mean = (kernel.sum() - kernel.diag().sum()) / (kernel.shape[0] * (kernel.shape[0] - 1))
    estimated_kernel = (diagonal_mean - off_diag_mean) * torch.eye(kernel.shape[0], dtype=kernel.dtype).to(kernel.device) + off_diag_mean 
    return torch.norm(estimated_kernel - kernel) / torch.norm(kernel)

# Define the modular addition function
def modular_addition(ops, mods):
    accu_ops = [0] * len(mods)
    for op in ops:
        accu_ops = [ (o + accu_op) % mod for o, accu_op, mod in zip(op, accu_ops, mods) ]
    return tuple(accu_ops)

def generate_modular_addition_dataset(M, num_of_ops=2):
    if isinstance(M, int):
        orders = [M]
    else: 
        orders = [ int(v) for v in M.split("x") ]

    cum_orders = [orders[0]]
    for o in orders[1:]:
        cum_orders.append(cum_orders[-1] * o)

    def flattern(xs):
        return sum( x * order for x, order in zip(xs[1:], cum_orders[:-1]) ) + xs[0]

    data = []
    # if we have multiple ops, do itertools.product for each op
    iters = itertools.product(*[ itertools.product(*(range(v) for v in orders)) for _ in range(num_of_ops) ])

    for ops in iters:
        z = modular_addition(ops, orders)
        record = [flattern(o) for o in ops]
        record.append(flattern(z))
        data.append(tuple(record))

    return data, math.prod(orders)

def generate_perm_dataset(M):
    # M is the size of the symmetric group
    g = SymmetricGroup(M)
    elements = { perm : i for i, perm in enumerate(g.generate_schreier_sims()) }
    
    # do a permutation
    data = []
    for g1, i in elements.items():
        for g2, j in elements.items():
            k = elements[g1 * g2]
            data.append((i, j, k))

    return data, int(g.order())

def to_zero_based_table(table_1b: List[List[int]], index_base: int = 1) -> List[List[int]]:
    off = index_base
    return [[x - off for x in row] for row in table_1b]

def triples_from_table(tbl0: List[List[int]]) -> List[Tuple[int,int,int]]:
    n = len(tbl0)
    return [(i, j, tbl0[i][j]) for i in range(n) for j in range(n)]

def load_non_abelian_collection(M, dk_max=2):
    # M is a index. 
    # Load the non-abelian collection from the file
    # Get all non-abelian group with max_k d_k == dk_max
    # Get the current folder of this script
    json_file = "/private/home/yuandong/luckmatters/ssl/real-dataset/cogo/smallgroups_nonabelian_upto_128.jsonl"
    data = [ json.loads(line) for line in open(json_file, "r") ]

    # find rec so that rec["irrep_degrees"] == dk_max
    data = [ rec for rec in data if max(rec["irrep_degrees"]) == dk_max ]

    print(f"Found {len(data)} non-abelian groups with max_k d_k == {dk_max}")

    # Load the group, get the cayley table
    rec = data[M]
    tbl0 = to_zero_based_table(rec["table"], rec.get("index_base", 1))
    triples = triples_from_table(tbl0)

    rec["name"] = rec["name"].replace("\'\'", "")
    print(f"SmallGroup({rec['order']},{rec['smallgroup_id']})  name={rec['name']}")
    print(f"  num_irreps={rec['num_irreps']}  first irrep degrees={rec['irrep_degrees'][:10]}{'...' if len(rec['irrep_degrees'])>10 else ''}")
    print(f"  triples sample: {triples[:min(6, len(triples))]}  (total {len(triples)})")

    return triples, int(rec["order"])

def load_expression(M, expr):
    data = []
    for x in range(M):
        for y in range(M):
            z = eval(expr, {}, dict(x=x,y=y)) % M
            # put them to the dataset
            data.append((x, y, z))
    return data, M

nll_criterion = nn.CrossEntropyLoss().cuda()

def compute_loss(outputs, labels, loss_type):
    loss = 0
    for i, o in enumerate(outputs):
        # Convert complex to real if needed (take real part)
        if o.is_complex():
            o = o.real
        if loss_type == "nll":
            loss = loss + nll_criterion(o, labels[:,i])
        elif loss_type == "mse":
            o_zero_mean = o - o.mean(dim=1, keepdim=True)
            loss = loss + o_zero_mean.pow(2).sum(dim=1).mean() - 2 * o_zero_mean.gather(1, labels[:,i].unsqueeze(1)).mean() + 1 - 1.0 / o.shape[1] 
        else:
            raise RuntimeError(f"Unknown loss! {loss_type}")

    return loss

def test_model(model, X_test, y_test, loss_type):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        corrects = [None] * len(outputs)
        for (i, o) in enumerate(outputs):
            # Convert complex to real if needed (take real part) for max operation
            o_real = o.real if o.is_complex() else o
            _, predicted = torch.max(o_real.data, 1)
            corrects[i] = (predicted == y_test[:,i]).sum().item() / y_test.size(0)

        loss = compute_loss(outputs, y_test, loss_type).item()

    return corrects, loss

class StatsTracker:
    def __init__(self):
        self.stats = defaultdict(dict)

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def update(self, **kwargs):
        # Convert any 0-order tensor to scalar
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and len(v.size()) == 0:
                v = v.item()
            self.stats[self.epoch][k] = v

    def save(self, filename):
        torch.save(self.stats, filename)

# Define the neural network model
class ModularAdditionNN(nn.Module):
    def __init__(self, M, num_of_ops, hidden_size, activation="sqr", embed_trainable=False, use_bn=False, inverse_mat_layer_reg=None, other_layers=0, use_inner_product_act=False, use_complex_weights=False):
        super(ModularAdditionNN, self).__init__()
        self.use_complex_weights = use_complex_weights
        
        if not embed_trainable:
            self.embedding = nn.Embedding(M, M).requires_grad_(False)
            with torch.no_grad():
                self.embedding.weight[:] = torch.eye(M, M)
        else:
            self.embedding = nn.Embedding(M, M)

        if use_inner_product_act:
            self.Ws = nn.ModuleList([ nn.Linear(M, hidden_size, bias=False) for _ in range(num_of_ops) ])
        else: 
            self.W = nn.Linear(num_of_ops*M, hidden_size, bias=False)

        self.other_layers = nn.ModuleList([ nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(other_layers) ])
        self.V = nn.Linear(hidden_size, M, bias=False)

        # Convert weights to complex if needed
        if use_complex_weights:
            log.debug("Using complex weights!")
            if use_inner_product_act:
                for W in self.Ws:
                    W.weight.data = W.weight.data.to(torch.cfloat)
            else:
                self.W.weight.data = self.W.weight.data.to(torch.cfloat)
            
            for layer in self.other_layers:
                layer.weight.data = layer.weight.data.to(torch.cfloat)
            
            self.V.weight.data = self.V.weight.data.to(torch.cfloat)

        self.num_other_layers = other_layers
        self.use_inner_product_act = use_inner_product_act

        if use_bn and not use_complex_weights:
            # BatchNorm doesn't support complex tensors
            self.bn = nn.BatchNorm1d(hidden_size)
            self.use_bn = True
        else:
            self.use_bn = False
            if use_bn and use_complex_weights:
                log.debug("BatchNorm is disabled when using complex weights (not supported)")

        self.relu = nn.ReLU()
        self.num_of_ops = num_of_ops
        self.activation = activation
        self.M = M
        self.inverse_mat_layer_reg = inverse_mat_layer_reg

        if self.activation == "sqr": 
            self.act_fun = lambda x: x.pow(2)
        elif self.activation == "relu":
            self.act_fun = lambda x: self.relu(x)
        elif self.activation == "silu":
            self.act_fun = lambda x: x * torch.sigmoid(x)
        elif self.activation == "relusqr":
            self.act_fun = lambda x: self.relu(x) ** 2
        else:
            raise RuntimeError(f"Unknown activation = {self.activation}")
    
    def forward(self, x, Y=None, stats_tracker=None):
        assert x.shape[1] == self.num_of_ops, f"x.shape[1] = {x.shape[1]} != {self.num_of_ops}"
        # x = torch.relu(self.layer1(x))

        if self.use_inner_product_act:
            embeddings = [self.embedding(x[:,i]) for i in range(self.num_of_ops)]
            # Convert embeddings to complex if using complex weights
            if self.use_complex_weights:
                embeddings = [emb.to(torch.cfloat) for emb in embeddings]
            results = [ self.Ws[i](embeddings[i]) for i in range(self.num_of_ops) ]
            # Then do inner product and sum up
            x = torch.stack(results, dim=2)
            x = x.prod(dim=2)
        else:
            embed_concat = torch.concat([self.embedding(x[:,i]) for i in range(self.num_of_ops)], dim=1)
            # Convert embedding to complex if using complex weights
            if self.use_complex_weights:
                embed_concat = embed_concat.to(torch.cfloat)
            x = self.W(embed_concat) 
            if self.use_bn:
                x = self.bn(x)

            x = self.act_fun(x)

        self.x_before_layerc = x.clone()

        if stats_tracker is not None:
            x_zero_mean = x - x.mean(dim=0, keepdim=True)
            # Use simple matrix inversion
            if self.use_complex_weights:
                kernel = x_zero_mean.conj().t() @ x_zero_mean
            else:
                kernel = x_zero_mean.t() @ x_zero_mean
            diag_avg, off_diag_avg = compute_diag_off_diag_avg(kernel)
            log.debug(f"~F^t ~F: diag_avg = {diag_avg}, off_diag_avg = {off_diag_avg}, off_diag_avg / diag_avg = {off_diag_avg / diag_avg}")

            if self.use_complex_weights:
                kernel2 = x @ x.conj().t()
            else:
                kernel2 = x @ x.t()
            dist_from_ideal = fit_diag_11(kernel2)
            # zero mean
            kernel2 = kernel2 - kernel2.mean(dim=0, keepdim=True)
            diag_avg2, off_diag_avg2 = compute_diag_off_diag_avg(kernel2)
            log.debug(f"F F^t: diag_avg = {diag_avg2}, off_diag_avg = {off_diag_avg2}, off_diag_avg / diag_avg = {off_diag_avg2 / diag_avg2}, distance from ideal, {dist_from_ideal}")

            # backpropagated gradient norm
            if self.num_other_layers == 0:
                Vx = self.V(x)
                if self.use_complex_weights:
                    # Convert Y to complex and compute residual in complex domain
                    Y_complex = Y.to(torch.cfloat)
                    residual = Y_complex - Vx
                    # For complex weights, compute gradient norm using the complex weight
                    # The gradient w.r.t. complex weight is computed in complex domain
                    backprop_grad_norm = torch.norm(residual @ self.V.weight)
                else:
                    residual = Y - Vx
                    backprop_grad_norm = torch.norm(residual @ self.V.weight) 
                log.debug(f"Backpropagated gradient norm: {backprop_grad_norm}")
            else:
                backprop_grad_norm = None

            stats_tracker.update(**{
                "~F^t~F_off_diag_avg": off_diag_avg,
                "~F^t~F_diag_avg": diag_avg,
                "FF^t_dist_from_ideal": dist_from_ideal,
                "FF^t_off_diag_avg": off_diag_avg2,
                "FF^t_diag_avg": diag_avg2,
                "dF_norm": backprop_grad_norm,
            })

        if self.inverse_mat_layer_reg is not None and Y is not None:
            use_svd = True
            update_weightc = True
            with torch.no_grad():
                # Compute the matrix that maps input to target
                # X [bs, d]
                # Y [bs, d_out]
                # we want to find W so that X W = Y, where W = [d, d_out] 

                if use_svd:
                    # Compute the SVD of input 
                    U, s, Vt = torch.linalg.svd(x, full_matrices=False)
                    log.info(f"Using SVD, singular value [min, max] are {s.min(), s.max()}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    # Then we invert to get W.  
                    # 
                    # self.V.weight[:] = (Vt.t() @ ((U.t() @ Y) / (s[:,None] + self.inverse_mat_layer_reg))).t()
                    reg_diag = s / (s.pow(2) + self.inverse_mat_layer_reg)
                    log.debug(f"Using SVD, singular value [min, max] are {s.min(), s.max()}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    if update_weightc:
                        if self.use_complex_weights:
                            self.V.weight[:] = (Vt.conj().t() @ ((U.conj().t() @ Y.to(torch.cfloat)) * reg_diag[:,None])).t()
                        else:
                            self.V.weight[:] = (Vt.t() @ ((U.t() @ Y) * reg_diag[:,None])).t()
                else:
                    if self.use_complex_weights:
                        kernel = x.conj().t() @ x
                    else:
                        kernel = x.t() @ x
                    # Check if the kernel scale is the same as the self.inverse_mat_layer_reg
                    kernel_scale = kernel.diag().mean().item()
                    log.debug(f"Kernel scale is {kernel_scale}, inverse_mat_layer_reg is {self.inverse_mat_layer_reg}")
                    if update_weightc:
                        eye_mat = torch.eye(x.shape[1], dtype=x.dtype).to(x.device)
                        if self.use_complex_weights:
                            self.V.weight[:] = (torch.linalg.inv(kernel + self.inverse_mat_layer_reg * eye_mat) @ x.conj().t() @ Y.to(torch.cfloat)).t()
                        else:
                            self.V.weight[:] = (torch.linalg.inv(kernel + self.inverse_mat_layer_reg * eye_mat) @ x.t() @ Y).t()

        for layer in self.other_layers:
            x = x + layer(x)
            x = self.act_fun(x)

        output = self.V(x)
        
        return [output]

    def normalize(self):
        with torch.no_grad():
            if self.use_inner_product_act:
                for W in self.Ws:
                    # If the weight norm is > 1, scale down to 1, otherwise don't normalize
                    norms = W.weight.norm(dim=1, keepdim=True)
                    W.weight[:] /= norms.clamp(min=1)
            else:
                self.W.weight[:] -= self.W.weight.mean(dim=1, keepdim=True) 
                self.V.weight[:] -= self.V.weight.mean(dim=0, keepdim=True) 

    def scale_down_top(self):
        # Scale down the top layer
        with torch.no_grad():
            cnorm = self.V.weight.norm() 
            if cnorm > 1:
                self.V.weight[:] /= cnorm


@hydra.main(config_path="config", config_name="dyn_madd.yaml")
def main(args):
    # Set random seed for reproducibility
    log.info(common_utils.print_info(args))
    common_utils.set_all_seeds(args.seed)
    # torch.manual_seed(args.seed)

    scaling_law_correction = 1

    # Generate dataset
    if args.group_type == "modular_addition":
        dataset, group_order = generate_modular_addition_dataset(args.M, args.num_of_ops)
    elif args.group_type == "sym":
        dataset, group_order = generate_perm_dataset(args.M)
    elif args.group_type == "collection":
        # In this case, M becomes a index. 
        dataset, group_order = load_non_abelian_collection(args.M, dk_max=args.group_collection_max_dk)
        scaling_law_correction = args.group_collection_max_dk / 2 
    elif args.group_type == "expression":
        # This is not necessarily a group
        # in this case, args.M is treated as a number and we use expression to compute the output
        dataset, group_order = load_expression(int(args.M), args.expression)
    else:
        raise RuntimeError(f"Unknown group type = {args.group_type}")

    dataset_size = len(dataset)

    # Prepare data for training and testing
    X = torch.LongTensor(dataset_size, args.num_of_ops)
    # Use 
    labels = torch.LongTensor(dataset_size, 1)

    for i, record in enumerate(dataset):
        for j, op in enumerate(record[:-1]):
            X[i, j] = op
        labels[i] = record[-1]

    y = labels

    # compute the test_size if use_critical_ratio is true
    if args.use_critical_ratio:
        # critical ratio delta
        test_size = 1 - scaling_law_correction * math.log(group_order) / group_order * (args.critical_ratio_multiplier - args.critical_ratio_delta) 
        test_size = max(min(test_size, 1), 0)
        log.warning(f"Use critical ratio has set. test_size = {test_size}")
    else:
        test_size = args.test_size
        log.warning(f"Use specified test_size = {test_size}")

    if args.load_dataset_split is not None:
        # Load dataset
        data = torch.load(args.load_dataset_split)
        train_indices = data["train_indices"]
        test_indices = data["test_indices"]

        X_train = X[train_indices, :]
        y_train = y[train_indices]

        X_test = X[test_indices, :]
        y_test = y[test_indices]
    
    else:
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=args.seed)

    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()

    # Initialize the model, loss function, and optimizer
    if args.set_weight_reg is not None:
        assert args.loss_func == "mse", "only MSE loss can use set_weight_reg != None" 

    model = ModularAdditionNN(group_order, args.num_of_ops, args.hidden_size, 
                              embed_trainable=args.embed_trainable,
                              activation=args.activation, 
                              use_bn=args.use_bn, 
                              inverse_mat_layer_reg=args.set_weight_reg, 
                              other_layers=args.other_layers,
                              use_inner_product_act=args.use_inner_product_act,
                              use_complex_weights=args.use_complex_weights)

    model = model.cuda()

    if args.optim == "sgd":
        optimizers = [optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)]
    elif args.optim == "adam":
        optimizers = [optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)]
    elif args.optim == "adamw":
        optimizers = [optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)]
    elif args.optim == "muon":
        optimizers = [
            optim.Adam(model.V.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
            MuonEnhanced(
                model.W.parameters(), 
                beta1 = 0.9,
                beta2 = 0.99,
                lr=args.learning_rate, 
                weight_decay=args.weight_decay, 
                use_bf16=False, 
                nesterov=False, 
                update_rms_compensate=False,
                update_spectral_compensate=False 
            )
        ]
    else:
        raise RuntimeError(f"Unknown optimizer! {args.optim}")

    # Create learning rate schedulers
    schedulers = []
    if hasattr(args, 'lr_decay_type') and args.lr_decay_type != "none":
        for opt in optimizers:
            if args.lr_decay_type == "step":
                scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
            elif args.lr_decay_type == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay_rate)
            elif args.lr_decay_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            elif args.lr_decay_type == "multistep":
                milestones = args.lr_decay_milestones if hasattr(args, 'lr_decay_milestones') and args.lr_decay_milestones else [args.lr_decay_step]
                scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.lr_decay_rate)
            else:
                raise RuntimeError(f"Unknown lr_decay_type: {args.lr_decay_type}")
            schedulers.append(scheduler)
        log.info(f"Using learning rate decay: {args.lr_decay_type} with rate {args.lr_decay_rate}")
    else:
        log.info("No learning rate decay (lr_decay_type is 'none' or not specified)")

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-4)

    results = []

    # Get a one hot y_train
    Y_train = F.one_hot(y_train.squeeze(), num_classes=group_order)
    Y_train = Y_train - 1.0 / group_order

    stats_tracker = StatsTracker()

    # Training loop
    for epoch in range(args.num_epochs):
        stats_tracker.set_epoch(epoch)
        # Test the model
        train_accuracies, train_loss = test_model(model, X_train, y_train, args.loss_func)
        test_accuracies, test_loss = test_model(model, X_test, y_test, args.loss_func)

        train_acc = train_accuracies[0]
        test_acc = test_accuracies[0]

        log.info(f"Train Accuracy/Loss: {train_acc}/{train_loss}")
        log.info(f"Test Accuracy/Loss: {test_acc}/{test_loss}\n")

        stats_tracker.update(**{
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })

        if args.save_interval is not None and (epoch % args.save_interval == 0 or epoch < args.init_save_range):
            results.append(dict(epoch=epoch, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss))

            filename = f"model{epoch:05}_train{train_acc:.2f}_loss{train_loss:.4f}_test{test_acc:.2f}_loss{test_loss:.4f}.pt" 

            data = dict(model=model.state_dict(), results=results) 

            torch.save(data, filename)

        model.train()
        
        [ opt.zero_grad() for opt in optimizers ]
        
        # Forward pass
        outputs = model(X_train, Y=Y_train, stats_tracker=stats_tracker)

        # loss = criterion(outputs, y_train)
        loss = compute_loss(outputs, y_train, args.loss_func)
        
        # Backward and optimize
        loss.backward()

        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         print(model.W.weight.grad.norm())
        # import pdb
        # pdb.set_trace()
        [ opt.step() for opt in optimizers ]

        if args.normalize:
            model.normalize()

        if args.scale_down_top:
            model.scale_down_top()

        # Update learning rate
        if schedulers:
            for scheduler in schedulers:
                scheduler.step()
            if epoch % args.eval_interval == 0:
                all_lrs = []
                for scheduler in schedulers:
                    all_lrs.extend(scheduler.get_last_lr())
                lr_str = ', '.join([f'{lr:.6f}' for lr in all_lrs])
                log.info(f'Epoch [{epoch}/{args.num_epochs}], Loss: {loss.item():.4f}, LR: [{lr_str}]')
        elif epoch % args.eval_interval == 0:
            log.info(f'Epoch [{epoch}/{args.num_epochs}], Loss: {loss.item():.4f}')

    # save the stats_tracker
    stats_tracker.save("stats_tracker.pt")

    if args.post_process:
        # Process the data and save to a final file.
        log.info("Post-Processing data ...")
        entry = process_one(os.getcwd())

        log.info("Saving ... ")
        torch.save(entry, "./data.pth")

    print(os.getcwd())

if __name__ == '__main__':
    main()
