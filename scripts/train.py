# Code adapted or modified from MinkLoc3DV2 repo: https://github.com/jac99/MinkLoc3Dv2

import os
import sys
import argparse
import torch
from shutil import copyfile
import time
import errno
import json
import numpy as np
import tqdm
import pathlib
from torch.utils.tensorboard import SummaryWriter  # TensorBoard import

from transloc4d import evaluate_4drad_dataset, save_recall_results
from transloc4d.misc.utils import TrainingParams
from transloc4d.datasets import WholeDataset, make_dataloaders
from transloc4d.models import (make_losses, get_model)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def get_global_stats(phase, stats):
    if 'loss1' not in stats:
        s = f"{phase}  loss: {stats['loss']:.4f}    embedding norm: {stats['avg_embedding_norm']:.3f}   "
    else:
        s = f"{phase}  loss: {stats['loss']:.4f}    loss1: {stats['loss1']:.4f}   loss2: {stats['loss2']:.4f}   "

    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
             f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats and int(stats['best_positive_ranking']) != 0:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    return s

def get_stats(phase, stats):
    return get_global_stats(phase, stats['global'])

def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats

def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}

    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}
        embeddings = y['global']

        loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)

        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats

def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['global'])

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        loss, stats = loss_fn(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad

    # Delete intermediary values
    embeddings_l, embeddings, y, loss = None, None, None, None

    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                embeddings = y['global']
                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats
        
def mkdir_if_missing(dir_path):
    if not dir_path: return
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def do_train(params: TrainingParams, model_name, weights_folder, resume_filename=None, saved_state_dict=None):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    val_sets = [
        {
            "dataset": WholeDataset(os.path.join(params.dataset_folder, params.val_databases[i]), os.path.join(params.dataset_folder, params.val_queries[i])),
            "database_name": params.val_databases[i],
            "query_name": params.val_queries[i]
        } for i in range(len(params.val_databases))
    ]
    
    # Create model class
    model = get_model(params, device, resume_filename)

    print(f"==> Loaded model checkpoint: {model_name}")

    model_pathname = os.path.join(weights_folder, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('  Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    model.to(device)
    print('  Model device: {}'.format(device))

    # set up dataloaders
    dataloaders = make_dataloaders(params, validation=False)

    loss_fn = make_losses(params)

    # Training elements
    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    if saved_state_dict is None:
        start_epoch = 0
    else:
        start_epoch = saved_state_dict['epoch']

        optimizer.load_state_dict(saved_state_dict['optimizer'])
        scheduler.load_state_dict(saved_state_dict['lr_schedule'])

    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
        train_step_fn = multistaged_training_step

    sys.stdout = Logger(os.path.join(weights_folder, "log.txt"))

    # Training statistics
    stats = {'train': [], 'eval': []}

    train_metrics = {
        "best_epoch": 1,
        "best_r@1": 0,
        "metrics": []
    }
    
    # Initialize TensorBoard writer using weights_folder as log directory.
    writer = SummaryWriter(log_dir=weights_folder)
    outer = tqdm.tqdm(
        range(start_epoch + 1, params.epochs + 1),
        desc="==> Training Epoch",
        unit="epoch",
        leave=False,
        position=0,
        dynamic_ncols=True
    )

    for epoch in outer:
        metrics = {'epoch': epoch, 'train': {}, 'val': {}}
        current_val_recall = 0
        outer.write(f">> epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']}")

        # Training phase
        phase = 'train'
        running_stats = []
        # Wrap inner loop with a tqdm progress bar.
        data_loader = dataloaders['train']
        try:
            total_batches = len(data_loader)
        except TypeError:
            total_batches = None
        global_iter = iter(data_loader)
        with tqdm.tqdm(total=total_batches, desc=f"===> Batches", unit="batch", leave=False, position=1, dynamic_ncols=True) as pbar:
            count_batches = 0
            while True:
                count_batches += 1
                if params.debug and count_batches > 2:
                    break
                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    running_stats.append({'global': temp_stats})
                    pbar.update(1)
                except StopIteration:
                    break

        epoch_stats = {}
        for substep in running_stats[0]:
            epoch_stats[substep] = {}
            for key in running_stats[0][substep]:
                temp = [e[substep][key] for e in running_stats]
                if type(temp[0]) is dict:
                    epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                elif type(temp[0]) is np.ndarray:
                    # Mean value per vector element
                    epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                else:
                    epoch_stats[substep][key] = np.mean(temp)

        stats[phase].append(epoch_stats)
        stat_string = get_stats(phase, epoch_stats)

        outer.write(f"    {stat_string}")

        metrics[phase]['loss1'] = epoch_stats['global']['loss']
        if 'num_non_zero_triplets' in epoch_stats['global']:
            metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']
        if 'positive_ranking' in epoch_stats['global']:
            metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']
        if 'recall' in epoch_stats['global']:
            # Save training recall@1 when phase is "train"
            metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]
        if 'ap' in epoch_stats['global']:
            metrics[phase]['AP'] = epoch_stats['global']['ap']


        if scheduler is not None:
            scheduler.step()

        # Validation phase
        if (epoch > params.save_from and epoch % params.save_freq == 0) or epoch in params.save_milestones:
            phase = 'val'
            model.eval()

            final_model_path = "{}_epoch_{}".format(model_pathname, epoch) + '.pth'
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'params': params,
                "epoch": epoch
            }
            model_n = os.path.basename(final_model_path).split('.')[0]
            result_dir = os.path.dirname(final_model_path)

            for val_idx, val_set in enumerate(val_sets):
                test_database_name = val_set['database_name']
                test_query_name = val_set['query_name']

                recall_metrics = evaluate_4drad_dataset(model, device, val_set['dataset'], params)
                outer.write(f"    Valset {test_query_name}-{test_query_name}: Recall@1: {recall_metrics[1]:.4f}, Recall@5: {recall_metrics[5]:.4f}, Recall@10: {recall_metrics[10]:.4f}")
                metrics[phase][f'{test_database_name}--{test_query_name}'] = {
                    'r@1': recall_metrics[1],
                    'r@5': recall_metrics[5],
                    'r@10': recall_metrics[10]
                }
                if val_idx == 0:
                    current_val_recall = recall_metrics[1]
                    if current_val_recall > train_metrics['best_r@1']:
                        train_metrics['best_r@1'] = current_val_recall
                        train_metrics['best_epoch'] = epoch
                        checkpoint = {
                            "net": model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_schedule': scheduler.state_dict(),
                            'params': params,
                            "epoch": epoch
                        }
                        torch.save(checkpoint, os.path.join(weights_folder, 'model_best.pth'))

                save_recall_results(model_n, f"{test_database_name}_{test_query_name}", recall_metrics, result_dir)


            torch.save(checkpoint, final_model_path)
            outer.write(f"    @@@@ Saved model with val_recall@1 {current_val_recall:.4f} @@@")

        if params.batch_expansion_th is not None:
            # Dynamic batch size expansion based on number of non-zero triplets
            # Ratio of non-zero triplets
            le_train_stats = stats['train'][-1]  # Last epoch training stats
            rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
            if rnz < params.batch_expansion_th:
                dataloaders['train'].batch_sampler.expand_batch()
        train_metrics['metrics'].append(metrics)
        with open(os.path.join(weights_folder, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f, indent=4)
        
        writer.add_scalar("train/loss1", metrics['train']['loss1'], epoch)
        # Log validation recall only when testing is performed.
        if len(metrics['val']) > 0:
            for key, value in metrics['val'].items():
                writer.add_scalar(f"val/{key}_r@1", value['r@1'], epoch)
                writer.add_scalar(f"val/{key}_r@5", value['r@5'], epoch)
                writer.add_scalar(f"val/{key}_r@10", value['r@10'], epoch)

    writer.close()  # Close the TensorBoard writer when training completes


def create_weights_folder(model_name, ssataset_name):
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_folder = os.path.join(temp, 'weights', f"{model_name}_{dataset_name}")
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    assert os.path.exists(weights_folder), 'Cannot create weights folder: {}'.format(weights_folder)
    return weights_folder




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TransLoc4D model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='resume')  # add resume
    parser.set_defaults(debug=False)
    parser.add_argument("--gpu_id", type=int, default=2, help="GPU ID to use")

    args = parser.parse_args()
    print('==> Training config path: {}'.format(args.config))
    print('==> Model config path: {}'.format(args.model_config))
    print('==> Debug mode: {}'.format(args.debug))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"==> GPU ID: {os.environ['CUDA_VISIBLE_DEVICES']}")

    resume_filename = None
    if args.resume is not None:
        resume_filename = args.resume
        print("==> Resuming From {}".format(resume_filename))
        saved_state_dict = torch.load(resume_filename)
        params = saved_state_dict['params']
    else:
        params = TrainingParams(args.config, args.model_config, debug=args.debug)
        saved_state_dict = None

    # params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    
    current_time = time.strftime("%Y%m%d_%H%M")
    model_name = f"{current_time}_{params.model_name}"
    dataset_name = args.config.split('/')[-1].split('.')[0]
    weights_folder = create_weights_folder(model_name, dataset_name)
    copyfile(args.config, os.path.join(weights_folder, 'train_config.txt'))
    copyfile(args.model_config, os.path.join(weights_folder, 'model_config.txt'))


    do_train(params, model_name, weights_folder, resume_filename=resume_filename, saved_state_dict = saved_state_dict)
