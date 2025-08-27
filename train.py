import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


def train(hyperparameters, training_options, device, tensorboard_writer=None):
    """
    Main training function for YOLOv7 model.
    
    Args:
        hyperparameters (dict): Training hyperparameters
        training_options (Namespace): Training configuration options
        device (torch.device): Training device (GPU/CPU)
        tensorboard_writer (SummaryWriter): Tensorboard logging writer
    """
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{key}={value}' for key, value in hyperparameters.items()))
    
    # Extract key training parameters
    save_directory = Path(training_options.save_dir)
    num_epochs = training_options.epochs
    batch_size = training_options.batch_size
    total_batch_size = training_options.total_batch_size
    pretrained_weights = training_options.weights
    process_rank = training_options.global_rank
    freeze_layers = training_options.freeze

    # Setup directories
    weights_directory = save_directory / 'weights'
    weights_directory.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = weights_directory / 'last.pt'
    best_checkpoint_path = weights_directory / 'best.pt'
    results_file_path = save_directory / 'results.txt'

    # Save training configuration
    with open(save_directory / 'hyp.yaml', 'w') as file:
        yaml.dump(hyperparameters, file, sort_keys=False)
    with open(save_directory / 'opt.yaml', 'w') as file:
        yaml.dump(vars(training_options), file, sort_keys=False)

    # Configure training environment
    should_create_plots = not training_options.evolve
    use_cuda = device.type != 'cpu'
    init_seeds(2 + process_rank)
    
    # Load dataset configuration
    with open(training_options.data) as file:
        dataset_config = yaml.load(file, Loader=yaml.SafeLoader)
    is_coco_dataset = training_options.data.endswith('coco.yaml')

    # Initialize logging
    loggers = {'wandb': None}
    if process_rank in [-1, 0]:
        training_options.hyp = hyperparameters
        wandb_run_id = torch.load(pretrained_weights, map_location=device, weights_only=False).get('wandb_id') if pretrained_weights.endswith('.pt') and os.path.isfile(pretrained_weights) else None
        wandb_logger = WandbLogger(training_options, Path(training_options.save_dir).stem, wandb_run_id, dataset_config)
        loggers['wandb'] = wandb_logger.wandb
        dataset_config = wandb_logger.data_dict
        if wandb_logger.wandb:
            pretrained_weights, num_epochs, hyperparameters = training_options.weights, training_options.epochs, training_options.hyp

    # Setup model parameters
    num_classes = 1 if training_options.single_cls else int(dataset_config['nc'])
    class_names = ['item'] if training_options.single_cls and len(dataset_config['names']) != 1 else dataset_config['names']
    assert len(class_names) == num_classes, f'{len(class_names)} names found for nc={num_classes} dataset in {training_options.data}'

    # Initialize model
    is_pretrained = pretrained_weights.endswith('.pt')
    if is_pretrained:
        with torch_distributed_zero_first(process_rank):
            attempt_download(pretrained_weights)
        checkpoint = torch.load(pretrained_weights, map_location=device, weights_only=False)
        model = Model(training_options.cfg or checkpoint['model'].yaml, ch=3, nc=num_classes, anchors=hyperparameters.get('anchors')).to(device)
        
        # Load pretrained weights
        exclude_keys = ['anchor'] if (training_options.cfg or hyperparameters.get('anchors')) and not training_options.resume else []
        pretrained_state_dict = checkpoint['model'].float().state_dict()
        pretrained_state_dict = intersect_dicts(pretrained_state_dict, model.state_dict(), exclude=exclude_keys)
        model.load_state_dict(pretrained_state_dict, strict=False)
        logger.info(f'Transferred {len(pretrained_state_dict)}/{len(model.state_dict())} items from {pretrained_weights}')
    else:
        model = Model(training_options.cfg, ch=3, nc=num_classes, anchors=hyperparameters.get('anchors')).to(device)
    
    # Check dataset
    with torch_distributed_zero_first(process_rank):
        check_dataset(dataset_config)
    train_dataset_path = dataset_config['train']
    validation_dataset_path = dataset_config['val']

    # Setup layer freezing
    freeze_layer_names = [f'model.{layer}.' for layer in (freeze_layers if len(freeze_layers) > 1 else range(freeze_layers[0]))]
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = True  # Enable gradients for all layers by default
        if any(freeze_name in parameter_name for freeze_name in freeze_layer_names):
            print(f'freezing {parameter_name}')
            parameter.requires_grad = False

    # Setup optimizer
    nominal_batch_size = 64
    gradient_accumulation_steps = max(round(nominal_batch_size / total_batch_size), 1)
    hyperparameters['weight_decay'] *= total_batch_size * gradient_accumulation_steps / nominal_batch_size
    logger.info(f"Scaled weight_decay = {hyperparameters['weight_decay']}")

    # Organize parameters into groups for different optimization strategies
    no_decay_params, weight_decay_params, bias_params = [], [], []
    
    for module_name, module in model.named_modules():
        # Collect bias parameters
        if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
            bias_params.append(module.bias)
        
        # BatchNorm weights (no decay)
        if isinstance(module, nn.BatchNorm2d):
            no_decay_params.append(module.weight)
        # Regular weights (with decay)
        elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            weight_decay_params.append(module.weight)
        
        # Handle various implicit parameters (no decay)
        implicit_param_names = ['im', 'imc', 'imb', 'imo', 'ia']
        for param_name in implicit_param_names:
            if hasattr(module, param_name):
                implicit_param = getattr(module, param_name)
                if hasattr(implicit_param, 'implicit'):
                    no_decay_params.append(implicit_param.implicit)
                else:
                    for implicit_item in implicit_param:
                        no_decay_params.append(implicit_item.implicit)
        
        # Handle attention parameters
        if hasattr(module, 'attn'):
            attention_param_names = ['logit_scale', 'q_bias', 'v_bias', 'relative_position_bias_table']
            for param_name in attention_param_names:
                if hasattr(module.attn, param_name):
                    no_decay_params.append(getattr(module.attn, param_name))
        
        # Handle RepVGG parameters
        if hasattr(module, 'rbr_dense'):
            repvgg_param_names = [
                'weight_rbr_origin', 'weight_rbr_avg_conv', 'weight_rbr_pfir_conv',
                'weight_rbr_1x1_kxk_idconv1', 'weight_rbr_1x1_kxk_conv2',
                'weight_rbr_gconv_dw', 'weight_rbr_gconv_pw', 'vector'
            ]
            for param_name in repvgg_param_names:
                if hasattr(module.rbr_dense, param_name):
                    no_decay_params.append(getattr(module.rbr_dense, param_name))

    # Initialize optimizer
    if training_options.adam:
        optimizer = optim.Adam(no_decay_params, lr=hyperparameters['lr0'], betas=(hyperparameters['momentum'], 0.999))
    else:
        optimizer = optim.SGD(no_decay_params, lr=hyperparameters['lr0'], momentum=hyperparameters['momentum'], nesterov=True)

    optimizer.add_param_group({'params': weight_decay_params, 'weight_decay': hyperparameters['weight_decay']})
    optimizer.add_param_group({'params': bias_params})
    logger.info(f'Optimizer groups: {len(bias_params)} .bias, {len(weight_decay_params)} conv.weight, {len(no_decay_params)} other')
    del no_decay_params, weight_decay_params, bias_params

    # Setup learning rate scheduler
    if training_options.linear_lr:
        lr_lambda = lambda epoch: (1 - epoch / (num_epochs - 1)) * (1.0 - hyperparameters['lrf']) + hyperparameters['lrf']
    else:
        lr_lambda = one_cycle(1, hyperparameters['lrf'], num_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Initialize Exponential Moving Average
    exponential_moving_average = ModelEMA(model) if process_rank in [-1, 0] else None

    # Handle checkpoint resuming
    start_epoch, best_fitness_score = 0, 0.0
    if is_pretrained:
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_fitness_score = checkpoint['best_fitness']

        if exponential_moving_average and checkpoint.get('ema'):
            exponential_moving_average.ema.load_state_dict(checkpoint['ema'].float().state_dict())
            exponential_moving_average.updates = checkpoint['updates']

        if checkpoint.get('training_results') is not None:
            results_file_path.write_text(checkpoint['training_results'])

        start_epoch = checkpoint['epoch'] + 1
        if training_options.resume:
            assert start_epoch > 0, f'{pretrained_weights} training to {num_epochs} epochs is finished, nothing to resume.'
        if num_epochs < start_epoch:
            logger.info(f'{pretrained_weights} has been trained for {checkpoint["epoch"]} epochs. Fine-tuning for {num_epochs} additional epochs.')
            num_epochs += checkpoint['epoch']

        del checkpoint, pretrained_state_dict

    # Setup image sizes and model parameters
    grid_size = max(int(model.stride.max()), 32)
    num_detection_layers = model.model[-1].nl
    train_image_size, test_image_size = [check_img_size(size, grid_size) for size in training_options.img_size]

    # Setup multi-GPU training
    if use_cuda and process_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Setup synchronized batch normalization for distributed training
    if training_options.sync_bn and use_cuda and process_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Create training dataloader
    train_dataloader, train_dataset = create_dataloader(
        train_dataset_path, train_image_size, batch_size, grid_size, training_options,
        hyp=hyperparameters, augment=True, cache=training_options.cache_images, 
        rect=training_options.rect, rank=process_rank, world_size=training_options.world_size, 
        workers=training_options.workers, image_weights=training_options.image_weights, 
        quad=training_options.quad, prefix=colorstr('train: ')
    )
    
    max_label_class = np.concatenate(train_dataset.labels, 0)[:, 0].max()
    num_batches = len(train_dataloader)
    assert max_label_class < num_classes, f'Label class {max_label_class} exceeds nc={num_classes} in {training_options.data}. Possible class labels are 0-{num_classes - 1}'

    # Setup validation for main process
    if process_rank in [-1, 0]:
        validation_dataloader = create_dataloader(
            validation_dataset_path, test_image_size, batch_size * 2, grid_size, training_options,
            hyp=hyperparameters, cache=training_options.cache_images and not training_options.notest, 
            rect=True, rank=-1, world_size=training_options.world_size, workers=training_options.workers,
            pad=0.5, prefix=colorstr('val: ')
        )[0]

        if not training_options.resume:
            all_labels = np.concatenate(train_dataset.labels, 0)
            label_classes = torch.tensor(all_labels[:, 0])
            
            if should_create_plots:
                if tensorboard_writer:
                    tensorboard_writer.add_histogram('classes', label_classes, 0)

            # Check and optimize anchors
            if not training_options.noautoanchor:
                check_anchors(train_dataset, model=model, thr=hyperparameters['anchor_t'], imgsz=train_image_size)
            model.half().float()  # Pre-reduce anchor precision

    # Setup Distributed Data Parallel
    if use_cuda and process_rank != -1:
        model = DDP(
            model, device_ids=[training_options.local_rank], 
            output_device=training_options.local_rank,
            find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        )

    # Scale hyperparameters based on model architecture
    hyperparameters['box'] *= 3. / num_detection_layers
    hyperparameters['cls'] *= num_classes / 80. * 3. / num_detection_layers
    hyperparameters['obj'] *= (train_image_size / 640) ** 2 * 3. / num_detection_layers
    hyperparameters['label_smoothing'] = training_options.label_smoothing
    
    # Attach parameters to model
    model.nc = num_classes
    model.hyp = hyperparameters
    model.gr = 1.0  # IoU loss ratio
    model.class_weights = labels_to_class_weights(train_dataset.labels, num_classes).to(device) * num_classes
    model.names = class_names

    # Initialize training variables
    training_start_time = time.time()
    num_warmup_iterations = max(round(hyperparameters['warmup_epochs'] * num_batches), 1000)
    mean_average_precisions = np.zeros(num_classes)
    validation_results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    
    # Initialize training components
    gradient_scaler = amp.GradScaler(enabled=use_cuda)
    compute_loss_ota = ComputeLossOTA(model)
    compute_loss_standard = ComputeLoss(model)
    
    logger.info(f'Image sizes {train_image_size} train, {test_image_size} test\n'
                f'Using {train_dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_directory}\n'
                f'Starting training for {num_epochs} epochs...')
    
    # Save initial model
    torch.save(model, weights_directory / 'init.pt')
    
    # Main training loop
    for current_epoch in range(start_epoch, num_epochs):
        model.train()

        # Update image weights for weighted sampling
        if training_options.image_weights:
            if process_rank in [-1, 0]:
                class_weights = model.class_weights.cpu().numpy() * (1 - mean_average_precisions) ** 2 / num_classes
                image_weights = labels_to_image_weights(train_dataset.labels, nc=num_classes, class_weights=class_weights)
                train_dataset.indices = random.choices(range(train_dataset.n), weights=image_weights, k=train_dataset.n)
            
            # Broadcast indices for distributed training
            if process_rank != -1:
                indices = (torch.tensor(train_dataset.indices) if process_rank == 0 else torch.zeros(train_dataset.n)).int()
                dist.broadcast(indices, 0)
                if process_rank != 0:
                    train_dataset.indices = indices.cpu().numpy()

        # Initialize loss tracking
        mean_losses = torch.zeros(4, device=device)
        if process_rank != -1:
            train_dataloader.sampler.set_epoch(current_epoch)
        
        progress_bar = enumerate(train_dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if process_rank in [-1, 0]:
            progress_bar = tqdm(progress_bar, total=num_batches)
        
        optimizer.zero_grad()
        
        # Batch training loop
        for batch_index, (images, targets, image_paths, _) in progress_bar:
            num_integrated_batches = batch_index + num_batches * current_epoch
            images = images.to(device, non_blocking=True).float() / 255.0

            # Warmup phase
            if num_integrated_batches <= num_warmup_iterations:
                warmup_range = [0, num_warmup_iterations]
                current_accumulation = max(1, np.interp(num_integrated_batches, warmup_range, [1, nominal_batch_size / total_batch_size]).round())
                
                for param_group_idx, param_group in enumerate(optimizer.param_groups):
                    # Different learning rate schedules for bias vs other parameters
                    if param_group_idx == 2:  # bias parameters
                        target_lr = hyperparameters['warmup_bias_lr']
                    else:
                        target_lr = 0.0
                    param_group['lr'] = np.interp(num_integrated_batches, warmup_range, [target_lr, param_group['initial_lr'] * lr_lambda(current_epoch)])
                    
                    if 'momentum' in param_group:
                        param_group['momentum'] = np.interp(num_integrated_batches, warmup_range, [hyperparameters['warmup_momentum'], hyperparameters['momentum']])

            # Multi-scale training
            if training_options.multi_scale:
                scale_size = random.randrange(train_image_size * 0.5, train_image_size * 1.5 + grid_size) // grid_size * grid_size
                scale_factor = scale_size / max(images.shape[2:])
                if scale_factor != 1:
                    new_shape = [math.ceil(x * scale_factor / grid_size) * grid_size for x in images.shape[2:]]
                    images = F.interpolate(images, size=new_shape, mode='bilinear', align_corners=False)

            # Forward pass
            with amp.autocast(enabled=use_cuda):
                predictions = model(images)
                
                # Choose loss computation method
                use_ota_loss = 'loss_ota' not in hyperparameters or hyperparameters['loss_ota'] == 1
                if use_ota_loss:
                    loss, loss_components = compute_loss_ota(predictions, targets.to(device), images)
                else:
                    loss, loss_components = compute_loss_standard(predictions, targets.to(device))
                
                # Scale loss for distributed training
                if process_rank != -1:
                    loss *= training_options.world_size
                if training_options.quad:
                    loss *= 4.

            # Backward pass
            gradient_scaler.scale(loss).backward()

            # Optimizer step
            if num_integrated_batches % current_accumulation == 0:
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                optimizer.zero_grad()
                if exponential_moving_average:
                    exponential_moving_average.update(model)

            # Progress logging
            if process_rank in [-1, 0]:
                mean_losses = (mean_losses * batch_index + loss_components) / (batch_index + 1)
                gpu_memory = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                progress_string = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{current_epoch}/{num_epochs - 1}', gpu_memory, *mean_losses, targets.shape[0], images.shape[-1]
                )
                progress_bar.set_description(progress_string)

                # Save training batch visualizations
                if should_create_plots and batch_index < 10:
                    batch_image_path = save_directory / f'train_batch{batch_index}.jpg'
                    Thread(target=plot_images, args=(images, targets, image_paths, batch_image_path), daemon=True).start()
                elif should_create_plots and batch_index == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(path), caption=path.name) 
                                                 for path in save_directory.glob('train*.jpg') if path.exists()]})

        # End of epoch processing
        current_learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]
        scheduler.step()

        # Validation and logging (main process only)
        if process_rank in [-1, 0]:
            exponential_moving_average.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            is_final_epoch = current_epoch + 1 == num_epochs
            
            # Run validation
            if not training_options.notest or is_final_epoch:
                wandb_logger.current_epoch = current_epoch + 1
                validation_results, mean_average_precisions, validation_times = test.test(
                    dataset_config, batch_size=batch_size * 2, imgsz=test_image_size,
                    model=exponential_moving_average.ema, single_cls=training_options.single_cls,
                    dataloader=validation_dataloader, save_dir=save_directory,
                    verbose=num_classes < 50 and is_final_epoch, plots=should_create_plots and is_final_epoch,
                    wandb_logger=wandb_logger, compute_loss=compute_loss_standard,
                    is_coco=is_coco_dataset, v5_metric=training_options.v5_metric
                )

            # Save results
            with open(results_file_path, 'a') as file:
                file.write(progress_string + '%10.4g' * 7 % validation_results + '\n')
            
            # Cloud storage backup
            if len(training_options.name) and training_options.bucket:
                os.system(f'gsutil cp {results_file_path} gs://{training_options.bucket}/results/results{training_options.name}.txt')

            # Tensorboard and W&B logging
            metric_tags = [
                'train/box_loss', 'train/obj_loss', 'train/cls_loss',
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',
                'x/lr0', 'x/lr1', 'x/lr2'
            ]
            
            metric_values = list(mean_losses[:-1]) + list(validation_results) + current_learning_rates
            for value, tag in zip(metric_values, metric_tags):
                if tensorboard_writer:
                    tensorboard_writer.add_scalar(tag, value, current_epoch)
                if wandb_logger.wandb:
                    wandb_logger.log({tag: value})

            # Update best fitness score
            current_fitness = fitness(np.array(validation_results).reshape(1, -1))
            if current_fitness > best_fitness_score:
                best_fitness_score = current_fitness
            wandb_logger.end_epoch(best_result=best_fitness_score == current_fitness)

            # Save model checkpoints
            if (not training_options.nosave) or (is_final_epoch and not training_options.evolve):
                checkpoint_data = {
                    'epoch': current_epoch,
                    'best_fitness': best_fitness_score,
                    'training_results': results_file_path.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(exponential_moving_average.ema).half(),
                    'updates': exponential_moving_average.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None
                }

                # Save different types of checkpoints
                torch.save(checkpoint_data, last_checkpoint_path)
                if best_fitness_score == current_fitness:
                    torch.save(checkpoint_data, best_checkpoint_path)
                if (best_fitness_score == current_fitness) and (current_epoch >= 200):
                    torch.save(checkpoint_data, weights_directory / f'best_{current_epoch:03d}.pt')
                if current_epoch == 0:
                    torch.save(checkpoint_data, weights_directory / f'epoch_{current_epoch:03d}.pt')
                elif ((current_epoch + 1) % 25) == 0:
                    torch.save(checkpoint_data, weights_directory / f'epoch_{current_epoch:03d}.pt')
                elif current_epoch >= (num_epochs - 5):
                    torch.save(checkpoint_data, weights_directory / f'epoch_{current_epoch:03d}.pt')
                
                # W&B model logging
                if wandb_logger.wandb:
                    if ((current_epoch + 1) % training_options.save_period == 0 and not is_final_epoch) and training_options.save_period != -1:
                        wandb_logger.log_model(last_checkpoint_path.parent, training_options, current_epoch, current_fitness, best_model=best_fitness_score == current_fitness)
                
                del checkpoint_data

    # Post-training cleanup and evaluation
    if process_rank in [-1, 0]:
        # Generate training plots
        if should_create_plots:
            plot_results(save_dir=save_directory)
            if wandb_logger.wandb:
                result_files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_directory / file), caption=file) 
                                            for file in result_files if (save_directory / file).exists()]})
        
        # Final model evaluation
        total_training_time = (time.time() - training_start_time) / 3600
        logger.info(f'{current_epoch - start_epoch + 1} epochs completed in {total_training_time:.3f} hours.\n')
        
        # COCO evaluation for final model
        if training_options.data.endswith('coco.yaml') and num_classes == 80:
            for checkpoint_path in (last_checkpoint_path, best_checkpoint_path) if best_checkpoint_path.exists() else (last_checkpoint_path,):
                validation_results, _, _ = test.test(
                    training_options.data, batch_size=batch_size * 2, imgsz=test_image_size,
                    conf_thres=0.001, iou_thres=0.7, model=attempt_load(checkpoint_path, device).half(),
                    single_cls=training_options.single_cls, dataloader=validation_dataloader,
                    save_dir=save_directory, save_json=True, plots=False,
                    is_coco=is_coco_dataset, v5_metric=training_options.v5_metric
                )

        # Optimize final models
        final_model_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
        for checkpoint_path in [last_checkpoint_path, best_checkpoint_path]:
            if checkpoint_path.exists():
                strip_optimizer(checkpoint_path)
        
        # Cloud storage upload
        if training_options.bucket:
            os.system(f'gsutil cp {final_model_path} gs://{training_options.bucket}/weights')
        
        # Log final model to W&B
        if wandb_logger.wandb and not training_options.evolve:
            wandb_logger.wandb.log_artifact(
                str(final_model_path), type='model',
                name='run_' + wandb_logger.wandb_run.id + '_model',
                aliases=['last', 'best', 'stripped']
            )
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
    return validation_results


def parse_training_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='YOLOv7 Training Script')
    
    # Model and data configuration
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    
    # Training modes and optimizations
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # Optimizer settings
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    
    # System and hardware settings
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    
    # Logging and output
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    
    return parser.parse_args()


def setup_distributed_training(training_options):
    """Configure distributed training settings."""
    training_options.world_size = int(os.environ.get('WORLD_SIZE', 1))
    training_options.global_rank = int(os.environ.get('RANK', -1))
    set_logging(training_options.global_rank)


def handle_training_resume(training_options):
    """Handle training resumption from checkpoints."""
    wandb_run = check_wandb_resume(training_options)
    
    if training_options.resume and not wandb_run:
        checkpoint_path = training_options.resume if isinstance(training_options.resume, str) else get_latest_run()
        assert os.path.isfile(checkpoint_path), 'ERROR: --resume checkpoint does not exist'
        
        # Store original settings
        original_global_rank, original_local_rank = training_options.global_rank, training_options.local_rank
        
        # Load checkpoint configuration
        with open(Path(checkpoint_path).parent.parent / 'opt.yaml') as file:
            training_options = argparse.Namespace(**yaml.load(file, Loader=yaml.SafeLoader))
        
        # Restore critical settings
        training_options.cfg, training_options.weights = '', checkpoint_path
        training_options.resume = True
        training_options.batch_size = training_options.total_batch_size
        training_options.global_rank, training_options.local_rank = original_global_rank, original_local_rank
        
        logger.info(f'Resuming training from {checkpoint_path}')
    else:
        # Validate configuration files
        training_options.data = check_file(training_options.data)
        training_options.cfg = check_file(training_options.cfg)
        training_options.hyp = check_file(training_options.hyp)
        
        assert len(training_options.cfg) or len(training_options.weights), 'either --cfg or --weights must be specified'
        
        # Extend image sizes if needed
        training_options.img_size.extend([training_options.img_size[-1]] * (2 - len(training_options.img_size)))
        
        # Set experiment name and directory
        training_options.name = 'evolve' if training_options.evolve else training_options.name
        training_options.save_dir = increment_path(
            Path(training_options.project) / training_options.name, 
            exist_ok=training_options.exist_ok | training_options.evolve
        )
    
    return training_options


def setup_device_and_batch_size(training_options):
    """Configure training device and adjust batch size for distributed training."""
    training_options.total_batch_size = training_options.batch_size
    device = select_device(training_options.device, batch_size=training_options.batch_size)
    
    if training_options.local_rank != -1:
        assert torch.cuda.device_count() > training_options.local_rank
        torch.cuda.set_device(training_options.local_rank)
        device = torch.device('cuda', training_options.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert training_options.batch_size % training_options.world_size == 0, '--batch-size must be multiple of CUDA device count'
        training_options.batch_size = training_options.total_batch_size // training_options.world_size
    
    return device


def run_hyperparameter_evolution(hyperparameters, training_options, device):
    """Run hyperparameter evolution to find optimal training parameters."""
    # Hyperparameter evolution metadata (mutation scale, lower_limit, upper_limit)
    evolution_metadata = {
        'lr0': (1, 1e-5, 1e-1),  # initial learning rate
        'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate
        'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
        'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs
        'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
        'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
        'box': (1, 0.02, 0.2),  # box loss gain
        'cls': (1, 0.2, 4.0),  # cls loss gain
        'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
        'obj': (1, 0.2, 4.0),  # obj loss gain
        'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
        'iou_t': (0, 0.1, 0.7),  # IoU training threshold
        'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
        'anchors': (2, 2.0, 10.0),  # anchors per output grid
        'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma
        'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation
        'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation
        'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation
        'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
        'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
        'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
        'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
        'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction)
        'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
        'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
        'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
        'mixup': (1, 0.0, 1.0),   # image mixup (probability)
        'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
        'paste_in': (1, 0.0, 1.0)    # segment copy-paste (probability)
    }
    
    # Load base hyperparameters
    with open(training_options.hyp, errors='ignore') as file:
        hyperparameters = yaml.safe_load(file)
        if 'anchors' not in hyperparameters:
            hyperparameters['anchors'] = 3
    
    assert training_options.local_rank == -1, 'DDP mode not implemented for --evolve'
    training_options.notest, training_options.nosave = True, True
    
    evolved_hyperparameters_file = Path(training_options.save_dir) / 'hyp_evolved.yaml'
    
    # Download existing evolution results if available
    if training_options.bucket:
        os.system(f'gsutil cp gs://{training_options.bucket}/evolve.txt .')
    
    # Evolution loop
    for generation in range(300):
        if Path('evolve.txt').exists():
            # Select parent hyperparameters
            parent_selection_method = 'single'
            evolution_results = np.loadtxt('evolve.txt', ndmin=2)
            num_parents = min(5, len(evolution_results))
            evolution_results = evolution_results[np.argsort(-fitness(evolution_results))][:num_parents]
            fitness_weights = fitness(evolution_results) - fitness(evolution_results).min()
            
            if parent_selection_method == 'single' or len(evolution_results) == 1:
                selected_parent = evolution_results[random.choices(range(num_parents), weights=fitness_weights)[0]]
            elif parent_selection_method == 'weighted':
                selected_parent = (evolution_results * fitness_weights.reshape(num_parents, 1)).sum(0) / fitness_weights.sum()

            # Mutate hyperparameters
            mutation_probability, mutation_sigma = 0.8, 0.2
            random_generator = np.random
            random_generator.seed(int(time.time()))
            
            gains = np.array([metadata[0] for metadata in evolution_metadata.values()])
            num_parameters = len(evolution_metadata)
            mutation_vector = np.ones(num_parameters)
            
            while all(mutation_vector == 1):  # Ensure at least one parameter changes
                mutation_vector = (gains * (random_generator.random(num_parameters) < mutation_probability) * 
                                 random_generator.randn(num_parameters) * random_generator.random() * mutation_sigma + 1).clip(0.3, 3.0)
            
            for param_idx, param_name in enumerate(hyperparameters.keys()):
                hyperparameters[param_name] = float(selected_parent[param_idx + 7] * mutation_vector[param_idx])

        # Apply hyperparameter constraints
        for param_name, (_, lower_limit, upper_limit) in evolution_metadata.items():
            hyperparameters[param_name] = max(hyperparameters[param_name], lower_limit)
            hyperparameters[param_name] = min(hyperparameters[param_name], upper_limit)
            hyperparameters[param_name] = round(hyperparameters[param_name], 5)

        # Train with mutated hyperparameters
        evolution_results = train(hyperparameters.copy(), training_options, device)

        # Record evolution results
        print_mutation(hyperparameters.copy(), evolution_results, evolved_hyperparameters_file, training_options.bucket)

    # Generate evolution plots
    plot_evolution(evolved_hyperparameters_file)
    print(f'Hyperparameter evolution complete. Best results saved as: {evolved_hyperparameters_file}\n'
          f'Command to train a new model with these hyperparameters: $ python train.py --hyp {evolved_hyperparameters_file}')


if __name__ == '__main__':
    # Parse command line arguments
    training_options = parse_training_arguments()
    
    # Setup distributed training
    setup_distributed_training(training_options)
    
    # Handle training resumption
    training_options = handle_training_resume(training_options)
    
    # Setup device and batch size
    device = setup_device_and_batch_size(training_options)

    # Load hyperparameters
    with open(training_options.hyp) as file:
        hyperparameters = yaml.load(file, Loader=yaml.SafeLoader)

    # Log training configuration
    logger.info(training_options)
    
    if not training_options.evolve:
        # Standard training
        tensorboard_writer = None
        if training_options.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {training_options.project}', view at http://localhost:6006/")
            tensorboard_writer = SummaryWriter(training_options.save_dir)
        
        train(hyperparameters, training_options, device, tensorboard_writer)
    else:
        # Hyperparameter evolution
        run_hyperparameter_evolution(hyperparameters, training_options, device)
