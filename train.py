import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tqdm
import argparse
from contextlib import nullcontext
from yacs.config import CfgNode as CN

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

import models
import metrics
from loss import ReconstructLoss, PerceptualLoss, StyleLoss, TVLoss
from utils.mask import DatasetWithMask
from utils.logger import StatusTracker, get_logger
from utils.data import get_dataset, get_data_generator
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint, image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-f', '--finetune', action='store_true', default=False,
        help='Finetune the model with frozen BN in encoder layers',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def train(args, cfg):
    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            cfg_dump=cfg.dump(sort_keys=False),
            exist_ok=cfg.train.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=cfg.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert cfg.train.batch_size % accelerator.num_processes == 0
    batch_size_per_process = cfg.train.batch_size // accelerator.num_processes
    micro_batch = cfg.dataloader.micro_batch or batch_size_per_process
    train_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='train',
    )
    valid_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='valid',
        subset_ids=torch.arange(5000),
    )
    train_set = DatasetWithMask(
        dataset=train_set,
        is_train=True,
        **cfg.mask,
    )
    valid_set = DatasetWithMask(
        dataset=valid_set,
        is_train=False,
        **cfg.mask,
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size_per_process,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=False,
        drop_last=False,
        batch_size=micro_batch,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Size of validation set: {len(valid_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {cfg.train.batch_size}')

    # BUILD MODELS AND OPTIMIZERS
    model = models.Generator(n_layers=cfg.model.n_layers, freeze_enc_bn=args.finetune)
    vgg16 = models.VGG16FeatureExtractor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.optim.lr, betas=cfg.train.optim.betas)
    step, best_psnr = 0, 0.

    def load_pretrained(model_path: str):
        ckpt_model = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {model_path}')

    def load_ckpt(ckpt_path: str):
        nonlocal step, best_psnr
        # load models
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {ckpt_path}')
        # load optimizers
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1
        best_psnr = ckpt_meta['best_psnr']

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save models
        accelerator.save(dict(model=accelerator.unwrap_model(model).state_dict()), os.path.join(save_path, 'model.pt'))
        # save optimizers
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step, best_psnr=best_psnr), os.path.join(save_path, 'meta.pt'))

    # LOAD PRETRAINED WEIGHTS
    if getattr(cfg.train, 'pretrained', None) is not None:
        load_pretrained(cfg.train.pretrained)
    elif args.finetune:
        logger.warning(f'The fine-tuning stage expects to load the pretrained model weights from '
                       f'the first training stage, but cfg.train.pretrained is None. '
                       f'Ignore this warning if you are doing this on purpose.')

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')
        logger.info(f'Best psnr so far: {best_psnr}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer = accelerator.prepare(model, optimizer)  # type: ignore
    train_loader, valid_loader = accelerator.prepare(train_loader, valid_loader)  # type: ignore

    # DEFINE LOSSES
    reconstruct = ReconstructLoss()
    perceptual = PerceptualLoss(vgg16)
    style = StyleLoss(vgg16)
    tv = TVLoss()

    # EVALUATION METRICS
    metric_l1 = metrics.L1(reduction='none').to(device)
    metric_psnr = metrics.PSNR(reduction='none', data_range=1.).to(device)
    metric_ssim = metrics.SSIM(size_average=False, data_range=1.).to(device)

    accelerator.wait_for_everyone()

    def run_step(_batch):
        optimizer.zero_grad()
        batch_gt_img, batch_mask = _batch
        batch_size = batch_gt_img.shape[0]
        loss_meter = metrics.KeyValueAverageMeter(
            keys=['loss_hole', 'loss_valid', 'loss_perceptual', 'loss_style', 'loss_tv'],
        )
        for i in range(0, batch_size, micro_batch):
            gt_img = batch_gt_img[i:i+micro_batch].float()
            mask = batch_mask[i:i+micro_batch].float().expand(gt_img.shape)
            cor_img = gt_img * mask
            loss_scale = gt_img.shape[0] / batch_size
            no_sync = (i + micro_batch) < batch_size
            cm = accelerator.no_sync(model) if no_sync else nullcontext()
            with cm:
                inpainted_img = model(cor_img, mask)
                loss_hole, loss_valid = reconstruct(inpainted_img, gt_img, mask)
                loss_perceptual = perceptual(inpainted_img, gt_img, mask)
                loss_style = style(inpainted_img, gt_img, mask)
                loss_tv = tv(inpainted_img)
                loss = (cfg.train.coef_hole * loss_hole +
                        cfg.train.coef_valid * loss_valid +
                        cfg.train.coef_perceptual * loss_perceptual +
                        cfg.train.coef_style * loss_style +
                        cfg.train.coef_tv * loss_tv)
                accelerator.backward(loss * loss_scale)
            loss_meter.update(dict(
                loss_hole=loss_hole.detach(),
                loss_valid=loss_valid.detach(),
                loss_perceptual=loss_perceptual.detach(),
                loss_style=loss_style.detach(),
                loss_tv=loss_tv.detach(),
            ), gt_img.shape[0])
        optimizer.step()
        return dict(**loss_meter.avg, lr=optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def evaluate(dataloader):
        metric_meter = metrics.KeyValueAverageMeter(keys=['l1', 'psnr', 'ssim'])
        for gt_img, mask in tqdm.tqdm(
                dataloader, desc='Evaluating', leave=False,
                disable=not accelerator.is_main_process,
        ):
            gt_img = gt_img.float()
            mask = mask.float().expand(gt_img.shape)
            cor_img = gt_img * mask
            inpainted_img = model(cor_img, mask).clamp(-1, 1)
            composited_img = (1 - mask) * inpainted_img + mask * gt_img

            gt_img = image_norm_to_float(gt_img)
            composited_img = image_norm_to_float(composited_img)
            l1 = metric_l1(composited_img, gt_img)
            psnr = metric_psnr(composited_img, gt_img)
            ssim = metric_ssim(composited_img, gt_img)
            l1, psnr, ssim = accelerator.gather_for_metrics((l1, psnr, ssim))
            metric_meter.update(dict(
                l1=l1.mean(),
                psnr=psnr.mean(),
                ssim=ssim.mean(),
            ), l1.shape[0])
        return metric_meter.avg

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        unwrapped_model = accelerator.unwrap_model(model)
        gt_img = torch.stack([valid_set[i][0] for i in range(12)], dim=0).float().to(device)
        mask = torch.stack([valid_set[i][1] for i in range(12)], dim=0).float().expand(gt_img.shape).to(device)
        cor_img = gt_img * mask

        inpainted_img = unwrapped_model(cor_img, mask).clamp(-1, 1)
        composited_img = (1 - mask) * inpainted_img + mask * gt_img
        show = []
        for i in tqdm.tqdm(range(12), desc='Sampling', leave=False,
                           disable=not accelerator.is_main_process):
            show.extend([
                image_norm_to_float(gt_img[i]).cpu(),
                image_norm_to_float(cor_img[i]).cpu(),
                image_norm_to_float(composited_img[i]).cpu(),
            ])
        save_image(show, savepath, nrow=6)

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
        with_tqdm=True,
    )
    while step < cfg.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        model.eval()
        # evaluate
        if check_freq(cfg.train.eval_freq, step):
            eval_status = evaluate(valid_loader)
            status_tracker.track_status('Eval', eval_status, step)
            if eval_status['psnr'] > best_psnr:
                best_psnr = eval_status['psnr']
                save_ckpt(os.path.join(exp_dir, 'ckpt', 'best'))
            accelerator.wait_for_everyone()
        # save checkpoint
        if check_freq(cfg.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(cfg.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(cfg.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info(f'Best valid psnr: {best_psnr}')
    logger.info('End of training')


def main():
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    train(args, cfg)


if __name__ == '__main__':
    main()
