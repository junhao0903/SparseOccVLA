# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import fcntl
import gzip
import hashlib
import mmcv
import os
import pickle
import struct
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import sys

sys.path.append(os.getcwd())
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp


def _to_pickleable(value):
    """Move tensors to CPU arrays before serializing inference results."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_pickleable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_pickleable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_pickleable(item) for item in value)
    return value


class IncrementalVLAResultWriter:
    """Append independently compressed, checksummed pickle records."""

    MAGIC = b'SPARSEVLA-PKL-1\n'
    HEADER = struct.Struct('>Q32s')

    def __init__(self, path, cfg, checkpoint_path):
        self.path = osp.abspath(path)
        mmcv.mkdir_or_exist(osp.dirname(self.path))
        self.metadata = {
            'config': osp.abspath(cfg.filename),
            'checkpoint': osp.abspath(checkpoint_path),
            'model_use_gen_token': bool(cfg.model.get('use_gen_token', False)),
        }
        self.lock_path = self.path + '.lock'
        self._ensure_header_and_metadata()

    def _lock(self):
        lock = open(self.lock_path, 'a+b')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return lock

    def _ensure_header_and_metadata(self):
        lock = self._lock()
        try:
            with open(self.path, 'a+b') as file:
                file.seek(0, os.SEEK_END)
                if file.tell() == 0:
                    file.write(self.MAGIC)
                    file.flush()
                    os.fsync(file.fileno())
                else:
                    file.seek(0)
                    if file.read(len(self.MAGIC)) != self.MAGIC:
                        raise ValueError('{} is not a SparseOccVLA incremental pkl'.format(self.path))
            # Recover a partial final record left by a forced process termination.
            self._repair_tail_locked()
            if not self._has_metadata():
                self._append_locked({'_record_type': 'metadata', 'metadata': self.metadata})
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _has_metadata(self):
        with open(self.path, 'rb') as file:
            file.seek(len(self.MAGIC))
            header = file.read(self.HEADER.size)
            if len(header) != self.HEADER.size:
                return False
            size, digest = self.HEADER.unpack(header)
            payload = file.read(size)
            if len(payload) != size or hashlib.sha256(payload).digest() != digest:
                return False
            record = pickle.loads(gzip.decompress(payload))
            return record.get('_record_type') == 'metadata'

    def _repair_tail_locked(self):
        valid_end = len(self.MAGIC)
        with open(self.path, 'rb') as file:
            if file.read(len(self.MAGIC)) != self.MAGIC:
                raise ValueError('Invalid SparseOccVLA result file header')
            while True:
                header = file.read(self.HEADER.size)
                if not header:
                    valid_end = file.tell()
                    break
                if len(header) != self.HEADER.size:
                    break
                size, digest = self.HEADER.unpack(header)
                payload = file.read(size)
                if len(payload) != size or hashlib.sha256(payload).digest() != digest:
                    break
                try:
                    pickle.loads(gzip.decompress(payload))
                except (OSError, EOFError, pickle.PickleError):
                    break
                valid_end = file.tell()
        with open(self.path, 'r+b') as file:
            file.seek(0, os.SEEK_END)
            if file.tell() != valid_end:
                file.truncate(valid_end)
                file.flush()
                os.fsync(file.fileno())

    def _append_locked(self, record):
        payload = gzip.compress(
            pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6)
        entry = self.HEADER.pack(len(payload), hashlib.sha256(payload).digest()) + payload
        with open(self.path, 'ab') as file:
            start = file.tell()
            try:
                file.write(entry)
                file.flush()
                os.fsync(file.fileno())
            except BaseException:
                file.truncate(start)
                file.flush()
                os.fsync(file.fileno())
                raise

    def append(self, output):
        result = _to_pickleable(output)
        if not isinstance(result, dict):
            result = {'model_output': result}
        result.setdefault('token', 'sample_{:06d}'.format(os.path.getsize(self.path)))
        result['language_prediction'] = result.get('generated_text')
        trajectory = result.get('traj_forecasting')
        if trajectory is not None:
            planning_reason = None
        elif self.metadata['model_use_gen_token']:
            planning_reason = 'The model did not return a trajectory for this sample.'
        else:
            planning_reason = 'This config does not emit trajectories (model.use_gen_token=False).'
        result['planning_result'] = {
            'available': trajectory is not None,
            'trajectory': trajectory,
            'reason': planning_reason,
        }
        lock = self._lock()
        try:
            self._append_locked({'_record_type': 'result', 'result': result})
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval_occ',action='store_true',help='eval_occ')
    parser.add_argument('--debug', action='store_true', help='Enable dataset debug mode')
    parser.add_argument('--save_path',help = 'evaluation path',default='./output/')
    parser.add_argument(
        '--out',
        help='incremental compressed VLA result file (.pkl; read with tools/visual.py)')
    parser.add_argument(
        '--use-template',
        action='store_true',
        help='reuse conv/val/template.json and desc/val/template.json for every sample')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        default=True,
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.use_template:
        cfg.model.language_template_token = 'template'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.save_path = args.save_path
    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    if distributed and args.out:
        raise ValueError('Incremental compressed --out currently supports single-GPU test only.')

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    if args.eval_occ:
        cfg.data.test['eval_occ'] = args.eval_occ
    if args.debug:
        cfg.data.test['debug'] = args.debug
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.eval_occ:
        model.eval_occ = True
        
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu',revise_keys=[('default.',''),('base_layer.','')])
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    result_writer = IncrementalVLAResultWriter(args.out, cfg, args.checkpoint) if args.out else None
    try:
        if not distributed:
            # assert False
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(
                model, data_loader, args.show, args.show_dir,
                result_callback=result_writer.append if result_writer else None)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = custom_multi_gpu_test(
                model, data_loader, args.tmpdir, args.gpu_collect)
    except KeyboardInterrupt:
        if result_writer:
            print('\nInterrupted. All completed sample records remain saved in {}.'.format(
                osp.abspath(args.out)))
        else:
            print('\nInterrupted before inference completed.')
        return

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print('Incremental compressed results saved to {}'.format(osp.abspath(args.out)))
        try:
            if not distributed:
                dataset.evaluate_occ(outputs)
            kwargs = {} if args.eval_options is None else args.eval_options
            kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
                '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
            if args.format_only and False:
                dataset.format_results(outputs, **kwargs)

            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))

                print(dataset.evaluate(outputs, **eval_kwargs))
        except KeyboardInterrupt:
            if args.out:
                print('\nInterrupted during post-inference evaluation; saved result records are intact.')
            else:
                print('\nInterrupted during post-inference evaluation.')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Previously committed result records remain intact.')
