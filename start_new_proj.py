#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import shutil

from common.utils.file_utils import copy_files, replace_file

CUSTOM_TRAINER_LINE_IDX = list(range(20))
EVAL_LINE_IDX = list(range(10, 20))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', type=str, required=True, help='Name of the new project')
    parser.add_argument('--proj_loc', type=str, required=True, help='Location of the new project')
    args, unknowns = parser.parse_known_args()

    proj_dir = osp.join(args.proj_loc, args.proj_name)
    if osp.isdir(proj_dir):
        print(
            'Project already exists in {}... '
            'Please remove it first if you want to start a new proj there...'.format(proj_dir)
        )
        exit()

    # copy files
    src_dir = osp.abspath(osp.join(__file__, '..'))
    os.makedirs(proj_dir, exist_ok=False)

    copy_files(src_dir, proj_dir, subdir_name='common')
    copy_files(src_dir, proj_dir, subdir_name='configs')
    copy_files(src_dir, proj_dir, subdir_name='custom')
    copy_files(src_dir, proj_dir, subdir_name='scripts')
    copy_files(src_dir, proj_dir, subdir_name='tests')
    copy_files(src_dir, proj_dir, subdir_name='docs', file_names='.gitignore')
    copy_files(src_dir, proj_dir, subdir_name='experiments', file_names='.gitignore')
    copy_files(src_dir, proj_dir, subdir_name='results', file_names='.gitignore')

    files = [
        'LICENSE', 'requirements.txt', 'setup.cfg', 'train.py', 'evaluate.py', '.pre-commit-config.yaml', '.gitignore'
    ]
    copy_files(src_dir, proj_dir, file_names=files)
    with open(osp.join(proj_dir, 'README.md'), 'w') as f:
        f.write('# {}\n'.format(args.proj_name))

    # copy readme
    shutil.copyfile(osp.join(src_dir, 'README.md'), osp.join(proj_dir, 'docs', 'common_trainer.md'))

    # remove tests_common
    shutil.rmtree(osp.join(proj_dir, 'tests', 'tests_common'))

    # renames
    replace_file(osp.join(proj_dir, 'LICENSE'), 'common_trainer', args.proj_name, line_idx=2)

    # custom/trainer/custom_trainer.py
    proj_name_lower = args.proj_name.lower()
    custom_dir = osp.join(proj_dir, proj_name_lower)
    os.rename(osp.join(proj_dir, 'custom'), custom_dir)
    custom_trainer = osp.join(custom_dir, 'trainer', 'custom_trainer.py')
    trainer_class_name = args.proj_name[0].upper() + args.proj_name[1:]
    replace_file(
        custom_trainer, ['CustomTrainer', 'custom.'], [trainer_class_name + 'Trainer', proj_name_lower + '.'],
        line_idx=CUSTOM_TRAINER_LINE_IDX
    )
    os.rename(custom_trainer, custom_trainer.replace('custom', proj_name_lower))

    # __init__
    dirs = ['datasets', 'loss', 'metric', 'models']
    for dir in dirs:
        init_file = osp.join(custom_dir, dir, '__init__.py')
        replace_file(init_file, 'custom.', proj_name_lower + '.')

    # train.py
    replace_file(
        osp.join(proj_dir, 'train.py'), ['CustomTrainer', 'custom.', 'custom_'],
        [trainer_class_name + 'Trainer', proj_name_lower + '.', proj_name_lower + '_']
    )

    # evaluate.py
    replace_file(osp.join(proj_dir, 'evaluate.py'), ['custom.'], [proj_name_lower + '.'], line_idx=EVAL_LINE_IDX)

    print('New project starts at {}'.format(proj_dir))
