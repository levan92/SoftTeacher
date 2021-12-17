# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
"""Generate labeled and unlabeled dataset for coco train.

Example:
python tools/coco_semi.py
"""

import argparse
import numpy as np
import json
import os
from pathlib import Path


def prepare_coco_data(
    seed=1,
    percent=10.0,
    version=2017,
    seed_offset=0,
    skip_coco_unlabelled=False,
    train_json=None,
):
    """Prepare COCO dataset for Semi-supervised learning
    Args:
      seed: random seed for dataset split
      percent: percentage of labeled dataset
      version: COCO dataset version
    """

    def _save_anno(name, images, annotations):
        """Save annotation."""
        print(
            ">> Processing dataset {}.json saved ({} images {} annotations)".format(
                name, len(images), len(annotations)
            )
        )
        new_anno = {}
        new_anno["images"] = images
        new_anno["annotations"] = annotations
        new_anno["licenses"] = anno["licenses"]
        new_anno["categories"] = anno["categories"]
        new_anno["info"] = anno["info"]
        path = Path(COCOANNODIR) / "semi_supervised"
        if not os.path.exists(path):
            os.mkdir(path)

        with (path / f"{save_name}.json").open("w") as f:
            json.dump(new_anno, f)
        print(
            ">> Data {}.json saved ({} images {} annotations)".format(
                name, len(images), len(annotations)
            )
        )

    np.random.seed(seed + seed_offset)

    if train_json:
        json_path = train_json
        COCOANNODIR = Path(json_path).parent
    else:
        COCOANNODIR = Path(DATA_DIR) / "annotations"
        json_path = Path(COCOANNODIR) / f"instances_train{version}.json"

    anno = json.load(open(json_path))

    image_list = anno["images"]
    labeled_tot = int(percent / 100.0 * len(image_list))
    labeled_ind = np.random.choice(
        range(len(image_list)), size=labeled_tot, replace=False
    )
    labeled_id = []
    labeled_images = []
    unlabeled_images = []
    labeled_ind = set(labeled_ind)
    for i in range(len(image_list)):
        if i in labeled_ind:
            labeled_images.append(image_list[i])
            labeled_id.append(image_list[i]["id"])
        else:
            unlabeled_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_id = set(labeled_id)
    labeled_annotations = []
    unlabeled_annotations = []
    for an in anno["annotations"]:
        if an["image_id"] in labeled_id:
            labeled_annotations.append(an)
        else:
            unlabeled_annotations.append(an)

    # save labeled and unlabeled
    save_name = "instances_train{version}.{seed}@{tot}".format(
        version=version, seed=seed, tot=int(percent)
    )
    _save_anno(save_name, labeled_images, labeled_annotations)
    save_name = "instances_train{version}.{seed}@{tot}-unlabeled".format(
        version=version, seed=seed, tot=int(percent)
    )
    _save_anno(save_name, unlabeled_images, unlabeled_annotations)

    if not skip_coco_unlabelled:
        # construct 120k unlabeled data
        unlabeled_ann_file = Path(COCOANNODIR) / f"instances_unlabeled{version}.json"
        if not unlabeled_ann_file.is_file():
            unlabeled_info = json.load(
                open(Path(COCOANNODIR) / f"image_info_unlabeled{version}.json")
            )
            unlabeled_info["annotations"] = []
            unlabeled_info["categories"] = anno["categories"]
            print(
                ">> Data {}.json saved({} images {} annotations)".format(
                    unlabeled_ann_file,
                    len(unlabeled_info["images"]),
                    len(unlabeled_info["annotations"]),
                )
            )
            json.dump(unlabeled_info, unlabeled_ann_file.open("w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--percent", type=float, default=10)
    parser.add_argument("--version", type=int, default=2017)
    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-json", type=str)
    parser.add_argument("--skip-coco-unlabelled", action="store_true")
    args = parser.parse_args()
    print(args)
    DATA_DIR = args.data_dir
    prepare_coco_data(
        args.seed,
        args.percent,
        args.version,
        args.seed_offset,
        train_json=args.train_json,
        skip_coco_unlabelled=args.skip_coco_unlabelled,
    )
