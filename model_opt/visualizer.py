
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

from PIL import Image, ImageDraw
from model_opt import utils

logger = logging.getLogger(__name__)

def draw_schedule(schedule, img_path="/tmp/output.png", verbose=False):
    if verbose:
        logger.basicConfig(level=logging.DEBUG)

    max_ts = 0
    max_address = 0
    boxes = {}
    for tensor, s in schedule.items():
        logger.debug(f"Tensor {tensor.name}, schedule = {s}")
        # No support for rematerialization yet
        assert len(s[0]) == 1
        # No support for spilling yet
        assert len(s[2]) == 0
        # Skip items for which we don't generate addresses
        if s[0][0].endswith("[weight]") or s[0][0].endswith("[ctrl]"):
            continue

        items = s[0][0].split("@")
        assert len(items) == 2
        allocate = int(items[0])
        address = int(items[1])
        deallocate = utils.parse_schedule_item(s[1][-1]) if len(s[1]) > 0 else allocate
        boxes[tensor] = (allocate, deallocate, address)
        logger.debug(f"    {boxes[tensor]}")
        max_ts = max(max_ts, deallocate)
        max_address = max(max_address, address + tensor.size)

    logger.info("Create img")
    ts_factor = math.ceil(max_address / max_ts)
    img = Image.new(mode="RGB", size=(1280, 1280), color=(200, 200, 200))
    ts_factor = 1280 / (max_ts + 1)
    address_factor = 1280 / max_address

    logger.info("Create draw")
    draw = ImageDraw.Draw(img)

    for tensor, box in boxes.items():
        logger.debug(f"drawing {tensor.name}")
        coordinates = (
            box[0] * ts_factor,
            box[2] * address_factor,
            (box[1] + 1) * ts_factor,
            (box[2] + tensor.size) * address_factor,
        )
        draw.rectangle(coordinates, fill=(50, 50, 200), outline=(255, 255, 255))
        draw.text(coordinates[:2], tensor.name, fill=(0, 0, 0))
    logger.info("Save img")
    img.save(img_path, "PNG")
