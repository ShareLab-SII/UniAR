"""GenEval evaluator — object detection + compositional check.

Requires:
- mmdet (Mask2Former checkpoint)
- open_clip (ViT-L-14 for color classification)

Set env vars:
    GENEVAL_CONFIG_PATH  path to mask2former config .py
    GENEVAL_CKPT_PATH    directory containing the .pth checkpoint
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from collections import defaultdict
from PIL import Image, ImageOps
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

DEVICE = "cuda"
MY_CONFIG_PATH = os.environ.get("GENEVAL_CONFIG_PATH", "")
MY_CKPT_PATH = os.environ.get("GENEVAL_CKPT_PATH", "")


def load_geneval():
    def timed(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            print(f"Function {fn.__name__!r} executed in {time.time() - t0:.3f}s", file=sys.stderr)
            return result
        return wrapper

    @timed
    def load_models():
        CONFIG_PATH = os.path.join(os.path.dirname(mmdet.__file__), MY_CONFIG_PATH)
        OBJECT_DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
        CKPT_PATH = os.path.join(MY_CKPT_PATH, f"{OBJECT_DETECTOR}.pth")
        object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

        clip_arch = "ViT-L-14"
        clip_model, _, transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", device=DEVICE,
        )
        tokenizer = open_clip.get_tokenizer(clip_arch)

        names_path = os.path.join(os.path.dirname(__file__), "object_names.txt")
        with open(names_path) as f:
            classnames = [line.strip() for line in f]

        return object_detector, (clip_model, transform, tokenizer), classnames

    COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
    COLOR_CLASSIFIERS = {}

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects):
            self._image = image.convert("RGB")
            self._blank = Image.new("RGB", image.size, color="#999")
            self._objects = objects

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            if mask is not None:
                assert tuple(self._image.size[::-1]) == tuple(mask.shape)
                image = Image.composite(self._image, self._blank, Image.fromarray(mask))
            else:
                image = self._image
            image = image.crop(box[:4])
            return (transform(image), 0)

    def color_classification(image, bboxes, classname):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                clip_model, tokenizer, COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                DEVICE,
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes), batch_size=16, num_workers=4,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(box_a, box_b):
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(
            np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0,
        ) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    def evaluate(image, objects, metadata):
        correct = True
        reason = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[:req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            matched_groups.append(found_objects if matched else None)
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)

    def evaluate_reward(image, objects, metadata):
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}=={req['count']}, found {len(found_objects)}")
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            matched_groups.append(found_objects if matched else None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)

    def evaluate_image(image_pils, metadatas, only_strict):
        results = inference_detector(object_detector, [np.array(img) for img in image_pils])
        ret = []
        for result, image_pil, metadata in zip(results, image_pils, metadatas):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
            for index, classname in enumerate(classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
                ordering = ordering[:MAX_OBJECTS].tolist()
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append(
                        (bbox[index][max_obj], None if segm is None else segm[index][max_obj])
                    )
                    ordering = [
                        obj for obj in ordering
                        if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
                    ]
                if not detected[classname]:
                    del detected[classname]
            is_strict_correct, score, reason = evaluate_reward(image, detected, metadata)
            is_correct = False if only_strict else evaluate(image, detected, metadata)[0]
            ret.append({
                "tag": metadata["tag"],
                "prompt": metadata["prompt"],
                "correct": is_correct,
                "strict_correct": is_strict_correct,
                "score": score,
                "reason": reason,
                "metadata": json.dumps(metadata),
                "details": json.dumps({
                    key: [box.tolist() for box, _ in value]
                    for key, value in detected.items()
                }),
            })
        return ret

    object_detector, (clip_model, transform, tokenizer), classnames = load_models()
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1

    @torch.no_grad()
    def compute_geneval(images, metadatas, only_strict=False):
        required_keys = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
        scores, rewards, strict_rewards = [], [], []
        grouped_strict_rewards = defaultdict(list)
        grouped_rewards = defaultdict(list)
        results = evaluate_image(images, metadatas, only_strict=only_strict)
        for result in results:
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                val = (1.0 if result["strict_correct"] else 0.0) if key == tag else -10.0
                grouped_strict_rewards[key].append(val)
                val = (1.0 if result["correct"] else 0.0) if key == tag else -10.0
                grouped_rewards[key].append(val)
        return scores, rewards, strict_rewards, dict(grouped_rewards), dict(grouped_strict_rewards)

    return compute_geneval
