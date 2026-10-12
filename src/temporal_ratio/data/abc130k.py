"""ABC-130k task lists (https://huggingface.co/datasets/XDOF/ABC-130k).

Pure-Python constants shared by the dataset registry, the video datasets and the MCAP -> LeRobot
converter (``scripts/data/convert_abc130k_to_lerobot.py``). A converted task is a LeRobot v2.1
dataset with id ``abc130k/<task>`` under ``$HF_LEROBOT_HOME``.

The released YAM checkpoints were trained on 24 bimanual tasks ("pp24"). 22 of them exist in
ABC-130k; ``handoff`` and ``putRelative`` do not. ``PP24_NAMES`` records the correspondence.
``EXTRA_TASKS`` are further ABC-130k tasks used here, and ``TASK_GROUPS`` names the task sets that
have a dataset config (``videoVLA_abc130k_<group>_short``) and a video mixture (``abc130k_<group>``).
Validation episodes (the Hub's ``val`` split) are converted to ``abc130k/<task>_val``.
"""

from __future__ import annotations

SKILLS: dict[str, tuple[str, ...]] = {
    "folding": ("fold_and_stack_the_t_shirts",),
    "garment": (
        "remove_the_shirt_from_the_hanger",
        "remove_the_t_shirt_from_the_hanger",
    ),
    "assembly": ("unscrew_the_bottle_caps",),
    "pickplace": (
        "place_the_mixed_dishes_into_the_plastic_bin",
        "place_the_plates_into_the_plastic_bin_on_the_countertop",
        "load_the_bowls_into_the_dish_rack",
        "load_the_cups_into_the_dish_rack",
        "load_the_mixed_dishes_into_the_dish_rack",
        "load_the_plates_into_the_dish_rack",
        "place_and_organize_the_beverage_onto_the_shelf",
        "place_and_organize_the_canned_foods_onto_the_counter",
        "place_and_organize_the_chips_bags_onto_the_shelf",
        "place_and_organize_the_fake_fruits_in_the_fruit_bowl",
        "place_the_snacks_into_the_paper_bag",
        "take_the_snacks_out_of_the_paper_bag",
        "throw_the_plastic_bottles_in_the_bin",
        "turn_the_mug_right_side_up",
        "unload_the_bowls_from_the_dish_rack",
        "unload_the_cups_from_the_dish_rack",
        "unload_the_mixed_dishes_from_the_dish_rack",
        "unload_the_plates_from_the_dish_rack",
    ),
}

#: All 22 tasks, in a fixed order (folding first, as in the original multi-task config).
TASKS: tuple[str, ...] = tuple(t for tasks in SKILLS.values() for t in tasks)

#: ABC-130k tasks outside the checkpoints' training set.
EXTRA_TASKS: tuple[str, ...] = ("open_the_pen_caps",)

#: Task sets with a dataset config and a video mixture.
TASK_GROUPS: dict[str, tuple[str, ...]] = {
    **SKILLS,
    "pen_caps": ("open_the_pen_caps",),
    # T-shirt folding, pen-cap opening and bottle-cap unscrewing.
    "fold_pen_bottle": ("fold_and_stack_the_t_shirts", "open_the_pen_caps", "unscrew_the_bottle_caps"),
}

#: ABC-130k task -> (pp24 skill, pp24 task) of the datasets the released checkpoints were trained on.
PP24_NAMES: dict[str, tuple[str, str]] = {
    "fold_and_stack_the_t_shirts": ("folding", "foldingTshirtPileAndStacking"),
    "remove_the_shirt_from_the_hanger": ("garmentManipulation", "removeTheShirtFromTheHanger"),
    "remove_the_t_shirt_from_the_hanger": ("garmentManipulation", "removeTheTshirtFromTheHanger"),
    "unscrew_the_bottle_caps": ("highPrecisionAssembly", "unscrewBottleCaps"),
    "place_the_mixed_dishes_into_the_plastic_bin": ("pickAndPlace", "busTablePlacingMixedDishesGlassesIntoAPlasticBin"),
    "place_the_plates_into_the_plastic_bin_on_the_countertop": ("pickAndPlace", "busTablePlacingPlatesIntoAPlasticBin"),
    "load_the_bowls_into_the_dish_rack": ("pickAndPlace", "loadBowlsIntoTabletopDishRack"),
    "load_the_cups_into_the_dish_rack": ("pickAndPlace", "loadCupsIntoTabletopDishRack"),
    "load_the_mixed_dishes_into_the_dish_rack": ("pickAndPlace", "loadMixedDishesIntoTabletopDishRack"),
    "load_the_plates_into_the_dish_rack": ("pickAndPlace", "loadPlatesIntoTabletopDishRack"),
    "place_and_organize_the_beverage_onto_the_shelf": ("pickAndPlace", "placeOrganizeBeverageOntoShelf"),
    "place_and_organize_the_canned_foods_onto_the_counter": ("pickAndPlace", "placeOrganizeCannedFoodsOntoCounter"),
    "place_and_organize_the_chips_bags_onto_the_shelf": ("pickAndPlace", "placeOrganizeChipsBagsOntoShelf"),
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": ("pickAndPlace", "placeOrganizeFakeFruitsIntoFruitBowl"),
    "place_the_snacks_into_the_paper_bag": ("pickAndPlace", "placeSnacksIntoPaperBag"),
    "take_the_snacks_out_of_the_paper_bag": ("pickAndPlace", "takeSnacksOutOfPaperBag"),
    "throw_the_plastic_bottles_in_the_bin": ("pickAndPlace", "throwPlasticBottlesInBin"),
    "turn_the_mug_right_side_up": ("pickAndPlace", "turnMugRightSideUp"),
    "unload_the_bowls_from_the_dish_rack": ("pickAndPlace", "unloadBowlsFromTabletopDishRack"),
    "unload_the_cups_from_the_dish_rack": ("pickAndPlace", "unloadCupsFromTabletopDishRack"),
    "unload_the_mixed_dishes_from_the_dish_rack": ("pickAndPlace", "unloadMixedDishesFromTabletopDishRack"),
    "unload_the_plates_from_the_dish_rack": ("pickAndPlace", "unloadPlatesFromTabletopDishRack"),
}

#: Per-camera frame size (H, W) written by the converter; the policy pads to 176 x 224.
IMAGE_SIZE = (168, 224)
FPS = 30
#: Temporal subsampling used for both the action and the video stream.
FRAME_SKIP = 3


def repo_id(task: str, split: str = "train") -> str:
    """LeRobot dataset id of a converted task (``split="val"`` for the validation episodes)."""
    if task not in PP24_NAMES and task not in EXTRA_TASKS:
        raise KeyError(f"unknown ABC-130k task {task!r}")
    return f"abc130k/{task}" if split == "train" else f"abc130k/{task}_{split}"


def repo_ids(group: str | None = None) -> tuple[str, ...]:
    """Dataset ids of one task group (``TASK_GROUPS``), or of the 22 tasks shared with pp24."""
    tasks = TASKS if group is None else TASK_GROUPS[group]
    return tuple(repo_id(t) for t in tasks)
