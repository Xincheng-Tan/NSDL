from __future__ import annotations


VIKI_L2_TRAIN_TASK_NAMES = {
    "clear_table_with_two_robots_and_put_in_cabinet",
    "cut_fruit_on_board",
    "cut_two_fruits_on_board",
    "dog_check_environment",
    "dog_push_box_for_two_panda_transport",
    "ensure_all_fruits_on_table",
    "parallel_human_dual_asset_to_plate_or_bowl",
    "sequential_pick_two_and_place",
    "serve_bread_after_checking_cabinet",
    "serve_bread_from_counter",
    "set_plate_and_fork_on_table",
    "single_move_asset_to_target",
    "toast_bread_and_set_plate",
    "wash_fruit_and_serve",
}

VIKI_L2_VAL_TASK_NAMES = {
    "bring_bowl_to_table_plate_already_there",
    "bring_plate_and_bowl_to_table",
    "bring_plate_to_table_bowl_already_there",
}

VIKI_L2_TEST_TASK_NAMES = {
    "clear_table_with_two_robots_and_put_in_cabinet",
    "cut_fruit_on_board",
    "cut_two_fruits_on_board",
    "dog_check_environment",
    "dog_push_box_for_two_panda_transport",
    "ensure_all_fruits_on_table",
    "parallel_human_dual_asset_to_plate_or_bowl",
    "sequential_pick_two_and_place",
    "serve_bread_after_checking_cabinet",
    "serve_bread_from_counter",
    "set_plate_and_fork_on_table",
    "single_move_asset_to_target",
    "toast_bread_and_set_plate",
    "wash_fruit_and_serve",
}

VIKI_L2_VAL_TEST_ONLY_TASK_NAMES = (
    VIKI_L2_VAL_TASK_NAMES | VIKI_L2_TEST_TASK_NAMES
) - VIKI_L2_TRAIN_TASK_NAMES


def assert_split_safe_macro_motif_names(motif_names: set[str]) -> None:
    overlap = motif_names & VIKI_L2_VAL_TEST_ONLY_TASK_NAMES
    if overlap:
        raise RuntimeError(
            "Macro motif names must not collide with val/test-only task names. "
            f"Found overlap: {sorted(overlap)}"
        )
