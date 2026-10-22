"""Map the BDDL paths stored in LIBERO demonstration files to the installed LIBERO package.

LIBERO HDF5 files store ``env_args.bddl_file`` relative to the original data
collection layout (``chiliocosm/bddl_files/<suite>/<task>.bddl``).  This module
resolves such paths against the ``bddl_files`` directory of the LIBERO package
found on ``sys.path`` (see ``README.md`` for the expected checkout).
"""
import glob
import os

from diffusion_policy.env_runner.libero_paths import get_libero_bddl_root

_BDDL_ROOT = get_libero_bddl_root()

bddl_file_name_dict = {}

_SUITES = ["libero_goal", "libero_10", "libero_90", "libero_object", "libero_spatial"]
for _suite in _SUITES:
    for _file in glob.glob(os.path.join(_BDDL_ROOT, _suite, "*.bddl")):
        _name = os.path.basename(_file)
        if _suite in ("libero_10", "libero_90"):
            bddl_file_name_dict[f"chiliocosm/bddl_files/libero_100/{_name}"] = _file
        else:
            bddl_file_name_dict[f"chiliocosm/bddl_files/{_suite}/{_name}"] = _file

# Tasks whose file names were changed between the data-collection layout and the
# public LIBERO release.
_RENAMED = {
    "libero_goal/open_the_middle_layer_of_the_drawer.bddl": "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
    "libero_goal/open_the_top_layer_of_the_drawer_and_put_the_bowl_inside.bddl": "libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl",
    "libero_goal/put_the_cream_cheese_on_the_bowl.bddl": "libero_goal/put_the_cream_cheese_in_the_bowl.bddl",
    "libero_goal/put_the_bowl_on_the_top_of_the_drawer.bddl": "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
    "libero_goal/put_the_wine_bottle_on_the_top_of_the_drawer.bddl": "libero_goal/put_the_wine_bottle_on_top_of_the_cabinet.bddl",
    "libero_100_debug/STUDY_TABLETOP_SCENE1_pick_up_the_book_and_place_it_in_the_back_of_the_caddy.bddl": "libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
    "libero_100/KITCHEN_SCENE2_put_the_black_bowl_in_the_middle_on_the_plate.bddl": "libero_90/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl",
    "libero_100/KITCHEN_SCENE2_stack_the_black_bowl_in_the_middle_on_the_black_bowl_at_the_front.bddl": "libero_90/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.bddl",
    "libero_100_debug/KITCHEN_TABLETOP_SCENE9_put_the_frypan_into_the_bottom_layer_of_the_cabinet.bddl": "libero_90/KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf.bddl",
    "libero_100/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_of_the_caddy.bddl": "libero_90/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl",
    "libero_100/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_compartment_of_the_caddy.bddl": "libero_90/STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.bddl",
    "libero_100/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_compartment_of_the_caddy.bddl": "libero_90/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy.bddl",
}
for _object in ["alphabet_soup", "bbq_sauce", "butter", "chocolate_pudding", "cream_cheese",
                "ketchup", "milk", "orange_juice", "salad_dressing", "tomato_sauce"]:
    _RENAMED[f"libero_object/pick_the_{_object}_and_place_it_in_the_basket.bddl"] = \
        f"libero_object/pick_up_the_{_object}_and_place_it_in_the_basket.bddl"
for _old, _new in [
    ("between_the_plate_and_the_ramekin", "between_the_plate_and_the_ramekin"),
    ("from_table_center", "from_table_center"),
    ("in_the_top_layer_of_the_wooden_cabinet", "in_the_top_drawer_of_the_wooden_cabinet"),
    ("next_to_the_cookies_box", "next_to_the_cookie_box"),
    ("next_to_the_plate", "next_to_the_plate"),
    ("next_to_the_ramekin", "next_to_the_ramekin"),
    ("on_the_cookies_box", "on_the_cookie_box"),
    ("on_the_ramekin", "on_the_ramekin"),
    ("on_the_stove", "on_the_stove"),
    ("on_the_wooden_cabinet", "on_the_wooden_cabinet"),
]:
    _RENAMED[f"libero_spatial/pick_the_akita_black_bowl_{_old}_and_place_it_on_the_plate.bddl"] = \
        f"libero_spatial/pick_up_the_black_bowl_{_new}_and_place_it_on_the_plate.bddl"

for _old, _new in _RENAMED.items():
    _target = os.path.join(_BDDL_ROOT, _new)
    if os.path.exists(_target):
        bddl_file_name_dict[f"chiliocosm/bddl_files/{_old}"] = _target


def resolve_bddl_file(stored_path: str) -> str:
    """Return a local BDDL path for ``stored_path`` (or ``stored_path`` itself if it already exists)."""
    if os.path.exists(stored_path):
        return stored_path
    if stored_path in bddl_file_name_dict:
        return bddl_file_name_dict[stored_path]
    raise FileNotFoundError(
        f"cannot resolve BDDL file '{stored_path}'. Is LIBERO checked out (bddl root: {_BDDL_ROOT})?")
