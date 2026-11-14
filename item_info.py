"""
Metadata and kinematic motion-type groupings for the 39 tool-use pantomime items.

Each gesture trial is scored manually on content (correct tool pantomimed: 0 or 1)
and, when content = 1, three binary kinematic components:
  - Hand Posture (HP): Grasp configuration and finger posture (0 or 1)
  - Arm Posture & Trajectory (AP): Arm orientation and workspace path (0 or 1)
  - Amplitude & Timing (AT): Movement scale, rhythm, and repetition count (0 or 1)
"""

ITEM_INFO = {
    1: {
        "name": "Beer Mug",
        "category": "drinking",
        "hand_posture": "Fist (grasping handle) or C-shape (grasping mug)",
        "arm_posture": "Lift to side of mouth (fist) or directly to mouth (C-shape)",
        "amplitude": "From table/space to face",
        "key_motion": "lifting, tilting",
    },
    2: {
        "name": "Ping-Pong Paddle",
        "category": "sports",
        "hand_posture": "Fist or open hand",
        "arm_posture": "Swinging back and forth in front of body",
        "amplitude": "Enough to move ball forward",
        "key_motion": "swinging, back-and-forth",
    },
    3: {
        "name": "Bolt Lock",
        "category": "tool",
        "hand_posture": "Fingers curled, thumb pinch with index/middle finger",
        "arm_posture": "Reach out in front, arm twists",
        "amplitude": "Hand must twist (not shake), 1-3 twists",
        "key_motion": "twisting, rotating",
    },
    4: {
        "name": "Tea Bag",
        "category": "food/drink",
        "hand_posture": "Pinch with fingertips",
        "arm_posture": "Move hand up and down (dipping motion)",
        "amplitude": "Hand shouldn't go too far into cup",
        "key_motion": "dipping, up-down",
    },
    5: {
        "name": "Comb",
        "category": "grooming",
        "hand_posture": "Clenched hand or fist",
        "arm_posture": "Must gesture to head",
        "amplitude": "Usually okay, can comb in same location",
        "key_motion": "combing, brushing head",
    },
    6: {
        "name": "Eraser",
        "category": "writing",
        "hand_posture": "Thumb to fingertips, not pendulum motion",
        "arm_posture": "Aim at table, within 4 inches, horizontal plane",
        "amplitude": "Not larger than sheet of paper, 3+ back-forth motions",
        "key_motion": "erasing, back-forth horizontal",
    },
    7: {
        "name": "Paint Roller",
        "category": "tool",
        "hand_posture": "Fist grip, roller perpendicular to surface",
        "arm_posture": "Paint toward table or wall",
        "amplitude": "Some arm movement required, one up-down OK",
        "key_motion": "rolling, up-down",
    },
    8: {
        "name": "Knife",
        "category": "kitchen",
        "hand_posture": "Fist grip, no wrist twist, angled at table",
        "arm_posture": "Down at table, within 18 inches",
        "amplitude": "Cannot hit table",
        "key_motion": "chopping, slicing",
    },
    9: {
        "name": "Axe",
        "category": "tool",
        "hand_posture": "Fist grip, wrist OK with arm movement",
        "arm_posture": "Chopping from shoulder height to table",
        "amplitude": "Full chop motion, cannot hit table",
        "key_motion": "chopping, large vertical",
    },
    10: {
        "name": "Bottle Opener",
        "category": "kitchen",
        "hand_posture": "Prying motion, must hold opener",
        "arm_posture": "Arm elevated from table",
        "amplitude": "Opener moves less than 12 inches, max 3 times",
        "key_motion": "prying, levering",
    },
    11: {
        "name": "Hammer",
        "category": "tool",
        "hand_posture": "Fist or slightly open, no pinching",
        "arm_posture": "Vertical hammering, wrist movement OK",
        "amplitude": "Hand below head, cannot touch table",
        "key_motion": "hammering, vertical striking",
    },
    12: {
        "name": "Iron",
        "category": "household",
        "hand_posture": "Holding posture or fist, no flagrant wrist",
        "arm_posture": "Parallel to table, within 12 inches",
        "amplitude": "At least one up-back motion",
        "key_motion": "ironing, back-forth horizontal",
    },
    13: {
        "name": "Saw",
        "category": "tool",
        "hand_posture": "Fist, back of hand facing left",
        "arm_posture": "Move arm away from body and back",
        "amplitude": "Extend at least 4 inches, one back-forth motion",
        "key_motion": "sawing, push-pull",
    },
    14: {
        "name": "Paper Clip",
        "category": "office",
        "hand_posture": "Pinch with thumb/index/middle, no duck hands",
        "arm_posture": "Hand in front of body, arm extends outward",
        "amplitude": "1-3 pinches OK",
        "key_motion": "pinching, clipping",
    },
    15: {
        "name": "Razor",
        "category": "grooming",
        "hand_posture": "Closed hand grip, no face touching",
        "arm_posture": "Toward shavable body part, within 4 inches",
        "amplitude": "Two razor strokes minimum",
        "key_motion": "shaving, short strokes on face",
    },
    16: {
        "name": "Fork",
        "category": "eating",
        "hand_posture": "Fist, pencil grip, or two-finger pinch",
        "arm_posture": "From table/space to mouth",
        "amplitude": "Within 4 inches of mouth",
        "key_motion": "eating, lifting to mouth",
    },
    17: {
        "name": "Cigarette",
        "category": "smoking",
        "hand_posture": "Pointer/middle finger OR thumb-index pinch",
        "arm_posture": "From space to mouth or mouth to space",
        "amplitude": "Usually okay, may do only once",
        "key_motion": "smoking, to/from mouth",
    },
    18: {
        "name": "Deodorant",
        "category": "grooming",
        "hand_posture": "Not too broad grip, no fist",
        "arm_posture": "Toward armpit area",
        "amplitude": "Some oscillation required",
        "key_motion": "applying, up-down under arm",
    },
    19: {
        "name": "Match",
        "category": "fire",
        "hand_posture": "Wrist flick OK, must hold match",
        "arm_posture": "Some arm movement required",
        "amplitude": "Reasonable amount to light",
        "key_motion": "striking, flicking",
    },
    20: {
        "name": "Toothbrush",
        "category": "grooming",
        "hand_posture": "At least 3 fingers",
        "arm_posture": "Up-down or back-forth, within 4 inches of face",
        "amplitude": "Only where teeth are, multiple oscillations",
        "key_motion": "brushing, oscillating near mouth",
    },
    21: {
        "name": "Clamp",
        "category": "tool",
        "hand_posture": "Pinch with thumb/index/middle, no duck hands",
        "arm_posture": "In front of body, rotation from arm",
        "amplitude": "Rotation 2-12 inches, at least 2 rotations",
        "key_motion": "clamping, rotating",
    },
    22: {
        "name": "Wrench",
        "category": "tool",
        "hand_posture": "Some wrist movement, most in arm",
        "arm_posture": "Torque motion, back-forth or side-side",
        "amplitude": "Generally okay",
        "key_motion": "wrenching, torque motion",
    },
    23: {
        "name": "Clothing Pin",
        "category": "household",
        "hand_posture": "Two-finger pinch",
        "arm_posture": "Oriented outward and upward if extended",
        "amplitude": "Fingers within 3 inches, max 3 pinches",
        "key_motion": "pinching, releasing",
    },
    24: {
        "name": "Screwdriver",
        "category": "tool",
        "hand_posture": "Fist grip, thumb curled or on top",
        "arm_posture": "Turn into space or table",
        "amplitude": "Some rotation, at least 2 twists",
        "key_motion": "screwing, rotating",
    },
    25: {
        "name": "Nail Clipper",
        "category": "grooming",
        "hand_posture": "Thumb to fingertips, open-close motion",
        "arm_posture": "Toward body or into space",
        "amplitude": "Cannot open more than 3 inches",
        "key_motion": "clipping, small pinching",
    },
    26: {
        "name": "Car Jack",
        "category": "tool",
        "hand_posture": "Fist grip",
        "arm_posture": "Extend and pull back, or crank motion",
        "amplitude": "8-14 inches, at least 2 pulls",
        "key_motion": "cranking, pumping",
    },
    27: {
        "name": "Camera",
        "category": "device",
        "hand_posture": "Hold camera and press button",
        "arm_posture": "In front of face, within 6 inch sphere",
        "amplitude": "Finger cannot go far into camera",
        "key_motion": "holding, pressing button",
    },
    28: {
        "name": "Remote Control",
        "category": "device",
        "hand_posture": "Not fist, use thumb to press",
        "arm_posture": "Aimed forward, minimal twisting",
        "amplitude": "Minimal arm movement",
        "key_motion": "pressing buttons with thumb",
    },
    29: {
        "name": "Stapler",
        "category": "office",
        "hand_posture": "Push with palm or bottom of fist",
        "arm_posture": "Aimed at table",
        "amplitude": "Enough force to staple",
        "key_motion": "pressing down",
    },
    30: {
        "name": "Key",
        "category": "household",
        "hand_posture": "Thumb to curled pointer or fingertips",
        "arm_posture": "Turn arm in space",
        "amplitude": "At least 45 degrees rotation, under 8 inches",
        "key_motion": "turning, rotating",
    },
    31: {
        "name": "Lipstick",
        "category": "grooming",
        "hand_posture": "Thumb to fingertips, not fist",
        "arm_posture": "Aimed at lips, within 2 inches",
        "amplitude": "Enough to cover lips",
        "key_motion": "applying, horizontal near lips",
    },
    32: {
        "name": "Scissors",
        "category": "office",
        "hand_posture": "Open-close motion, two-finger pinch OK",
        "arm_posture": "Forward motion required",
        "amplitude": "Scissor at least twice, in sync with arm",
        "key_motion": "cutting, open-close with forward motion",
    },
    33: {
        "name": "Corkscrew",
        "category": "kitchen",
        "hand_posture": "Twist with entire hand, fist OK",
        "arm_posture": "Elevated at least 4 inches, below head",
        "amplitude": "Twist at least 90 degrees, at least twice",
        "key_motion": "twisting, screwing motion",
    },
    34: {
        "name": "Lighter",
        "category": "fire",
        "hand_posture": "Thumb flick down while holding",
        "arm_posture": "Generally correct, minimal movement",
        "amplitude": "Movement mainly in thumb",
        "key_motion": "flicking, thumb motion",
    },
    35: {
        "name": "Syringe",
        "category": "medical",
        "hand_posture": "Inject with thumb, hold with 2 fingers",
        "arm_posture": "Into space or toward arm/leg",
        "amplitude": "Max 3 times",
        "key_motion": "injecting, pressing plunger",
    },
    36: {
        "name": "Soap Dispenser",
        "category": "household",
        "hand_posture": "Press with hand/finger/thumb",
        "arm_posture": "4-10 inches from table",
        "amplitude": "Reasonable press depth",
        "key_motion": "pressing down, pumping",
    },
    37: {
        "name": "Calculator",
        "category": "device",
        "hand_posture": "Poke with one or multiple fingers",
        "arm_posture": "Toward table",
        "amplitude": "At least 2 presses",
        "key_motion": "poking, pressing buttons",
    },
    38: {
        "name": "Keyboard",
        "category": "device",
        "hand_posture": "Keyboard hands, all fingers must move",
        "arm_posture": "Toward table, side-to-side movement",
        "amplitude": "Press multiple times (3+)",
        "key_motion": "typing, multiple finger motion",
    },
    39: {
        "name": "Watch",
        "category": "device",
        "hand_posture": "Pinching to wind",
        "arm_posture": "Generally static",
        "amplitude": "No excessively large movements",
        "key_motion": "winding, small twisting",
    },
}

CATEGORIES = {
    "tool": [3, 7, 9, 11, 13, 21, 22, 24, 26],
    "grooming": [5, 15, 18, 20, 25, 31],
    "kitchen": [8, 10, 33],
    "eating/drinking": [1, 4, 16, 17],
    "office": [6, 14, 29, 32],
    "household": [12, 23, 30, 36],
    "device": [27, 28, 37, 38, 39],
    "fire": [19, 34],
    "medical": [35],
    "sports": [2],
}

MOTION_TYPES = {
    "vertical_striking": [9, 11],
    "horizontal_back_forth": [6, 12, 13],
    "rotating_twisting": [3, 21, 22, 24, 30, 33, 39],
    "lifting_to_face": [1, 4, 5, 15, 16, 17, 20, 31],
    "pressing_poking": [27, 28, 29, 34, 35, 36, 37, 38],
    "pinching": [14, 23, 25, 32],
    "swinging": [2],
    "up_down": [7, 18, 26],
    "prying": [10],
    "other": [8, 19],
}


def get_item_name(item_id):
    """Return the display name for `item_id` (1-39)."""
    if item_id in ITEM_INFO:
        return ITEM_INFO[item_id]["name"]
    return f"Item {item_id}"


def get_motion_type(item_id):
    """Return the primary kinematic motion category for `item_id`."""
    for motion_type, items in MOTION_TYPES.items():
        if item_id in items:
            return motion_type
    return "unknown"


def print_all_items():
    """Print a table of all 39 tool-use pantomime items."""
    for item_id in sorted(ITEM_INFO.keys()):
        info = ITEM_INFO[item_id]
        motion = get_motion_type(item_id)
        print(f"{item_id:2d}. {info['name']:<20} | {motion:<22} | {info['key_motion']}")


if __name__ == "__main__":
    print("TOOL-USE PANTOMIME GESTURE ITEMS (39)")
    print("=" * 72)
    print_all_items()
