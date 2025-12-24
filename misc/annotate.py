import cv2
import numpy as np
import json
import os
import sys
import glob

BASE_DIR = "../ml/data/images"
ANNOTATION_DIR = "../ml/data/annotations"
os.makedirs(ANNOTATION_DIR, exist_ok=True)

clicked_points = []
NUM_CLICKS_REQUIRED = 4

CLICK_LABELS = ["TL", "TR", "BL", "BR"]
CLICK_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
]


def interpolate_centroids(P_TL, P_TR, P_BL, P_BR, num_rows, num_cols):
    """
    Generate a num_rows × num_cols grid via bilinear interpolation.

    Args:
        P_TL (tuple): Top-left corner point (x, y).
        P_TR (tuple): Top-right corner point (x, y).
        P_BL (tuple): Bottom-left corner point (x, y).
        P_BR (tuple): Bottom-right corner point (x, y).
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.

    Returns:
        list: List of [x, y] coordinates for all grid points.
    """
    all_points = []
    tl, tr, bl, br = map(np.array, [P_TL, P_TR, P_BL, P_BR])

    for r in range(num_rows):
        for c in range(num_cols):
            u = c / max(1, num_cols - 1)
            v = r / max(1, num_rows - 1)
            pt = (1 - u) * (1 - v) * tl + \
                 u * (1 - v) * tr + \
                 (1 - u) * v * bl + \
                 u * v * br
            all_points.append([int(pt[0]), int(pt[1])])

    return all_points


def mouse_callback_grid(event, x, y, flags, param):
    """
    Record 4-corner clicks for interpolation.

    Mouse callback function for OpenCV window. Records up to 4 corner clicks
    and visualizes them on the image with labels and colors.

    Args:
        event: OpenCV mouse event type.
        x (int): X-coordinate of mouse click.
        y (int): Y-coordinate of mouse click.
        flags: OpenCV mouse event flags.
        param (dict): Dictionary containing "image" and "window_name" keys.
    """
    global clicked_points

    img = param["image"]
    window = param["window_name"]

    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < NUM_CLICKS_REQUIRED:
        clicked_points.append((x, y))
        idx = len(clicked_points) - 1
        cv2.circle(img, (x, y), 6, CLICK_COLORS[idx], -1)
        cv2.putText(img, CLICK_LABELS[idx], (x + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLICK_COLORS[idx], 2)
        cv2.imshow(window, img)
        print(f"[{CLICK_LABELS[idx]}] = ({x}, {y})")


def load_json(split_name):
    """
    Load JSON annotations for a dataset split.

    Args:
        split_name (str): Dataset split name (e.g., "train", "val", "test").

    Returns:
        dict: Annotation dictionary, or empty dict if file doesn't exist or is corrupted.
    """
    path = os.path.join(ANNOTATION_DIR, f"{split_name}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        print(f"⚠ JSON corrupted: {path}, resetting.")
        return {}


def save_json(data, split_name):
    """
    Save annotation data to JSON file.

    Args:
        data (dict): Annotation data dictionary.
        split_name (str): Dataset split name (e.g., "train", "val", "test").
    """
    path = os.path.join(ANNOTATION_DIR, f"{split_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def review_and_edit(img, points):
    """
    Preview annotation and allow user to edit or save.

    Shows preview of annotated points. User can:
    - Press 'e' → edit mode (remove/add points)
    - Press 'm' → redo entire annotation manually
    - Press any other key → save without modification

    Args:
        img (np.ndarray): Image array.
        points (list): List of (x, y) coordinate tuples.

    Returns:
        list: Final list of (x, y) coordinate tuples after user interaction.
    """
    def draw(base, pts):
        canvas = base.copy()
        for (x, y) in pts:
            cv2.circle(canvas, (x, y), 4, (255, 0, 0), -1)
        return canvas

    win = "Review (e=edit, m=manual, any key=save)"
    preview = draw(img, points)
    cv2.namedWindow(win)
    cv2.imshow(win, preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)

    if key == ord("m"):
        print("→ FULL MANUAL MODE: click all centroids, press 'q' when done.")
        new_pts = []
        work = img.copy()
        cv2.namedWindow(win)

        def manual_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                new_pts.append((x, y))
                cv2.circle(work, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(win, work)
                print(f"Added → ({x},{y})")

        cv2.setMouseCallback(win, manual_click)

        while True:
            cv2.imshow(win, work)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        cv2.destroyWindow(win)
        print(f"✔ Manual annotation complete: {len(new_pts)} points.")
        return new_pts

    if key == ord("e"):
        print("→ EDIT MODE: click a point to remove, click empty space to add, press 's' to save.")
        edited = points.copy()
        work = draw(img, edited)
        cv2.namedWindow(win)

        def edit_click(event, x, y, flags, param):
            nonlocal edited, work

            if event == cv2.EVENT_LBUTTONDOWN:
                removed = False
                for i, (px, py) in enumerate(edited):
                    if abs(px - x) < 10 and abs(py - y) < 10:
                        removed_point = edited.pop(i)
                        print(f"Removed point {removed_point}")
                        removed = True
                        break

                if not removed:
                    edited.append((x, y))
                    print(f"Added ({x},{y})")

                work = draw(img, edited)
                cv2.imshow(win, work)

        cv2.setMouseCallback(win, edit_click)

        while True:
            cv2.imshow(win, work)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                break

        cv2.destroyWindow(win)
        print(f"✔ Edited annotation: {len(edited)} points.")
        return edited

    print("✔ Saved without modification.")
    return points


def batch_annotate_images(base_dir, ann_dir):
    """Batch annotate images with interactive grid or manual point selection.

    Processes all unannotated BMP images in the base directory. For each image:
    1. Shows preview and asks for grid configuration
    2. Supports staggered fields (manual mode) or regular grids (4-corner interpolation)
    3. Allows review/edit before saving annotations to JSON

    Args:
        base_dir (str): Base directory containing train/val/test subdirectories.
        ann_dir (str): Directory where annotation JSON files are saved.
    """
    global clicked_points

    splits = ["train", "val", "test"]
    annotations = {s: load_json(s) for s in splits}

    image_paths = sorted(glob.glob(os.path.join(base_dir, "**", "*.bmp"), recursive=True))
    if not image_paths:
        print("No images found.")
        return

    todo = []
    for p in image_paths:
        filename = os.path.basename(p)
        norm = p.replace("\\", "/")
        split = next((s for s in splits if f"/{s}/" in norm), None)
        if not split:
            continue
        if filename not in annotations[split]:
            todo.append((p, split))

    print(f"\nTotal images: {len(image_paths)}")
    print(f"To annotate: {len(todo)}")

    for idx, (path, split) in enumerate(todo):
        filename = os.path.basename(path)
        print(f"\n[{idx + 1}/{len(todo)}] {filename} (split: {split})")

        img = cv2.imread(path)
        if img is None:
            print("Could not read image.")
            continue

        h, w = img.shape[:2]
        display_img = img.copy()

        preview_win = f"Preview - {filename}"
        cv2.namedWindow(preview_win)
        cv2.imshow(preview_win, img)
        print("\n➡ Preview displayed. Examine the image to determine rows & columns.")
        print("   Press ANY KEY in the image window to continue.")
        cv2.waitKey(0)
        cv2.destroyWindow(preview_win)

        print("\nEnter grid configuration for this image:")
        rows = int(input("  Number of rows: "))
        cols = int(input("  Number of columns: "))
        staggered = input("  Is this a staggered field? (y/n): ").strip().lower()

        if staggered == "y":
            print("➡ STAGGERED FIELD → FULL MANUAL MODE")
            print("Click ALL centroids. Press 'q' when done.")

            win = f"Manual - {filename}"
            temp = img.copy()
            clicked_points = []

            def manual_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked_points.append((x, y))
                    cv2.circle(temp, (x, y), 4, (0, 0, 255), -1)
                    print(f"Point {len(clicked_points)} → ({x},{y})")
                    cv2.imshow(win, temp)

            cv2.namedWindow(win)
            cv2.setMouseCallback(win, manual_click, {"image": temp})

            while True:
                cv2.imshow(win, temp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

            cv2.destroyWindow(win)
            final_pts = clicked_points[:]
            print(f"✔ Manual annotation complete: {len(final_pts)} points.")

            final_pts = review_and_edit(img, final_pts)

        else:
            print(f"➡ AUTO-GRID MODE → {rows}×{cols}")
            print("Click TL → TR → BL → BR of the grid.")

            win = f"Grid - {filename}"
            clicked_points = []
            grid_img = display_img.copy()

            cv2.namedWindow(win)
            cv2.setMouseCallback(win, mouse_callback_grid,
                                 {"image": grid_img, "window_name": win})

            while True:
                cv2.imshow(win, grid_img)
                k = cv2.waitKey(1) & 0xFF
                if k in [ord('q'), 27] or len(clicked_points) == 4:
                    break

            cv2.destroyWindow(win)

            if len(clicked_points) != 4:
                print("⚠ Not enough points. Skipping.")
                continue

            TL, TR, BL, BR = clicked_points[:4]
            final_pts = interpolate_centroids(TL, TR, BL, BR, rows, cols)

            final_pts = review_and_edit(img, final_pts)

        annotations[split][filename] = {
            "width": w,
            "height": h,
            "centroids": final_pts
        }

        save_json(annotations[split], split)
        print(f"✔ Saved {len(final_pts)} points → {split}.json")

    print("\n=== ANNOTATION COMPLETE ===")
    for s in splits:
        print(f"{s.upper()}: {len(annotations[s])} annotated images.")
    print("===========================")


if __name__ == "__main__":
    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: Base directory {BASE_DIR} not found.")
        sys.exit(1)

    batch_annotate_images(BASE_DIR, ANNOTATION_DIR)
