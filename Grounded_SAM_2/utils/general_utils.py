import numpy as np

def save_boxes(input_boxes, file_path):
    """
    Convert bounding boxes from x1, y1, x2, y2 to x, y, w, h format and save to a text file.

    Parameters:
        input_boxes (numpy.ndarray): Array of boxes in x1, y1, x2, y2 format.
        file_path (str): Path to save the converted boxes in .txt format.
    """
    # Initialize an array for the converted boxes
    boxes_xywh = np.zeros_like(input_boxes)
    # Convert format
    boxes_xywh[:, 0] = input_boxes[:, 0]  # x = x1
    boxes_xywh[:, 1] = input_boxes[:, 1]  # y = y1
    boxes_xywh[:, 2] = input_boxes[:, 2] - input_boxes[:, 0]  # w = x2 - x1
    boxes_xywh[:, 3] = input_boxes[:, 3] - input_boxes[:, 1]  # h = y2 - y1

    # Save the result to a txt file
    np.savetxt(file_path, boxes_xywh, fmt="%.6f", delimiter=",", header="", comments='')


def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts