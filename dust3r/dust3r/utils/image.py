# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import glob 
import time

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.jpg']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs


def load_images_intr(folder_or_list, size, intrinsics, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    images = []

    # Process all the cameras
    for index, folder in enumerate(folder_or_list):
        if isinstance(folder, str):
            if verbose:
                print(f'>> Loading images from {folder}')
            root, folder_content = folder, sorted(os.listdir(folder))

        elif isinstance(folder, list):
            if verbose:
                print(f'>> Loading a list of {len(folder)} images')
            root, folder_content = '', folder

        else:
            raise ValueError(f'bad {folder=} ({type(folder)})')

        supported_images_extensions = ['.jpg', '.jpeg', '.png', '.jpg']
        if heif_support_enabled:
            supported_images_extensions += ['.heic', '.heif']
        supported_images_extensions = tuple(supported_images_extensions)

        imgs = []
        original_image_size = None
        for i, path in enumerate(folder_content):
            if not path.lower().endswith(supported_images_extensions):
                continue
            img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
            if i==0:
                original_image_size = img.size[::-1]
            W1, H1 = img.size

            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
            W, H = img.size

            #if intrinsics[index] is None:
            cx, cy = W//2, H//2
                #print(f' - using center {cx, cy}')
            # else:
            #     cx, cy = intrinsics[index]['pp'][0], intrinsics[index]['pp'][1]
                #print(f' - using given intrinsics for center {cx, cy}')

            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not (square_ok) and W == H:
                    halfh = 3*halfw/4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

            W2, H2 = img.size
            if verbose:
                print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
            imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
                [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

        assert imgs, 'no images foud at '+root
        if verbose:
            print(f' (Found {len(imgs)} images)')
        images.append(imgs)
        
    return images, original_image_size


def load_single_images_intr(folder, size, intrinsics, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    images = []
    # Process all the cameras
    if isinstance(folder, str):
        if verbose:
            print(f'>> Loading images from {folder}')
        root, folder_content = folder, sorted(os.listdir(folder))

    elif isinstance(folder, list):
        if verbose:
            print(f'>> Loading a list of {len(folder)} images')
        root, folder_content = '', folder

    else:
        raise ValueError(f'bad {folder=} ({type(folder)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.jpg']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    original_image_size = None
    for i, path in enumerate(folder_content):
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        if i==0:
            original_image_size = img.size[::-1]
        W1, H1 = img.size

        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size

        cx, cy = W//2, H//2

        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    images.append(imgs)
    images = images[0]
        
    return images, original_image_size


def load_masks(folder_or_list, image_list, intrinsics, size, square_ok=False, verbose=True):
    """
    Open and process mask images while aligning them with an images list.
    If a mask for an image is missing, appends None to ensure alignment.

    Args:
        folder_or_list (str or list): A folder path containing mask images or a list of mask paths.
        image_list (list): A list of image filenames (without extensions) to align masks with.
        size (int): Target size for resizing the masks.
        square_ok (bool): Whether to allow square masks without additional cropping.
        verbose (bool): Whether to print detailed logs.

    Returns:
        list: A list aligned with the image_list, where each entry is:
            - Processed mask data if the mask exists.
            - None if the mask is missing.
    """
    aligned_masks_total = []  # Final list aligned with image_list

    for i, folder in enumerate(folder_or_list):
        aligned_masks = []  # Masks aligned with the current image_list
        if isinstance(folder, str):
            if verbose:
                print(f'>> Loading masks from {folder}')
            root, folder_content = folder, sorted(os.listdir(folder))
        elif isinstance(folder, list):
            if verbose:
                print(f'>> Loading a list of {len(folder)} masks')
            root, folder_content = '', folder
        else:
            raise ValueError(f'Invalid input: {folder=} ({type(folder)})')

        # Supported mask extensions
        supported_masks_extensions = ['.png', '.bmp', '.jpg', '.jpeg']
        supported_masks_extensions = tuple(supported_masks_extensions)

        mask_paths = folder

        # Normalize paths to use forward slashes for compatibility
        mask_paths = [path.replace("\\", "/") for path in mask_paths]
        
        # Sort the paths to ensure consistent ordering
        mask_paths.sort()

        mask_filenames = {os.path.splitext(os.path.basename(path))[0]: path for path in mask_paths}
        img_filenames = {os.path.splitext(os.path.basename(path))[0]: path for path in image_list[i]}

        # Align masks with the image list
        for img_name in img_filenames:
            mask_path = mask_filenames.get(img_name, None)

            if mask_path:
                # Load and process the mask
                mask = PIL.Image.open(mask_path).convert('L')  # Convert to grayscale
                W1, H1 = mask.size

                # Resize and crop
                if size == 224:
                    mask = _resize_pil_image(mask, round(size * max(W1 / H1, H1 / W1)))
                else:
                    mask = _resize_pil_image(mask, size)

                W, H = mask.size
                #if intrinsics[i] is None:
                cx, cy = W // 2, H // 2
                    #print(f' - using center {cx, cy} for mask {img_name}')
                # else:
                #     cx, cy = intrinsics[i]['pp'][0], intrinsics[i]['pp'][1]
                    #print(f' - using given intrinsics for center {cx, cy} for mask {img_name}')

                if size == 224:
                    half = min(cx, cy)
                    mask = mask.crop((cx - half, cy - half, cx + half, cy + half))
                else:
                    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                    if not square_ok and W == H:
                        halfh = 3 * halfw // 4
                    mask = mask.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

                # Convert to NumPy array
                mask_data = np.array(mask, dtype=np.uint8)

                if verbose:
                    print(f' - Found mask for {img_name}: {mask_path}, resized to {W}x{H}')
                aligned_masks.append(mask_data)  # Append processed mask
            else:
                if verbose:
                    print(f' - No mask found for {img_name}')
                aligned_masks.append(None)  # Append None if mask is missing
        aligned_masks_total.append(aligned_masks)

    return aligned_masks_total



def load_single_masks(folder, image_list, intrinsics, size, square_ok=False, verbose=True):
    """
    Open and process mask images while aligning them with an images list.
    If a mask for an image is missing, appends None to ensure alignment.

    Args:
        folder_or_list (str or list): A folder path containing mask images or a list of mask paths.
        image_list (list): A list of image filenames (without extensions) to align masks with.
        size (int): Target size for resizing the masks.
        square_ok (bool): Whether to allow square masks without additional cropping.
        verbose (bool): Whether to print detailed logs.

    Returns:
        list: A list aligned with the image_list, where each entry is:
            - Processed mask data if the mask exists.
            - None if the mask is missing.
    """
    aligned_masks_total = []  # Final list aligned with image_list

    aligned_masks = []  # Masks aligned with the current image_list
    if isinstance(folder, str):
        if verbose:
            print(f'>> Loading masks from {folder}')
        root, folder_content = folder, sorted(os.listdir(folder))
    elif isinstance(folder, list):
        if verbose:
            print(f'>> Loading a list of {len(folder)} masks')
        root, folder_content = '', folder
    else:
        raise ValueError(f'Invalid input: {folder=} ({type(folder)})')

    # Supported mask extensions
    supported_masks_extensions = ['.png', '.bmp', '.jpg', '.jpeg']
    supported_masks_extensions = tuple(supported_masks_extensions)

    mask_paths = folder

    # Normalize paths to use forward slashes for compatibility
    mask_paths = [path.replace("\\", "/") for path in mask_paths]
    
    # Sort the paths to ensure consistent ordering
    mask_paths.sort()

    mask_filenames = {os.path.splitext(os.path.basename(path))[0]: path for path in mask_paths}
    img_filenames = {os.path.splitext(os.path.basename(path))[0]: path for path in image_list}

    # Align masks with the image list
    for img_name in img_filenames:
        mask_path = mask_filenames.get(img_name, None)

        if mask_path:
            # Load and process the mask
            mask = PIL.Image.open(mask_path).convert('L')  # Convert to grayscale
            W1, H1 = mask.size

            # Resize and crop
            if size == 224:
                mask = _resize_pil_image(mask, round(size * max(W1 / H1, H1 / W1)))
            else:
                mask = _resize_pil_image(mask, size)

            W, H = mask.size
            #if intrinsics[i] is None:
            cx, cy = W // 2, H // 2
                #print(f' - using center {cx, cy} for mask {img_name}')
            # else:
            #     cx, cy = intrinsics[i]['pp'][0], intrinsics[i]['pp'][1]
                #print(f' - using given intrinsics for center {cx, cy} for mask {img_name}')

            if size == 224:
                half = min(cx, cy)
                mask = mask.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not square_ok and W == H:
                    halfh = 3 * halfw // 4
                mask = mask.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

            # Convert to NumPy array
            mask_data = np.array(mask, dtype=np.uint8)

            if verbose:
                print(f' - Found mask for {img_name}: {mask_path}, resized to {W}x{H}')
            aligned_masks.append(mask_data)  # Append processed mask
        else:
            if verbose:
                print(f' - No mask found for {img_name}')
            aligned_masks.append(None)  # Append None if mask is missing
    aligned_masks_total.append(aligned_masks)

    return aligned_masks_total