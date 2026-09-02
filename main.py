import argparse
import cv2 as cv
import os

from features.sift_extractor import SIFTExtractor
from features.orb_detector import ORBExtractor
import config

def parse_args():
    parser = argparse.ArgumentParser(description = "Structure from Motion")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--extractor", type=str, default="sift", choices=["sift", "orb"])
    parser.add_argument("--n_features", type=int, default=0)
    parser.add_argument("--output", type=str, default="output")

    return parser.parse_args()

def load_images(folder):
    """
    Load all images from a folder.
    
    Returns:
        list of (filename, image) tuples, sorted by filename
    """
    images = []
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    filenames = [filename for filename in os.listdir(folder) if filename.lower().endswith(valid_extensions)]
    filenames.sort()
    for filename in filenames:
        path = os.path.join(folder, filename)
        image = cv.imread(path)
        if image is not None:
            images.append((filename, image))
    
    print(f"Loaded {len(images)} images from '{folder}'")
    return images

def get_extractor(args):
    if args.extractor == "sift":
        return SIFTExtractor(n_features=args.n_features)
    elif args.extractor == "orb":
        return ORBExtractor(n_features=args.n_features)
    else:
        raise ValueError(f"use either 'sift' or 'orb': {args.extractor}")

def main():
        args      = parse_args()
        extractor = get_extractor(args)
        images    = load_images(args.images)

        print(f"\nUsing extractor: {args.extractor}")

        for i, (filename, image) in enumerate(images):
            kp, des = extractor.extract(image)
            print(f"[{i + 1}/{len(images)}] {filename} : {len(kp)} keypoints")
if __name__ == "__main__":
    main()
