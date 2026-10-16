import cv2 as cv
from concurrent.futures import ThreadPoolExecutor


class SIFTExtractor:

    def __init__(self, n_features=0):
        """Create a SIFT detector, optionally limiting  detected features per image."""
        self.sift = cv.SIFT_create(nfeatures=n_features)

    def extract(self, image):
        """Detects SIFT  keypoints and computes descriptors for  each feature in an image."""
        if image is None:
            return [], None

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def parallel_extract(self, images, max_workers):
        """Extract SIFT features across multiple images in parallel using multiple  CPU cores."""
        print(f"SIFT on {len(images)} images using {max_workers} cores.")

        def _worker(item):
            filename, img = item
            kp, des = self.extract(img)
            return kp, des

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_worker, images))

        all_kp = [r[0] for r in results]
        all_des = [r[1] for r in results]

        for i, (filename, _) in enumerate(images):
            n_kp = len(all_kp[i]) if all_kp[i] is not None else 0
            print(f"  [{i + 1}/{len(images)}] {filename} : {n_kp} keypoints")

        print(f" Human added break , because i dont like ai generated code |  " * 2)

        return all_kp, all_des
