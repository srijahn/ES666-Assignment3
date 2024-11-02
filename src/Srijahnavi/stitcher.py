import glob
import cv2
import os
import numpy as np

class PanaromaStitcher:
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')

        if len(all_images) < 2:
            print("Need at least two images to stitch.")
            return None, []

        # Load images and filter invalid images
        images = [cv2.imread(img) for img in all_images]
        images = [img for img in images if img is not None]
        num_images = len(images)

        if num_images < 2:
            print("Insufficient valid images to stitch.")
            return None, []

        # Canvas size 
        height, width = images[0].shape[:2]
        canvas_width = width * num_images * 2
        canvas_height = height * 2
        panorama_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Center the first image on the canvas
        center_x, center_y = canvas_width // 2 - width // 2, canvas_height // 2 - height // 2
        panorama_canvas[center_y:center_y + height, center_x:center_x + width] = images[0]

        # Initializing homography list and cumulative matrix
        homography_matrix_list = []
        cumulative_H = np.eye(3)

        # Keypoint matcher initialization
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for i in range(1, num_images):
            kp1, des1 = sift.detectAndCompute(images[i - 1], None)
            kp2, des2 = sift.detectAndCompute(images[i], None)

            # Matching the descriptors and filter good matches
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 0.75 * matches[-1].distance]

            if len(good_matches) < 10:
                print(f"Skipping image {i} due to insufficient good matches.")
                continue

            # Points for homography calculation
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)           # Homography calculation

            if H is None:                                  # Affine transform if homography fails
                print(f"Homography failed between images {i - 1} and {i}. Using affine transform as fallback.")
                H, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
                H = np.vstack([H, [0, 0, 1]])  # Convert affine to homography format for consistency

            # Accumulating homographies and center transformation
            cumulative_H = cumulative_H @ H
            homography_matrix_list.append(cumulative_H)
            shift_matrix = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])
            centered_H = shift_matrix @ cumulative_H

            warped_image = cv2.warpPerspective(images[i], centered_H, (canvas_width, canvas_height))

            # Mask to blend smoothly
            mask = (warped_image.sum(axis=2) > 0).astype(np.uint8) * 255
            panorama_canvas = cv2.bitwise_and(panorama_canvas, panorama_canvas, mask=~mask)
            panorama_canvas = cv2.add(panorama_canvas, warped_image)

        # Crop black borders
        gray = cv2.cvtColor(panorama_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        final_panorama = panorama_canvas[y:y + h, x:x + w]

        return final_panorama, homography_matrix_list









