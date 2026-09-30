import os

import cv2
import numpy as np

def save_image(image, filename):
    cv2.imwrite(filename, image)

def load_image(image_path):
    cv2_image = cv2.imread(image_path)
    if cv2_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2_image

# Detect keypoints and compute descriptors using SIFT
def detect_and_compute_sift(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# keypoint visualization
def draw_keypoints(image, keypoints):
    output_image = cv2.UMat(image)
    cv2.drawKeypoints(
        image,
        keypoints,
        output_image,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return output_image.get()

# descriptor matching
def match_descriptors(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None or len(descriptors2) < 2:
        return []

    descriptors1 = np.asarray(descriptors1, dtype=np.float32)
    descriptors2 = np.asarray(descriptors2, dtype=np.float32)
    matches = []

    # Process chunks to avoid allocating a full descriptor-distance tensor.
    for start in range(0, len(descriptors1), 128):
        end = min(start + 128, len(descriptors1))
        distances = np.sum(
            (descriptors1[start:end, None, :] - descriptors2[None, :, :]) ** 2,
            axis=2,
        )
        nearest = np.argpartition(distances, 1, axis=1)[:, :2]
        for local_index, pair in enumerate(nearest):
            first, second = pair[np.argsort(distances[local_index, pair])]
            first_distance = float(np.sqrt(distances[local_index, first]))
            second_distance = float(np.sqrt(distances[local_index, second]))
            first_match = cv2.DMatch(
                start + local_index,
                int(first),
                0,
                first_distance,
            )
            second_match = cv2.DMatch(
                start + local_index,
                int(second),
                0,
                second_distance,
            )
            matches.append((first_match, second_match))

    return matches


def ratio_test(matches, ratio_threshold=0.7):
    return [
        first_match
        for first_match, second_match in matches
        if first_match.distance < ratio_threshold * second_match.distance
    ]

# matching visualization
def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None, # type: ignore
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    ) # type: ignore

if __name__ == "__main__":

    ############################################################################
    # PART 1: Image Capture and Feature Extraction
    ############################################################################

    # Load images
    image_1 = load_image("Images/Set1_1.jpeg")
    image_2 = load_image("Images/Set1_2.jpeg")

    # Detect Keypoints
    output_images = []
    keypoints1, descriptors1 = detect_and_compute_sift(image_1)
    output_images.append(draw_keypoints(image_1, keypoints1))
    
    keypoints2, descriptors2 = detect_and_compute_sift(image_2)
    output_images.append(draw_keypoints(image_2, keypoints2))

    # Resize outputs to the same height and display them side by side.
    display_height = min(image.shape[0] for image in output_images)
    resized_images = [
        cv2.resize(
            image,
            (int(image.shape[1] * display_height / image.shape[0]), display_height),
        )
        for image in output_images
    ]

    os.makedirs("outputs", exist_ok=True)
    save_image(cv2.hconcat(resized_images), "outputs/Set1_keypoints.jpg")

    ############################################################################
    # PART 2: Descriptor Matching
    ############################################################################

    # Draw matches
    nearest_matches = match_descriptors(descriptors1, descriptors2)
    candidate_matches = [first_match for first_match, _ in nearest_matches]
    matches = ratio_test(nearest_matches)
    print("Keypoints in Image 1:", len(keypoints1))
    print("Keypoints in Image 2:", len(keypoints2))
    print("Candidate matches:", len(candidate_matches))
    print("Matches after ratio test:", len(matches))

    matched_image_before = draw_matches(
        image_1, keypoints1, image_2, keypoints2, candidate_matches
    )
    matched_image = draw_matches(image_1, keypoints1, image_2, keypoints2, matches)
    save_image(matched_image_before, "outputs/Set1_matches_before_ratio.jpg")
    save_image(matched_image, "outputs/Set1_matches_after_ratio.jpg")

    ############################################################################
    # PART 3: Homography Estimation
    ############################################################################
    
    def project(H, pts):                            # matrix multiply, then homogenous divide
        ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
        return ph[:, :2] / ph[:, 2:3]
    
    def mean_err(H, src, dst):
        return np.linalg.norm(project(H, src) - dst, axis=1).mean()

    def normalize_points(points):
        points = np.asarray(points, dtype=np.float64)
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        mean_distance = distances.mean()
        if mean_distance == 0:
            raise ValueError("Cannot normalize coincident points.")

        scale = np.sqrt(2.0) / mean_distance
        transform = np.array([
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ])
        homogeneous = np.column_stack([points, np.ones(len(points))])
        normalized = (transform @ homogeneous.T).T
        return normalized[:, :2], transform

    def find_homography(src, dst):
        if len(src) != len(dst) or len(src) < 4:
            raise ValueError("Homography estimation requires at least four pairs.")

        normalized_src, src_transform = normalize_points(src)
        normalized_dst, dst_transform = normalize_points(dst)

        # Solve DLT in normalized coordinates for numerical stability.
        A = []
        for (x, y), (u, v) in zip(normalized_src, normalized_dst):
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        normalized_H = Vt[-1].reshape(3, 3)

        H = np.linalg.inv(dst_transform) @ normalized_H @ src_transform
        return H / H[2, 2]
    
    H_true = np.array([[ 0.9,   0.1,  40.],
                    [-0.05,  1.1,  20.],
                    [ 3e-4, -2e-4,  1. ]])       # nonzero bottom row: real perspective
    
    src = np.array([[50., 40.], [600., 70.], [560., 430.], [80., 400.]])
    dst = project(H_true, src)                      # (88.38, 61.07)  (503.43, 57.46) ...
    H = find_homography(src, dst)                   # your HW1 code (DLT) goes here 
    print("Exact 4-point mean error:", mean_err(H, src, dst), "px")

    rng = np.random.default_rng(0)
    src = rng.uniform([0, 0], [640, 480], size=(20, 2))   # 20 points over the frame
    dest = project(H_true, src)

    H = find_homography(src, dest)
    print("Exact 20-point mean error:", mean_err(H, src, dest), "px")

    noisy = dest + rng.normal(0, 0.5, dest.shape)          # sigma = 0.5 px
    H = find_homography(src, noisy)

    print("Noisy fit error:", mean_err(H, src, noisy), "px")
    print("Noisy fit error against clean points:", mean_err(H, src, dest), "px")

    ############################################################################
    # PART 4: Basic Robust Estimation with RANSAC
    ############################################################################