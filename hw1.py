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

    for query_index, descriptor in enumerate(descriptors1):
        distances = np.sum((descriptors2 - descriptor) ** 2, axis=1)
        nearest_indices = np.argsort(distances)[:2]
        first_index, second_index = nearest_indices
        matches.append((
            cv2.DMatch(
                query_index,
                int(first_index),
                0,
                float(np.sqrt(distances[first_index])),
            ),
            cv2.DMatch(
                query_index,
                int(second_index),
                0,
                float(np.sqrt(distances[second_index])),
            ),
        ))

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
    print("Keypoints in Image 1:", len(keypoints1))
    print("Keypoints in Image 2:", len(keypoints2))

    ############################################################################
    # PART 2: Descriptor Matching
    ############################################################################

    # Draw matches
    nearest_matches = match_descriptors(descriptors1, descriptors2)
    candidate_matches = [first_match for first_match, _ in nearest_matches]
    matches = ratio_test(nearest_matches)
    
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
    
    def project(H, pts):           # matrix multiply, then homogenous divide
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
                    [ 3e-4, -2e-4,  1. ]]) # nonzero bottom row: real perspective
    
    src = np.array([[50., 40.], [600., 70.], [560., 430.], [80., 400.]])
    dst = project(H_true, src)          # (88.38, 61.07)  (503.43, 57.46) ...
    H = find_homography(src, dst)       # your HW1 code (DLT) goes here 
    print("Exact 4-point mean error:", f"{mean_err(H, src, dst):.2e}", "px")

    rng = np.random.default_rng(0)
    src = rng.uniform([0, 0], [640, 480], size=(20, 2))   # 20 points over the frame
    dest = project(H_true, src)

    H = find_homography(src, dest)
    print("Exact 20-point mean error:", f"{mean_err(H, src, dest):.2e}", "px")

    noisy = dest + rng.normal(0, 0.5, dest.shape)     # sigma = 0.5 px
    H = find_homography(src, noisy)

    print("Noisy fit error:", f"{mean_err(H, src, noisy):.2f}", "px")
    print("Noisy fit error against clean points:", f"{mean_err(H, src, dest):.2f}", "px")

    ############################################################################
    # PART 4: Basic Robust Estimation with RANSAC
    ############################################################################

    src_points = np.float64([keypoints1[m.queryIdx].pt for m in matches]) # type: ignore
    dst_points = np.float64([keypoints2[m.trainIdx].pt for m in matches]) # type: ignore

    def non_degenerate(points):
        a, b, c, d = points
        ab = b - a
        ac = c - a
        ad = d - a
        area_abc = ab[0] * ac[1] - ab[1] * ac[0]
        area_abd = ab[0] * ad[1] - ab[1] * ad[0]
        return abs(area_abc) > 1e-6 or abs(area_abd) > 1e-6

    def ransac(src, dst, threshold=5.0, iterations=1000):
        rng = np.random.default_rng(0)
        best_H = None
        best_mask = None
        best_count = 0

        for _ in range(iterations):
            sample_indices = rng.choice(len(src), 4, replace=False)
            sample_src = src[sample_indices]
            sample_dst = dst[sample_indices]

            if not non_degenerate(sample_src) or not non_degenerate(sample_dst):
                continue

            candidate_H = find_homography(sample_src, sample_dst)
            errors = np.linalg.norm(project(candidate_H, src) - dst, axis=1)
            
            inlier_mask = errors < threshold
            inlier_count = int(inlier_mask.sum())

            if inlier_count > best_count:
                best_H = candidate_H
                best_count = inlier_count
                best_mask = inlier_mask

        return best_H, best_mask

    H_before_refit, inlier_mask = ransac(src_points, dst_points)
    H_after_refit = find_homography(src_points[inlier_mask], dst_points[inlier_mask])

    before_error = np.linalg.norm(
        project(H_before_refit, src_points[inlier_mask])
        - dst_points[inlier_mask], axis=1
    ).mean()

    after_error = np.linalg.norm(
        project(H_after_refit, src_points[inlier_mask])
        - dst_points[inlier_mask], axis=1
    ).mean()

    print("Total matches:", len(matches))
    print("RANSAC inliers:", int(inlier_mask.sum()))
    print("Inlier ratio:", f"{inlier_mask.mean():.2f}")
    print("Mean inlier error before refit:", f"{before_error:.2f}", "px")
    print("Mean inlier error after refit:", f"{after_error:.2f}", "px")

    outlier_matches = [m for m, is_inlier in zip(matches, inlier_mask) if not is_inlier]
    inlier_matches = [m for m, is_inlier in zip(matches, inlier_mask) if is_inlier]

    ransac_image = cv2.drawMatches(
        image_1, keypoints1, image_2, keypoints2, tuple(outlier_matches), None, # type: ignore
        (0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    ) # type: ignore

    for match in inlier_matches:
        point1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
        point2 = tuple(
            np.round(keypoints2[match.trainIdx].pt).astype(int)
            + np.array([image_1.shape[1], 0])
        )
        cv2.line(ransac_image, point1, point2, (0, 255, 0), 1, cv2.LINE_AA)

    save_image(ransac_image, "outputs/Set1_ransac_matches.jpg")

    ############################################################################
    # PART 5: Warping and Compositing
    ############################################################################

    def warp_image(image, image_to_canvas, canvas_shape):
        canvas_height, canvas_width = canvas_shape
        rows, cols = np.indices((canvas_height, canvas_width), dtype=np.float64)

        canvas_points = np.column_stack([cols.ravel(), rows.ravel(), np.ones(rows.size)])

        source = (np.linalg.inv(image_to_canvas) @ canvas_points.T).T
        source = source[:, :2] / source[:, 2:3]
        source_x = source[:, 0].reshape(canvas_shape).astype(np.float32)
        source_y = source[:, 1].reshape(canvas_shape).astype(np.float32)

        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0)
            & (source_x <= image.shape[1] - 1)
            & (source_y >= 0)
            & (source_y <= image.shape[0] - 1)
        )

        warped = cv2.remap(
            image,
            source_x,
            source_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return warped, valid

    image_2_to_image_1 = np.linalg.inv(H_after_refit)

    corners_1 = np.array([
        [0, 0],
        [image_1.shape[1] - 1, 0],
        [image_1.shape[1] - 1, image_1.shape[0] - 1],
        [0, image_1.shape[0] - 1],
    ], dtype=np.float64)

    corners_2 = np.array([
        [0, 0],
        [image_2.shape[1] - 1, 0],
        [image_2.shape[1] - 1, image_2.shape[0] - 1],
        [0, image_2.shape[0] - 1],
    ], dtype=np.float64)

    transformed_corners = np.vstack([
        corners_1,
        project(image_2_to_image_1, corners_2),
    ])

    minimum = np.floor(transformed_corners.min(axis=0)).astype(int)
    maximum = np.ceil(transformed_corners.max(axis=0)).astype(int)

    offset = np.array([
        [1.0, 0.0, -minimum[0]],
        [0.0, 1.0, -minimum[1]],
        [0.0, 0.0, 1.0],
    ])

    canvas_shape = (
        int(maximum[1] - minimum[1] + 1),
        int(maximum[0] - minimum[0] + 1),
    )

    warped_1, valid_1 = warp_image(image_1, offset, canvas_shape)
    warped_2, valid_2 = warp_image(image_2, offset @ image_2_to_image_1, canvas_shape)

    panorama = np.zeros_like(warped_1)
    panorama[valid_1] = warped_1[valid_1]
    panorama[valid_2] = warped_2[valid_2]

    save_image(panorama, "outputs/Set1_panorama.jpg")
    print("Panorama canvas:", canvas_shape[1], "x", canvas_shape[0])
    print("Reference image: Set1_1.jpeg")

    def transform_corners(corners, image_to_canvas):
        homogeneous = np.column_stack([corners, np.ones(len(corners))])
        transformed = (image_to_canvas @ homogeneous.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    boundary_image = panorama.copy()
    boundary_1 = np.round(transform_corners(corners_1, offset)).astype(np.int32)
    boundary_2 = np.round(
        transform_corners(corners_2, offset @ image_2_to_image_1)
    ).astype(np.int32)

    cv2.polylines(boundary_image, [boundary_1], True, (255, 0, 0), 6)
    cv2.polylines(boundary_image, [boundary_2], True, (0, 255, 255), 6)

    save_image(boundary_image, "outputs/Set1_panorama_annotated.jpg")

    # Synthetic warp checks with identifiable landmarks.
    test_image = np.zeros((80, 100, 3), dtype=np.uint8)
    landmarks = np.array([[20, 20], [70, 25], [35, 60]], dtype=np.float64)

    for x, y in landmarks.astype(int):
        cv2.rectangle(test_image, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)

    def check_warp(name, transform, offset):
        image_to_canvas = offset @ transform
        warped, valid = warp_image(test_image, image_to_canvas, (140, 180))
        expected = project(image_to_canvas, landmarks)

        for x, y in np.round(expected).astype(int):
            patch = warped[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            
            if not valid[y, x] or patch.max() < 200:
                raise AssertionError(f"{name} landmark check failed at {(x, y)}")
            
        save_image(warped, f"outputs/synthetic_warp_{name}.png")

        print(f"Synthetic {name} transform:\n", transform)
        print(f"Synthetic {name} landmarks verified at:", expected.astype(int).tolist())

    test_offset = np.array([
        [1.0, 0.0, 30.0],
        [0.0, 1.0, 25.0],
        [0.0, 0.0, 1.0],
    ])

    check_warp(
        "translation",
        np.array([[1.0, 0.0, -20.0], [0.0, 1.0, -15.0], [0.0, 0.0, 1.0]]),
        test_offset,
    )
    check_warp(
        "scale",
        np.array([[1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 1.0]]),
        test_offset,
    )

    save_image(test_image, "outputs/synthetic_warp_input.png")

    

    