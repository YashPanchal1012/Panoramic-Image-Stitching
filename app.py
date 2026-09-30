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
        None,
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
    
    