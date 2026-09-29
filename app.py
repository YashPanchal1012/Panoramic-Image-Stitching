import cv2

def load_image(image_path):
    """
    Load an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image.
    """
    cv2_image = cv2.imread(image_path)
    if cv2_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2_image


# Detect keypoints and compute descriptors using SIFT
def detect_and_compute_sift(image):
    """
    Detect keypoints and compute descriptors using SIFT.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the detected keypoints and their descriptors.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# keypoint visualization
def draw_keypoints(image, keypoints):
    """
    Draw keypoints on the image.

    Args:
        image (numpy.ndarray): The input image.
        keypoints (list): A list of detected keypoints.

    Returns:
        numpy.ndarray: The image with keypoints drawn.
    """
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
    """
    Match descriptors between two sets using FLANN-based matcher.

    Args:
        descriptors1 (numpy.ndarray): Descriptors from the first image.
        descriptors2 (numpy.ndarray): Descriptors from the second image.

    Returns:
        list: A list of matched keypoints.
    """
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

# matching visualization
def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    """
    Draw matches between two images.

    Args:
        image1 (numpy.ndarray): The first input image.
        keypoints1 (list): Keypoints from the first image.
        image2 (numpy.ndarray): The second input image.
        keypoints2 (list): Keypoints from the second image.
        matches (list): A list of matched keypoints.

    Returns:
        numpy.ndarray: The image with matches drawn.
    """
    output_image = cv2.UMat()
    cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        output_image,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return output_image.get()

if __name__ == "__main__":

    # SET 1 - Fixed center rotation

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

    # Display keypoints
    combined_output = cv2.hconcat(resized_images)
    cv2.imshow("Keypoints", combined_output)

    # Draw matches
    matches = match_descriptors(descriptors1, descriptors2)
    matched_image = draw_matches(image_1, keypoints1, image_2, keypoints2, matches)
    cv2.imshow("Matches", matched_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()