import os
import cv2 as cv


def normalize():
    # Get the list of files in the 'silhouettes' directory
    frames = os.listdir(os.path.join(os.getcwd(), 'silhouettes'))

    # Iterate over each frame
    for frame in frames:
        # Read the image
        img = cv.imread(os.path.join(os.getcwd(), 'silhouettes', frame))

        # Extract top, end, start, and last rows/columns
        top_rows = img[:25, :]
        end_rows = img[-25:, :]
        start_cols = img[:, :10]
        last_cols = img[:, -35:]

        # Check conditions for normalization
        if 255 in img:
            if (255 not in top_rows or 255 in img[:, 10:36]) and 255 not in end_rows and 255 not in start_cols and (255 not in last_cols or 255 in img[:, 11:-36]):
                # Construct absolute path for saving normalized image
                output_path = os.path.join(os.getcwd(), 'normalize', frame)

                # Save normalized image
                cv.imwrite(output_path, img)
