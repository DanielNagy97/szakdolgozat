import cv2
import numpy as np

cropped_frame = cv2.imread("Nelly.png")

overlay_image = cv2.imread("transparent.png", -1)


# Split out the transparency mask from the colour info
overlay_img = overlay_image[:, :, :3]   # BRG planes

cv2.imwrite("overlay_im.png", overlay_img)

overlay_mask = overlay_image[:, :, 3:]  # alpha plane

# Acalculate the inverse mask
background_mask = np.subtract(255, overlay_mask)

# Turn the masks into three channel, so we can use them as weights
overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

cv2.imwrite("alpha_channel.png", overlay_mask)

cv2.imwrite("inverted_alpha_channel.png", background_mask)

# Create a masked out face image, and masked out overlay
# We convert the images to floating point in range 0.0 - 1.0
background_part =\
    np.multiply((np.multiply(cropped_frame,
                             (1 / 255.0))),
                (np.multiply(background_mask,
                             (1 / 255.0))))

cv2.imwrite("background_part.png", background_part*255)

overlay_part =\
    np.multiply((np.multiply(overlay_img,
                             (1 / 255.0))),
                (np.multiply(overlay_mask,
                             (1 / 255.0))))

cv2.imwrite("overlay_part.png", overlay_part*255)

# And finally just add them together
# and rescale it back to an 8bit integer image
blended = np.uint8(cv2.addWeighted(background_part, 255.0,
                                   overlay_part, 255.0, 0))

cv2.imwrite("blended_result.png", blended)
