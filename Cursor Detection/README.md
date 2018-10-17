# Cursor Detection

You may find three sub-folders i.e. Normal Cursor, Black Cursor and Hand Cursor. Each folder addresses the implementation of detecting that particular cursor on the sample images.


Proposed Methodology

Step 1: Storing the set of images with similar cursor into a list 

Step 2: Traversing the list and reading each and every image at a time

Step 3: Reading the template in every iteration, templates named as ‘template_1_hand.png’ for hand cursor, ‘template_1_blk.png’ for black point cursor and ‘template_1_white.png’ for normal cursor

Step 4: Resizing the template as per required

Step 5: Applying Gaussian Filter over the image as blur = gaussian_filter(img,0) where 0 is the sigma value. Library used is from ‘scipy.ndimage.filters import gaussian_filter’

Step 6: Calculating Laplacian of Gaussian filtered image and the template as:

lap_i = cv2.Laplacian(blur, cv2.CV_32F)

lap_t = cv2.Laplacian(temp, cv2.CV_32F)

Step 7: Matching template by passing lap_i and lap_t and also the method:

cv2.matchTemplate(lap_i, lap_t, cv2.TM_CCORR_NORMED)

Step 8: Setting up a relevant threshold, threshold varies according to the cursor that is to be detected. For eg. To detect black point cursor, threshold set is 0.44 and to detect hand cursor, threshold set is 0.45

Step 9: Plotting a rectangle where the cursor is detected
