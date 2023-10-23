import scipy.io
mat = scipy.io.loadmat('./datasets/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')

# Replace 'imgidx' with the specific image index you're interested in
imgidx = 1  # For example

# Access the 'act_name' field for the specified image
act_name = mat['RELEASE']['act']
# Now, the variable 'act_name' contains the activity name for the specified image.
print(f'Activity Name for Image {imgidx}: {act_name}')
