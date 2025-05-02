# CT SYNTHETIC SWEEP IMAGE GENERATION
# created 04022025

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#################### LOAD REFERENCE IMAGE ####################

# Ask user for folder name
folder_path = "/Users/emiz/Desktop/"  # <- EDIT HERE 
data = input("\nInput Folder Name: ")

# Build full path to input file
input_4d_path = os.path.join(folder_path, data, f"{data}.nii.gz")
output_3d_path = os.path.join(folder_path, data, f"{data}_3D.nii.gz")

# Save the 3D image
if not os.path.exists(output_3d_path):
    # Read 4D image
    img4d = sitk.ReadImage(input_4d_path)

    # Check if it's really 4D
    if img4d.GetDimension() != 4:
        raise ValueError(f"Expected a 4D image, but got {img4d.GetDimension()}D")

    # Extract first timepoint (t=0)
    img = img4d[:, :, :, 0]
    sitk.WriteImage(img, output_3d_path)
    print(f"\nSaved first timepoint to: {output_3d_path}")
else:
    print(f"\nSkipped writing: {output_3d_path} already exists.")

img = sitk.ReadImage(output_3d_path)
# load tform
ref_tform = sitk.ReadTransform(folder_path + data + "/" + data + "_tform.txt")

def expand_bb(img, tform):
    dim = img.GetDimension()
    img_size = img.GetSize()
    img_spacing = img.GetSpacing()
    img_origin = img.GetOrigin()
    img_direction = np.array(img.GetDirection()).reshape((dim, dim))

    # get corner points
    corners = []
    for i in range(2**dim):
        idx = [(i >> d) & 1 for d in range(dim)]
        physical = img_origin + img_direction @ (np.array(idx) * img_spacing * np.array(img_size))
        corners.append(physical)
    corners = np.array(corners)

    # Transform corners
    transformed_corners = np.array([tform.TransformPoint(p.tolist()) for p in corners])

    # Find bounding box
    min_corner = transformed_corners.min(axis=0)
    max_corner = transformed_corners.max(axis=0)

    # Compute new size
    new_origin = min_corner
    new_size = np.ceil((max_corner - min_corner) / img_spacing).astype(int).tolist()

    return new_size, new_origin

new_size, new_origin = expand_bb(img, ref_tform)

# apply tform to get reference image/pose
ref_img = sitk.Resample(
    img, 
    size=new_size, 
    transform=ref_tform,
    outputSpacing=img.GetSpacing(),
    outputOrigin=img.GetOrigin(),
    outputDirection=img.GetDirection(),
    defaultPixelValue=0,
    interpolator=sitk.sitkLinear
)
img_array = sitk.GetArrayFromImage(ref_img)

# save as new nifti image
sitk.WriteImage(ref_img, folder_path + data + "/" + data + "_ref.nii.gz")

# coronal slice: 
coronal = int(input("\nA4CH Coronal Slice: "))
print("\n")

#################### SELECT LANDMARKS ####################

landmarks = []
landmarks2D = []

def select_landmark(event): 
    """
    Allows user to select necessary landmarks. 
    """
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata

    # update title
    if len(landmarks) == 0: 
        ax.set_title(f"Coronal Slice {coronal}: Landmark - MITRAL ANNULUS (left)")
    if len(landmarks) == 1:
        ax.set_title(f"Coronal Slice {coronal}: Landmark - TRICUSPID ANNULUS (right)")
    if len(landmarks) == 2: 
        ax.set_title(f"Coronal Slice {coronal}: Landmark - BASE")
    
    if x is not None and y is not None and len(landmarks) < 4:
        landmarks2D.append((x, y))
        z_index = coronal - 1  # coronal slice corresponds to a fixed Z index
        y_index = int(y)  # Y index corresponds to the vertical direction in the 2D slice
        x_index = int(x)  # X index corresponds to the horizontal direction in the 2D slice
        print(f"Selected landmark at (img_array Z, Y, X): ({z_index}, {y_index}, {x_index})")
        landmarks.append((z_index, y_index, x_index))
        ax.plot(x, y, 'ro')  # mark landmark with a red dot
        fig.canvas.draw()
    
    if len(landmarks) == 4:
        ax.set_title("4 Landmarks Selected — Closing...")
        fig.canvas.draw()
        fig.canvas.flush_events()  # ensure everything is rendered
        plt.close(fig)
        
coronal_slice = np.fliplr(img_array[:, coronal-1, :])

# create a figure to display the coronal slice
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(coronal_slice, cmap="gray")
ax.set_title(f"Coronal Slice {coronal}: Landmark - APEX")
ax.axis("off")

# connect the on_click function to the figure
fig.canvas.mpl_connect('button_press_event', select_landmark)
plt.show()

#################### SAVE LANDMARKS TO TXT ####################

landmark_file = os.path.join(folder_path, data, f"{data}_landmarks.txt")
with open(landmark_file, 'w') as f:
    f.write("3D Landmarks (Z, Y, X):\n")
    for pt in landmarks:
        f.write(f"{pt[0]}, {pt[1]}, {pt[2]}\n")
    
    f.write("\n2D Landmarks (X, Y) in coronal slice:\n")
    for pt in landmarks2D:
        f.write(f"{pt[0]:.2f}, {pt[1]:.2f}\n")

print(f"\nSaved landmark coordinates to {landmark_file}")

#################### CROP IMAGE ####################

def crop_cone(shape, landmarks2D, angle_span=np.pi/2):
    """
    Create a cone-shaped mask centered around the vector from apex to base.
    Inputs: 
        - shape: (height, width) of the image
        - landmarks2D: list of 2D landmarks [(x, y), ...]
            landmarks2D[0] = apex
            landmarks2D[3] = base
        - angle_span: width of the cone in radians (default: 90 degrees)
    """
    Y, X = shape

    # flip y-coordinates for image coordinates
    apex = np.array([landmarks2D[0][0], Y - landmarks2D[0][1] - 1])
    base = np.array([landmarks2D[3][0], Y - landmarks2D[3][1] - 1])

    # compute radius
    radius = np.linalg.norm(base - apex)

    # angle of direction from apex to base
    direction = base - apex
    angle_center = (np.arctan2(direction[1], direction[0]) + 2 * np.pi) % (2 * np.pi)

    # define cone angles centered at angle_center
    half_span = angle_span / 2
    angle_min = (angle_center - half_span + 2 * np.pi) % (2 * np.pi)
    angle_max = (angle_center + half_span + 2 * np.pi) % (2 * np.pi)

    # grid
    yy, xx = np.mgrid[0:Y, 0:X]
    yy = Y - yy - 1  # flip vertical axis

    dx = xx - apex[0]
    dy = yy - apex[1]

    dist = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)

    # handle angle wraparound (e.g. if the cone crosses 0 rad)
    if angle_min < angle_max:
        mask = (dist <= radius) & (theta >= angle_min) & (theta <= angle_max)
    else:
        mask = (dist <= radius) & ((theta >= angle_min) | (theta <= angle_max))
    
    return mask, radius

# generate quarter cone mask (adjust angle range if needed)
cone_mask, radius = crop_cone(coronal_slice.shape, landmarks2D)

# apply to image
masked_slice = np.where(cone_mask, coronal_slice, 0)

# visualize
# plt.figure(figsize=(8, 8))
# plt.imshow(masked_slice, cmap='gray')
# plt.title("Quarter Cone Crop from Apex to Base")
# plt.axis("off")
#plt.show()

#################### GENERATE IMAGE SWEEPS ####################

def apply_tform_ab(ref_img, landmarks, angle):
    """
    Applies a rotation to ref_img around the axis defined by apex -> base.
    """
    angle = np.deg2rad(angle)    
    # get apex landmark
    apex = np.array(landmarks[0], dtype=float)
    apex_phys = np.array(ref_img.TransformIndexToPhysicalPoint([int(x) for x in apex[::-1]]))

    # set up Euler 3D transform
    euler = sitk.Euler3DTransform()
    euler.SetCenter(apex_phys.tolist())

    # rotate about vertical axis (Z-axis)
    euler.SetRotation(
        0.0,            # rotation around x-axis (pitch)
        0.0,            # rotation around y-axis (roll)
        angle           # rotation around z-axis (yaw) — vertical
    )
    euler.SetTranslation([0.0, 0.0, 0.0])

    # apply transform
    resampled_img = sitk.Resample(
        ref_img, ref_img, euler, sitk.sitkLinear, 0.0, ref_img.GetPixelID()
    )

    return resampled_img

def apply_tform_mt(ref_img, landmarks, angle):
    """
    Applies a rotation to ref_img around the direction defined by mitral -> tricuspid annulus from apex. 
    """
    angle = np.deg2rad(angle)
    # get apex landmark
    apex = np.array(landmarks[0], dtype=float)
    apex_phys = np.array(ref_img.TransformIndexToPhysicalPoint([int(x) for x in apex[::-1]]))

    # set up Euler 3D transform
    euler = sitk.Euler3DTransform()
    euler.SetCenter(apex_phys.tolist())

    # rotate about horizontal axis (X-axis)
    euler.SetRotation(
        angle,         # rotation around x-axis (pitch) - horizontal
        0.0,           # rotation around y-axis (roll)
        0.0            # rotation around z-axis (yaw)
    )
    euler.SetTranslation([0.0, 0.0, 0.0])

    # apply transform
    resampled_img = sitk.Resample(
        ref_img, ref_img, euler, sitk.sitkLinear, 0.0, ref_img.GetPixelID()
    )

    return resampled_img

########## GENERATE AXIS-BASE SWEEPS ##########

def save_rotated_slices(ref_img, landmarks, cone_mask, angles, rot, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    for _, angle in enumerate(angles):

        # apply transform to image at the current angle
        if rot == 'ab':
            rotated_img = apply_tform_ab(ref_img, landmarks, -angle)
        if rot == 'mt': 
            rotated_img = apply_tform_mt(ref_img, landmarks, -angle)

        rotated_array = sitk.GetArrayFromImage(rotated_img)
        coronal_slice = np.fliplr(rotated_array[:, coronal - 1, :])
        masked_slice = np.where(cone_mask, coronal_slice, 0)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.flipud(masked_slice), cmap='gray', origin='lower')
        ax.set_title(f"Rotated {angle}°")
        ax.axis("off")

        # save the figure
        file_path = os.path.join(save_dir, f"rotated_{angle:+03}.png")
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"Saved {len(angles)} images to '{save_dir}'")

# generate apex-base sweeps -90 to +90 degrees
# angles = range(-90, 91, 5)
# output_path = folder_path + data + "/" + data + "/apex-base-rotation"
# save_rotated_slices(ref_img, landmarks, cone_mask, angles, 'ab', output_path)

# # generate mitral-tricuspid sweeps -15 to +15 degrees
# angles = range(-15, 16, 1)
# output_path = folder_path + data + "/" + data + "/mitral-tricuspid-rotation"
# save_rotated_slices(ref_img, landmarks, cone_mask, angles, 'mt', output_path)

########## VISUALIZATION FOR TESTING ##########

# list of angles to loop through (step of 5)
# angles = range(-90, 91, 5)

# # Calculate the number of rows and columns for the grid
# n_plots = len(angles)
# n_cols = 5  # You can adjust this to change the number of columns
# n_rows = (n_plots + n_cols - 1) // n_cols  # Round up to fill rows

# # Create a figure with a grid of subplots
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))

# # Flatten the axes array for easy iteration
# axes = axes.flatten()

# # Loop through the angles
# for i, angle in enumerate(angles):
#     # Apply transform to image at the current angle (negative for counterclockwise)
#     rotated_img = apply_tform_ab(ref_img, landmarks, -angle)
    
#     # Convert to NumPy array
#     rotated_array = sitk.GetArrayFromImage(rotated_img)
#     coronal_slice = np.fliplr(rotated_array[:, coronal-1, :])
#     # Apply cone mask/cropping
#     masked_slice = np.where(cone_mask, coronal_slice, 0)

#     # Display the coronal slice
#     axes[i].imshow(np.flipud(coronal_slice), cmap='gray', origin='lower')

#     # Plot the apex point
#     axes[i].plot(landmarks2D[0][0], coronal_slice.shape[0] - landmarks2D[0][1] -1, 'bo', markersize=5)

#     # Final touches
#     axes[i].set_title(f"Rotated {angle}°")
#     axes[i].axis("off")

# # Turn off any remaining unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis('off')

# plt.tight_layout()
# plt.show()