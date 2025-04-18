# CT SYNTHETIC SWEEP IMAGE GENERATION
# created 04022025

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#################### LOAD REFERENCE IMAGE ####################

path = "/Users/emiz/Desktop/img3d_CT_bav75_preop"
img = sitk.ReadImage(path + ".nii.gz")

# load tform
ref_tform = sitk.ReadTransform(path + "_tform.txt")

# apply tform to get reference image/pose
ref_img = sitk.Resample(img, transform=ref_tform)
img_array = sitk.GetArrayFromImage(ref_img)

# save as new nifti image
sitk.WriteImage(ref_img, path + "_ref.nii.gz")

# coronal slice: 
coronal = 282

#################### SELECT LANDMARKS ####################

landmarks = []
landmarks2D = []

def select_landmark(event): 
    """
    Allows user to select necessary landmarks. 
    """
    global coronal

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

# Generate quarter cone mask (adjust angle range if needed)
cone_mask, radius = crop_cone(coronal_slice.shape, landmarks2D)

# Apply to image
masked_slice = np.where(cone_mask, coronal_slice, 0)

# Visualize
plt.figure(figsize=(8, 8))
plt.imshow(masked_slice, cmap='gray')
plt.title("Quarter Cone Crop from Apex to Base")
plt.axis("off")
plt.show()

#################### GENERATE IMAGE SWEEPS ####################

def apply_tform(ref_img, landmarks, angle):
    """
    Applies a rotation to ref_img around the axis defined by apex → base.
    """
    angle_radians = np.deg2rad(angle)
    
    # Convert apex and base to numpy arrays
    apex = np.array(landmarks[0], dtype=float)
    base = np.array(landmarks[3], dtype=float)

    # Convert to physical space
    apex_phys = np.array(ref_img.TransformIndexToPhysicalPoint([int(x) for x in apex[::-1]]))
    base_phys = np.array(ref_img.TransformIndexToPhysicalPoint([int(x) for x in base[::-1]]))

    # Axis of rotation (unit vector)
    axis = base_phys - apex_phys
    axis /= np.linalg.norm(axis)

    # Set up versor transform
    versor = sitk.VersorRigid3DTransform()
    versor.SetCenter(apex_phys.tolist())
    versor.SetRotation(axis.tolist(), angle_radians)
    versor.SetTranslation([0.0, 0.0, 0.0])

    # Apply transform
    resampled_img = sitk.Resample(
        ref_img, ref_img, versor, sitk.sitkLinear, 0.0, ref_img.GetPixelID()
    )

    return resampled_img

######### TESTING ##########

# List of angles to loop through
angles = range(0, 46, 5)  # From 0 to 45 degrees, with a step of 5

# Calculate the number of rows and columns for the grid
n_plots = len(angles)
n_cols = 5  # You can adjust this to change the number of columns
n_rows = (n_plots + n_cols - 1) // n_cols  # Round up to fill rows

# Create a figure with a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the angles
for i, angle in enumerate(angles):
    # Apply transform to image at the current angle
    rotated_img = apply_tform(ref_img, landmarks, -angle)  # Negative for counterclockwise rotation
    sitk.WriteImage(rotated_img, f"/Users/emiz/Desktop/trial_{angle}.nii.gz")
    
    # Convert to NumPy and extract coronal slice
    rotated_array = sitk.GetArrayFromImage(rotated_img)
    coronal_slice_ = np.fliplr(rotated_array[:, coronal-1, :])
    
    # Plot the rotated slice in the current subplot
    axes[i].imshow(coronal_slice_, cmap='gray')
    axes[i].set_title(f"Rotated {angle}°")
    axes[i].axis("off")

# Remove empty subplots if the number of angles is less than the grid size
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Display the plot
plt.tight_layout()
plt.show()


# angles = range(-10, 10, 1)  # from -90 to 90 degrees inclusive
# sweep_images = []  # to store resampled images
# coronal_slices = []  # to store 2D coronal slices at apex-base level
# cropped_slices = []

# # Determine coronal slice index based on apex and base landmarks
# # Since coronal slices are along the YZ plane, use the X index
# apex_z, apex_y, apex_x = landmarks[0]
# base_z, base_y, base_x = landmarks[3]
# coronal_index = int((apex_z + base_z) / 2)

# print(f"Using coronal slice Z index (slice axis): {coronal_index}")

# for angle in angles:
#     print(f"Applying rotation: {angle} degrees")

#     # Rotate image
#     rotated_img = apply_tform(ref_img, landmarks, angle)
#     sweep_images.append(rotated_img)

#     # Extract and flip coronal slice
#     array = sitk.GetArrayFromImage(rotated_img)
#     coronal_slice = np.fliplr(array[:, coronal_index, :])  # coronal = Z fixed, axis (Y,X)
#     coronal_slices.append(coronal_slice)

#     # Apply cone crop to 2D coronal slice
#     cone_mask, radius = crop_cone(coronal_slice.shape, landmarks2D)
#     masked_slice = np.where(cone_mask, coronal_slice, 0)

#     cropped_slices.append(masked_slice)

# # Plot all coronal slices
# n_cols = 6
# n_rows = int(np.ceil(len(angles) / n_cols))

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
# axes = axes.flatten()

# for i, (angle, slice_img) in enumerate(zip(angles, cropped_slices)):
#     axes[i].imshow(slice_img, cmap="gray")
#     axes[i].set_title(f"{angle}°")
#     axes[i].axis("off")

# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     axes[j].axis("off")

# plt.tight_layout()
# plt.suptitle("Coronal Slices at Apex-Base Plane (Sweep -10° to 10°)", y=1.02, fontsize=16)
# plt.show()

################### TESTING ####################

axial = 173
sagittal = 299
coronal = 282

# Extract slices from (Z, Y, X) format
axial_slice = np.fliplr(img_array[axial-1, :, :])  # Axial: Top-down (Z slice)
sagittal_slice = img_array[:, :, sagittal-1]  # Sagittal: Side view (X slice)
coronal_slice = np.fliplr(img_array[:, coronal-1, :])  # Coronal: Front view (Y slice)

# Create a figure to display all three views
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot axial view
axes[0].imshow(axial_slice, cmap="gray")
axes[0].set_title(f"Axial Slice {axial}")
axes[0].axis("off")

# Plot sagittal view
axes[1].imshow(sagittal_slice, cmap="gray")
axes[1].set_title(f"Sagittal Slice {sagittal}")
axes[1].axis("off")

# Plot coronal view
axes[2].imshow(coronal_slice, cmap="gray")
axes[2].set_title(f"Coronal Slice {coronal}")
axes[2].axis("off")

# Show all views
#plt.show()