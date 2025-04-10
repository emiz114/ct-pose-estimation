# CT SYNTHETIC SWEEP IMAGE GENERATION
# created 04022025

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support

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

def select_landmark(event): 
    """
    
    """

    global coronal

    if event.inaxes == ax:
        x, y = event.xdata, event.ydata

    # update title
    if len(landmarks) == 0: 
        ax.set_title(f"Coronal Slice {coronal}: Landmark - (LEFT)")
    if len(landmarks) == 1:
        ax.set_title(f"Coronal Slice {coronal}: Landmark - (RIGHT)")
    if len(landmarks) == 2: 
        ax.set_title(f"Coronal Slice {coronal}: Landmark - (TOP)")
    
    if x is not None and y is not None and len(landmarks) < 4:
        z_index = coronal - 1  # The coronal slice corresponds to a fixed Z index
        y_index = int(y)  # Y index corresponds to the vertical direction in the 2D slice
        x_index = int(x)  # X index corresponds to the horizontal direction in the 2D slice
        print(f"Selected landmark at (img_array Z, Y, X): ({z_index}, {y_index}, {x_index})")
        landmarks.append((z_index, y_index, x_index))
        ax.plot(x, y, 'ro')  # Mark the landmark with a red dot
        fig.canvas.draw()
    
    if len(landmarks) == 4:
        ax.set_title("4 Landmarks Selected — Closing...")
        fig.canvas.draw()
        fig.canvas.flush_events()  # Ensure everything is rendered
        plt.close(fig)
        
coronal_slice = np.fliplr(img_array[:, coronal-1, :])  # Coronal: Front view (Y slice)

# Create a figure to display the coronal slice
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the coronal slice
ax.imshow(coronal_slice, cmap="gray")
ax.set_title(f"Coronal Slice {coronal}: Landmark - APEX")
ax.axis("off")

# Connect the on_click function to the figure
fig.canvas.mpl_connect('button_press_event', select_landmark)

# Show the figure and wait for user input (clicking on the coronal slice)
plt.show()

#################### GENERATE IMAGE SWEEPS ####################

cor = landmarks[0]
print(cor)
tform = sitk.Euler3DTransform()
tform.SetCenter(cor)
tform.SetRotation(0, 0, np.deg2rad(15))

resampled_img = sitk.Resample(ref_img, ref_img, tform, sitk.sitkLinear, 0.0, ref_img.GetPixelID())
sitk.WriteImage(resampled_img, path + "_test.nii.gz")

#################### CROP IMAGE ####################

def crop_cube(img_array):
    """
    Crops image into a cube based on the minimum slice dimension. 
    """
    z, y, x = img_array.shape
    min_dim = min(z, y, x)
    
    z_start = (z - min_dim) // 2
    y_start = (y - min_dim) // 2
    x_start = (x - min_dim) // 2

    return img_array[z_start:z_start+min_dim,
                     y_start:y_start+min_dim,
                     x_start:x_start+min_dim]

img_array_crop = crop_cube(img_array)

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