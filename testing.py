import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def rotate_same_dimensions(img, center_phys, axis, angle_degrees):
    
    # Normalize axis and convert to Versor
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_degrees)

    transform = sitk.VersorRigid3DTransform()
    transform.SetCenter(center_phys)
    transform.SetRotation(axis.tolist(), angle_rad)

    # Apply rotation, preserving all original dimensions and spacing
    rotated = sitk.Resample(
        img,
        img,  # match size, spacing, origin, direction
        transform,
        sitk.sitkLinear,
        0.0,
        img.GetPixelID()
    )
    
    return rotated

# Load image
img = sitk.ReadImage("/Users/emiz/Desktop/img3d_CT_tav087_preAV/img3d_CT_tav087_preAV_ref.nii.gz")

# Define voxel center (X, Y, Z) and convert to physical
center_voxel = (346, 300, 347)  # (X, Y, Z)
center_phys = img.TransformIndexToPhysicalPoint(center_voxel)
print(center_phys)

# Rotate
rotated = rotate_same_dimensions(img, center_phys, axis=(0, 0, 1), angle_degrees=40)
rotated_array = sitk.GetArrayFromImage(rotated)
print(rotated_array.shape)
coronal_slice = rotated_array[:, 299, :]

plt.imshow(coronal_slice, 
           cmap="gray",
           origin="lower")
plt.show()
plt.axis("off")

# Save
sitk.WriteImage(rotated, "/Users/emiz/Desktop/rotated_output_same_dims.nii.gz")




