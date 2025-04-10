# CT SYNTHETIC SWEEP IMAGE GENERATION
# created 04022025

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support

########## LOAD REFERENCE IMAGE ##########
path = "/Users/emiz/Desktop/img3d_CT_bav75_preop"
img = sitk.ReadImage(path + ".nii.gz")

# load tform
ref_tform = sitk.ReadTransform(path + "_tform.txt")

# apply tform to get reference image/pose
ref_img = sitk.Resample(img, transform=ref_tform)
img_array = sitk.GetArrayFromImage(ref_img)

# save as new nifti image
sitk.WriteImage(ref_img, path + "_ref.nii.gz")

########## GENERATE IMAGE SWEEPS ##########

cor = [0, 0, 0]
tform = sitk.Euler3DTransform()
tform.SetCenter(cor)
tform.SetRotation(0, 0, np.deg2rad(15))

resampled_img = sitk.Resample(ref_img, ref_img, tform, sitk.sitkLinear, 0.0, ref_img.GetPixelID())
sitk.WriteImage(resampled_img, path + "_test.nii.gz")

########## CROP IMAGE ##########
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

# ######### TESTING ##########
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
plt.show()


# def show_ct_3d_vtk(img_array, img_array_crop):
#     # Convert the numpy arrays to VTK format
#     def create_vtk_volume(img_array, color='white', opacity='sigmoid'):
#         data = vtk.vtkImageData()
#         data.SetDimensions(img_array.shape[2], img_array.shape[1], img_array.shape[0])

#         # Flatten the image array and convert it to a VTK array
#         flat_array = img_array.flatten(order="F")  # Flatten in Fortran order
#         vtk_array = vtk.util.numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=vtk.VTK_FLOAT)

#         # Set the point data for the VTK image
#         data.GetPointData().SetScalars(vtk_array)

#         # Set the spacing (for example, 1x1x1, adjust as needed)
#         data.SetSpacing(1.0, 1.0, 1.0)

#         # Create a volume mapper and volume
#         volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
#         volume_mapper.SetInputData(data)

#         # Set up the volume properties (e.g., color and opacity)
#         volume_property = vtk.vtkVolumeProperty()

#         # Color transfer function
#         color_func = vtk.vtkColorTransferFunction()
#         if color == 'red':
#             color_func.AddRGBPoint(0, 1.0, 0.0, 0.0)  # Red for low values
#             color_func.AddRGBPoint(255, 1.0, 0.0, 0.0)  # Red for high values
#         else:
#             color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black for low values
#             color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)  # White for high values

#         # Opacity transfer function
#         opacity_func = vtk.vtkPiecewiseFunction()
#         opacity_func.AddPoint(0, 0.0)  # Transparent for low values
#         opacity_func.AddPoint(255, 1.0)  # Opaque for high values

#         volume_property.SetColor(color_func)
#         volume_property.SetScalarOpacity(opacity_func)

#         # Create the volume
#         volume = vtk.vtkVolume()
#         volume.SetMapper(volume_mapper)
#         volume.SetProperty(volume_property)
#         return volume

#     # Create two volumes with different properties
#     volume1 = create_vtk_volume(img_array, color='white', opacity='sigmoid')  # Transparent for original
#     volume2 = create_vtk_volume(img_array_crop, color='red', opacity='sigmoid')  # Red for cropped

#     # Set up the renderer, render window, and interactor
#     renderer = vtk.vtkRenderer()
#     render_window = vtk.vtkRenderWindow()
#     render_window.AddRenderer(renderer)

#     render_window_interactor = vtk.vtkRenderWindowInteractor()
#     render_window_interactor.SetRenderWindow(render_window)

#     # Add volumes to renderer
#     renderer.AddVolume(volume1)
#     renderer.AddVolume(volume2)

#     # Set background color
#     renderer.SetBackground(0.1, 0.2, 0.4)

#     # Start rendering
#     render_window.Render()
#     render_window_interactor.Start()

# # Assuming img_array and img_array_crop are your loaded CT scan volumes
# show_ct_3d_vtk(img_array, img_array_crop)