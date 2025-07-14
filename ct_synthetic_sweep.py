# CT SYNTHETIC SWEEP IMAGE GENERATION
# created 04022025

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#################### LOAD REFERENCE IMAGE ####################

# Ask user for folder name
folder_path = "/Users/emiz/Desktop/"  # <- EDIT HERE 
image_dim = input("\nType of Image (3D/4D): ")
data = input("\nInput Folder Name: ")

# input/output paths
input_path = os.path.join(folder_path, data, f"{data}.nii.gz")
output_3d_path = os.path.join(folder_path, data, f"{data}_3D.nii.gz")

# 3D image exists
if image_dim == "3D":
    img = sitk.ReadImage(input_path)

# 4D image, need to slice
else: 
    # already sliced
    if not os.path.exists(output_3d_path):
        # Read 4D image
        img4d = sitk.ReadImage(input_path)

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

#################### REFERENCE IMAGE ####################

def resample_expand_bounds(img, transform, expand_mm):
    """
    Expand resampling volume around original image bounds by a fixed physical amount (in mm).
    """
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    origin = np.array(img.GetOrigin())

    physical_extent = spacing * size
    center_phys = origin + direction @ (spacing * size / 2.0)

    # new physical extent
    new_extent = physical_extent + 2 * expand_mm
    new_size = np.ceil(new_extent / spacing).astype(int)

    # compute new origin so that center stays fixed
    new_origin = center_phys - direction @ (spacing * new_size / 2.0)

    # resample with expanded bounds
    ref_img = sitk.Resample(
        img,
        size=new_size.tolist(),
        outputSpacing=img.GetSpacing(),
        outputOrigin=new_origin.tolist(),
        outputDirection=img.GetDirection(),
        transform=transform,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0,
    )
    return ref_img

# check if reference image already exists
if os.path.exists(folder_path + data + "/" + data + "_ref.nii.gz"):
    ref_img = sitk.ReadImage(folder_path + data + "/" + data + "_ref.nii.gz")
else: 
    # generate reference img
    ref_img = resample_expand_bounds(img, ref_tform, expand_mm=50)
    sitk.WriteImage(ref_img, folder_path + data + "/" + data + "_ref.nii.gz")

# apply tform to get reference image/pose
# ref_img = sitk.Resample(
#     img, 
#     size=img.GetSize(),
#     outputSpacing=img.GetSpacing(),
#     outputOrigin=img.GetOrigin(),
#     outputDirection=img.GetDirection(),
#     transform=ref_tform,
#     defaultPixelValue=0,
#     interpolator=sitk.sitkLinear
# )

img_array = sitk.GetArrayFromImage(ref_img) # [Z, Y, X]
print(img_array.shape)

# coronal slice: 
coronal = int(input("\nA4CH Coronal Slice: "))
print("\n")

#################### SELECT LANDMARKS AND SAVE TO .TXT FILE ####################

landmarks = []
landmarks2D = []

def parse_landmark_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    landmarks_3d = []
    landmarks_2d = []
    mode = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("3D Landmarks"):
            mode = '3d'
            continue
        elif line.startswith("2D Landmarks"):
            mode = '2d'
            continue

        if mode == '3d':
            x, y, z = map(int, line.split(','))
            landmarks_3d.append((x, y, z))
        elif mode == '2d':
            x, y = map(float, line.split(','))
            landmarks_2d.append((x, y))

    return landmarks_3d, landmarks_2d

def select_landmark(event): 
    """
    Allows user to select necessary landmarks. 
    """
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata

    # update title
    if len(landmarks) == 0: 
        ax.set_title(f"Coronal Slice {coronal}: Landmark - MITRAL ANNULUS (right)")
    if len(landmarks) == 1:
        ax.set_title(f"Coronal Slice {coronal}: Landmark - TRICUSPID ANNULUS (left)")
    if len(landmarks) == 2: 
        ax.set_title(f"Coronal Slice {coronal}: Landmark - BASE")
    
    if x is not None and y is not None and len(landmarks) < 4:
        landmarks2D.append((x, y))
        y_index = coronal - 1
        z_index = int(y / spacing[2])  # convert mm → index in Z
        x_index = int(x / spacing[0])  # convert mm → index in X
        print(f"Selected landmark at (img_array X, Y, Z): ({x_index}, {y_index}, {z_index})")

        landmarks.append((x_index, y_index, z_index))
        ax.plot(x, y, 'ro')  # mark landmark with a red dot
        fig.canvas.draw()
    
    if len(landmarks) == 4:
        ax.set_title("4 Landmarks Selected — Closing...")
        fig.canvas.draw()
        fig.canvas.flush_events()  # ensure everything is rendered
        plt.close(fig)

coronal_slice = img_array[:, coronal-1, :]
Y, X = coronal_slice.shape

spacing = ref_img.GetSpacing()
x_extent = spacing[0] * img_array.shape[2]
y_extent = spacing[1] * img_array.shape[1]
z_extent = spacing[2] * img_array.shape[0]

# check if landmarks file already exists
if os.path.exists(os.path.join(folder_path, data, f"{data}_landmarks.txt")):
    # load landmarks
    landmarks, landmarks2D = parse_landmark_file(os.path.join(folder_path, data, f"{data}_landmarks.txt"))
    #print("Landmarks file loaded!")
else: 
    # create a figure to display the coronal slice
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(
        coronal_slice,
        cmap="gray",
        origin="lower",
        extent=[0, x_extent, 0, z_extent],
        #aspect='auto'
    )
    ax.set_title(f"Coronal Slice {coronal}: Landmark - APEX")
    ax.axis("off")

    # connect the on_click function to the figure
    fig.canvas.mpl_connect('button_press_event', select_landmark)
    plt.show()

    landmark_file = os.path.join(folder_path, data, f"{data}_landmarks.txt")

    with open(landmark_file, 'w') as f:
        f.write("3D Landmarks (X, Y, Z):\n")
        for pt in landmarks:
            f.write(f"{pt[0]}, {pt[1]}, {pt[2]}\n")
        
        f.write("\n2D Landmarks (X, Y) in coronal slice:\n")
        for pt in landmarks2D:
            f.write(f"{pt[0]:.2f}, {pt[1]:.2f}\n")

    print(f"\nSaved landmark coordinates to {landmark_file}")


#################### CROP IMAGE ####################

def crop_cone(shape, landmarks2D, angle_span=4*np.pi/9):
    """
    Create a cone-shaped mask centered around the vector from apex to base.
    Inputs: 
        - shape: (height, width) of the image
        - landmarks2D: list of 2D landmarks [(x, y), ...]
            landmarks2D[0] = apex
            landmarks2D[3] = base
        - angle_span: width of the cone in radians (default: 90 degrees)
    """
    height, width = shape
    dx, _, dz = spacing

    # Apex in mm
    apex = np.array(landmarks2D[0])
    base = np.array(landmarks2D[3])

    # Grid in physical coordinates
    x_vals = np.arange(width) * dx
    z_vals = np.arange(height) * dz
    xx, zz = np.meshgrid(x_vals, z_vals, indexing='xy')  # [Z, X] shape

    # Distance and angle from apex to all pixels
    dxs = xx - apex[0]
    dzs = zz - apex[1]
    dist = np.sqrt(dxs**2 + dzs**2)
    theta = (np.arctan2(dzs, dxs) + 2 * np.pi) % (2 * np.pi)

    # Symmetric cone centered at downward direction (270° = 3π/2)
    angle_center = 3 * np.pi / 2  # pointing down in image (Z increasing)
    half_span = angle_span / 2
    angle_min = (angle_center - half_span + 2 * np.pi) % (2 * np.pi)
    angle_max = (angle_center + half_span + 2 * np.pi) % (2 * np.pi)

    # Radius = distance from apex to base
    radius = np.linalg.norm(base - apex)

    # Mask construction
    if angle_min < angle_max:
        mask = (dist <= radius) & (theta >= angle_min) & (theta <= angle_max)
    else:
        mask = (dist <= radius) & ((theta >= angle_min) | (theta <= angle_max))

    return mask, radius

# generate quarter cone mask (adjust angle range if needed)
cone_mask, radius = crop_cone(coronal_slice.shape, landmarks2D)

# apply to image
masked_slice = np.where(cone_mask, coronal_slice, 0)

# plt.imshow(masked_slice,
#     cmap="gray",
#     origin="lower",
#     extent=[0, x_extent, 0, z_extent],)
# plt.title("Cone Mask Applied")
# plt.show()

def crop_image(image_slice, cone_mask):
    """
    Crops a 2D image slice using the bounding box of a boolean cone mask.
    
    Args:
        image_slice (2D np.ndarray): The original image (e.g., coronal_slice)
        cone_mask (2D np.ndarray): Boolean array with same shape as image_slice

    Returns:
        cropped_image (2D np.ndarray): The cropped slice
        crop_bounds (tuple): (z_min, z_max, x_min, x_max)
    """
    if not np.any(cone_mask):
        raise ValueError("Cone mask is empty.")

    # Find bounding box
    z_indices, x_indices = np.where(cone_mask)
    z_min, z_max = z_indices.min(), z_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Crop both image and mask
    cropped_img = image_slice[z_min:z_max+1, x_min:x_max+1]
    cropped_mask = cone_mask[z_min:z_max+1, x_min:x_max+1]

    # Determine vmin from inside the cone only
    inside_values = cropped_img[cropped_mask]
    vmin = np.min(inside_values)

    # Set values outside the cone to vmin
    cropped_masked_img = np.where(cropped_mask, cropped_img, vmin)

    return cropped_masked_img, (z_min, z_max, x_min, x_max)

cropped_slice, bounds = crop_image(masked_slice, cone_mask)

# Unpack bounds
z_min, z_max, x_min, x_max = bounds

# Compute extent in mm to preserve physical scaling
extent = [
    x_min * spacing[0], (x_max + 1) * spacing[0],  # X range in mm
    z_min * spacing[2], (z_max + 1) * spacing[2],  # Z range in mm
]

# Plot cropped slice with correct proportions
plt.imshow(cropped_slice, cmap="gray", origin="lower", extent=extent)
plt.gca().set_aspect('equal')  # Optional: ensures 1:1 aspect in mm
plt.axis("off")
plt.title("Cropped Cone Region (Scaled)")
plt.show()

#################### GENERATE IMAGE SWEEPS ####################

def apply_tform(ref_img, center_phys, axis, angle_deg):
    """
    Applies a VersorRigid3DTransform around the physical-space center and axis.
    The axis is initially defined in voxel-space (e.g., [0,0,1] for Z),
    and then converted to physical-space using the direction matrix.
    """
    # Convert voxel-space axis to physical-space axis
    axis = np.array(axis, dtype=float)
    direction_matrix = np.array(ref_img.GetDirection()).reshape(3, 3)
    axis_phys = direction_matrix @ axis  # now a physical-space axis
    axis_phys /= np.linalg.norm(axis_phys)

    angle_rad = np.deg2rad(angle_deg)

    # Define the transform
    transform = sitk.VersorRigid3DTransform()
    transform.SetCenter(center_phys)
    transform.SetRotation(axis_phys.tolist(), angle_rad)

    # Apply rotation and preserve all original dimensions and spacing
    rotated_img = sitk.Resample(
        ref_img,
        ref_img,  # match size, spacing, origin, direction
        transform,
        sitk.sitkLinear,
        0.0,
        ref_img.GetPixelID()
    )

    return rotated_img


########## GENERATE AXIS-BASE SWEEPS ##########

def save_rotated_slices(ref_img, cone_mask, angles, rot, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    
    for _, angle in enumerate(angles):

        #print(angle) 
        #apply transform to image at the current angle
        if rot == 'ab':
            rotated_img = apply_tform(ref_img, center_phys, axis=(0, 0, 1), angle_deg=-angle)
        if rot == 'mt': 
            rotated_img = apply_tform(ref_img, center_phys, axis=(1, 0, 0), angle_deg=angle)

        rotated_array = sitk.GetArrayFromImage(rotated_img)
        coronal_slice = rotated_array[:, coronal - 1, :]
        masked_slice = np.where(cone_mask, coronal_slice, 0)
        cropped_slice, _ = crop_image(masked_slice, cone_mask)

        # Display the image
        plt.imshow(
            cropped_slice,
            cmap="gray",
            origin="lower",
            extent=extent
        )
        # plt.show()
        plt.gca().set_aspect('equal')
        plt.axis("off")  # turn off axes

        # Save the figure
        file_path = os.path.join(save_dir, f"rotated_{angle:+03}.png")
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

        # Close the figure to free memory
        plt.close()

    print(f"Saved {len(angles)} images to '{save_dir}'")

center_vox = tuple(int(i) for i in np.array(landmarks[0]))
center_phys = ref_img.TransformIndexToPhysicalPoint(center_vox)

# generate apex-base sweeps -90 to +90 degrees
angles = range(-90, 91, 5)
output_path = folder_path + data + "/apex-base-rotation"
save_rotated_slices(ref_img, cone_mask, angles, 'ab', output_path)

# generate mitral-tricuspid sweeps -15 to +15 degrees
angles = range(-15, 16, 1)
output_path = folder_path + data + "/mitral-tricuspid-rotation"
save_rotated_slices(ref_img, cone_mask, angles, 'mt', output_path)