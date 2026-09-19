import streamlit as st
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import nibabel as nib
import numpy as np
from stpyvista import stpyvista
import pyvista as pv
import tempfile
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import segmentation_models_3D as sm
import os

# Function to load NIfTI file from uploaded file
def load_nifti(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name  # Get the path of the saved temp file
    return nib.load(temp_file_path).get_fdata()

# Function to create a smoothed VTK volume for the brain
def create_brain_mesh(data, threshold_value):
    dims = data.shape
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(dims[0], dims[1], dims[2])
    vtk_data.SetSpacing(1, 1, 1)
    # Convert NumPy array to VTK array
    vtk_array = numpy_to_vtk(num_array=data.flatten(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("ImageScalars")
    vtk_data.GetPointData().SetScalars(vtk_array)
    # Apply Gaussian smoothing (light smoothing to preserve shape)
    gaussian_filter = vtk.vtkImageGaussianSmooth()
    gaussian_filter.SetInputData(vtk_data)
    gaussian_filter.SetStandardDeviations(1.0, 1.0, 1.0)  # Light smoothing
    gaussian_filter.SetRadiusFactors(1.0, 1.0, 1.0)
    gaussian_filter.Update()
    # Extract brain surface using Marching Cubes
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(gaussian_filter.GetOutputPort())
    marching_cubes.SetValue(0, threshold_value)  # Use user-defined threshold
    marching_cubes.ComputeNormalsOn()  # Ensure normals are computed for smooth shading
    marching_cubes.Update()
    # Apply light smoothing to the extracted surface
    smoother = vtk.vtkWindowedSincPolyDataFilter()  # Better for preserving shape
    smoother.SetInputConnection(marching_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(10)  # Fewer iterations to preserve shape
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.1)  # Higher values preserve more details
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())  # Convert to PyVista format

# Function to create a mesh for the mask
def create_mask_mesh(data, value, threshold_value):
    # Create a binary mask for the specific value
    binary_mask = (data == value).astype(np.float32)
    dims = binary_mask.shape
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(dims[0], dims[1], dims[2])
    vtk_data.SetSpacing(1, 1, 1)
    # Convert NumPy array to VTK array
    vtk_array = numpy_to_vtk(num_array=binary_mask.flatten(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("ImageScalars")
    vtk_data.GetPointData().SetScalars(vtk_array)
    # Extract surface using Marching Cubes
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_data)
    marching_cubes.SetValue(0, threshold_value)  # Use user-defined threshold
    marching_cubes.ComputeNormalsOn()  # Ensure normals are computed for smooth shading
    marching_cubes.Update()
    # Apply light smoothing to the extracted surface
    smoother = vtk.vtkWindowedSincPolyDataFilter()  # Better for preserving shape
    smoother.SetInputConnection(marching_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(10)  # Fewer iterations to preserve shape
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetPassBand(0.1)  # Higher values preserve more details
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())  # Convert to PyVista format

# Function to visualize a 2D slice
def visualize_slice(data, axis, slice_index):
    """
    Visualize a 2D slice along the specified axis.
    :param data: 3D NumPy array
    :param axis: Axis to slice along ('x', 'y', or 'z')
    :param slice_index: Index of the slice
    """
    if axis == "x":
        slice_data = data[slice_index, :, :]
    elif axis == "y":
        slice_data = data[:, slice_index, :]
    elif axis == "z":
        slice_data = data[:, :, slice_index]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slice_data, cmap="gray", origin="lower")
    ax.axis("off")
    st.pyplot(fig)

# Main function
def main():
    st.title("3D Brain MRI and Mask Visualizer")
    st.sidebar.header("Upload your files")

    # File uploaders for brain MRI modalities (FLAIR, T1CE, T2) and mask
    flair_file = st.sidebar.file_uploader("Upload FLAIR Scan (.nii)", type=["nii"])
    t1ce_file = st.sidebar.file_uploader("Upload T1CE Scan (.nii)", type=["nii"])
    t2_file = st.sidebar.file_uploader("Upload T2 Scan (.nii)", type=["nii"])
    #mask_file = st.sidebar.file_uploader("Upload Mask (.nii)", type=["nii"])

    if flair_file and t1ce_file and t2_file: #and mask_file:
        st.write("### 3D Visualization")
        # Load brain and mask data
        flair_data = load_nifti(flair_file)
        t1ce_data = load_nifti(t1ce_file)
        t2_data = load_nifti(t2_file)
        start,end = 34,226
        fla_cla = cv2.normalize(flair_data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        t1ce_cla = cv2.normalize(t1ce_data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        t2_cla = cv2.normalize(t2_data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        combined_arr_cla = np.stack([fla_cla, t1ce_cla, t2_cla], axis=3)
        combined_arr_cla = combined_arr_cla[start:end, start:end, 13:141, :]
        mask_file = np.zeros((240,240,155))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(9,9))
        n = 192
        for l in range(combined_arr_cla.shape[2]):
            for j in range(0, combined_arr_cla[:, :, l, :].shape[0], n):
                for k in range(0, combined_arr_cla[:, :, l, :].shape[1], n):
                    for m in range(combined_arr_cla[:, :, l, :].shape[2]):
                        combined_arr_cla[:, :, l, m] = clahe.apply(combined_arr_cla[:, :, l, m])


        def dice_loss(y_true, y_pred, smooth=1e-6):
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

        def dice_bce_loss(y_true, y_pred, alpha=0.5):
            bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
            dice = dice_loss(y_true, y_pred)
            return alpha * dice + (1 - alpha) * bce

        dice_loss = sm.losses.DiceLoss(class_weights=[0.25, 0.25, 0.25, 0.25])
        focal = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1.0 * focal)

        default_model_dir = os.path.join(os.path.dirname(__file__), 'Attention_final_model_192_clahe_1024')
        model_path = os.path.join(default_model_dir, 'best_model1.h5')
        if not os.path.exists(model_path):
            model_path = 'best_model1.h5'

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'IOUScore': sm.metrics.IOUScore(),
                'FScore': sm.metrics.FScore()
            },
            compile=False
        )

        for i in range(combined_arr_cla.shape[2]):
            mask_file[start:end,start:end,i+13] = model.predict(np.expand_dims(combined_arr_cla[:, :, i, :],axis=0)).reshape((192,192,4)).argmax(axis=2)


        mask_data = mask_file

        # Add slider for brain threshold
        brain_threshold = st.sidebar.slider(
            "Brain Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        st.sidebar.write(f"Brain Threshold: {brain_threshold}")

        # Add slider for mask threshold
        mask_threshold = st.sidebar.slider(
            "Mask Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        st.sidebar.write(f"Mask Threshold: {mask_threshold}")

        # Create brain meshes for each modality
        flair_mesh = create_brain_mesh(flair_data, brain_threshold)
        t1ce_mesh = create_brain_mesh(t1ce_data, brain_threshold)
        t2_mesh = create_brain_mesh(t2_data, brain_threshold)

        # Initialize PyVista plotter
        plotter = pv.Plotter(window_size=[800, 600])
        plotter.set_background('black')  # Set background color

        # Sidebar visualization controls
        st.sidebar.subheader("Visualization Controls")

        # Slider for brain opacity
        brain_opacity = st.sidebar.slider("Brain Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

        # Button to reset camera
        if st.sidebar.button("Reset Camera"):
            plotter.reset_camera()  # Reset the camera to fit the entire scene

        # Add brain meshes with adjustable opacity
        st.sidebar.subheader("MRI Modalities")
        show_flair = st.sidebar.checkbox("Show FLAIR", value=True)
        show_t1ce = st.sidebar.checkbox("Show T1CE", value=True)
        show_t2 = st.sidebar.checkbox("Show T2", value=True)

        if show_flair:
            plotter.add_mesh(flair_mesh, opacity=brain_opacity, color="cyan", name="FLAIR")
        if show_t1ce:
            plotter.add_mesh(t1ce_mesh, opacity=brain_opacity, color="magenta", name="T1CE")
        if show_t2:
            plotter.add_mesh(t2_mesh, opacity=brain_opacity, color="yellow", name="T2")

        # Handle mask visualization
        unique_mask_values = np.unique(mask_data[mask_data > 0])  # Get unique non-zero mask values
        colormap = ["red", "blue", "green", "yellow", "purple", "orange"]  # Define a colormap
        mask_properties = {}  # Dictionary to store mask meshes and properties

        # Add checkboxes and sliders for each mask category
        st.sidebar.subheader("Mask Categories")
        for i, value in enumerate(unique_mask_values):
            mask_mesh = create_mask_mesh(mask_data, value, mask_threshold)  # Create mesh for the specific mask value
            # Assign a color from the colormap
            color = colormap[i % len(colormap)]
            # Add checkbox for visibility
            visible = st.sidebar.checkbox(f"Show Mask Category {value}", value=True)
            # Add slider for opacity
            opacity = st.sidebar.slider(f"Opacity for Mask Category {value}", min_value=0.1, max_value=1.0, value=0.9,
                                        step=0.1)
            # Store mask properties
            mask_properties[value] = {
                "mesh": mask_mesh,
                "color": color,
                "visible": visible,
                "opacity": opacity
            }

        # Add mask regions to the plotter based on properties
        for value, props in mask_properties.items():
            if props["visible"]:
                plotter.add_mesh(props["mesh"], opacity=props["opacity"], color=props["color"], name=f"Mask_{value}")

        # Automatically adjust the camera to focus on the brain
        plotter.camera_position = 'xy'
        plotter.reset_camera()

        # Show the 3D plot in Streamlit using stpyvista
        stpyvista(plotter)

        # Get dimensions of the data
        dims = flair_data.shape
        x_dim, y_dim, z_dim = dims

        # Add sliders for 2D slice visualization
        st.subheader("2D Slice Visualization")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_slice = st.slider("X-Axis Slice", min_value=0, max_value=x_dim - 1, value=x_dim // 2)
            visualize_slice(flair_data, "x", x_slice)
        with col2:
            y_slice = st.slider("Y-Axis Slice", min_value=0, max_value=y_dim - 1, value=y_dim // 2)
            visualize_slice(flair_data, "y", y_slice)
        with col3:
            z_slice = st.slider("Z-Axis Slice", min_value=0, max_value=z_dim - 1, value=z_dim // 2)
            visualize_slice(flair_data, "z", z_slice)

        # Optional: Add mask visualization
        st.subheader("Mask Slice Visualization")
        col1_mask, col2_mask, col3_mask = st.columns(3)
        with col1_mask:
            visualize_slice(mask_data, "x", x_slice)
        with col2_mask:
            visualize_slice(mask_data, "y", y_slice)
        with col3_mask:
            visualize_slice(mask_data, "z", z_slice)

if __name__ == "__main__":
    main()