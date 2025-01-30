import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.efficientnet import decode_predictions
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

import warnings
warnings.filterwarnings("ignore")

# Set up the app
st.set_page_config(page_title="CNN Visualization App", layout="wide")
st.title("CNN Visualization App 📊")

# Sidebar for Model Selection
st.sidebar.header("Using Pretrained VGG16")
model_type = "VGG16"

# Load the pre-trained model
@st.cache_resource
def load_model(model_name):
    if model_name == "EfficientNetB0":
        return EfficientNetB0(weights='imagenet')
    else:
        return VGG16(weights='imagenet', include_top=True)

model = load_model(model_type)

# Image Upload Section
uploaded_file = st.file_uploader("Upload an Image 🖼️", type=["jpg", "jpeg", "png"])
class_index = None
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img_array = np.array(image.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)

    # Model-specific preprocessing
    img_array = preprocess_eff(img_array) if model_type == "EfficientNetB0" else preprocess_vgg(img_array)

    # Prediction Section
    st.subheader("🔍 Predictions")
    if st.button("Predict Image"):
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=1)[0][0]
        predicted_class = decoded_preds[1]
        class_index = np.argmax(preds)
        st.success(f"Predicted Class: **{predicted_class}**")
    
    # Saliency Map Implementation
    st.subheader("🧠 Saliency Map Visualization")

    def compute_saliency_map(image_array, model, class_index):
    # Convert the image to a tensor and watch it for gradients
        input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        input_tensor = tf.Variable(input_tensor)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            preds = model(input_tensor)
            loss = preds[0][class_index]

        # Compute gradients
        gradients = tape.gradient(loss, input_tensor)
        gradients = tf.reduce_max(tf.abs(gradients), axis=-1)[0]

        # Apply Gaussian smoothing to increase receptive field
        smoothed_gradients = gaussian_filter(gradients, sigma=5)

        # Normalize gradients with power normalization to boost visibility
        saliency_map = np.power(smoothed_gradients, 0.8)  # Control the power for better spread
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-9)

        return saliency_map

    # Generate Saliency Map Button
    if st.button("Generate Saliency Map"):
        class_index = np.argmax(model.predict(img_array))
        if class_index is not None:
            saliency_map = compute_saliency_map(img_array, model, class_index)

            # Bright green and blue colormap
            custom_cmap = LinearSegmentedColormap.from_list(
                "BrightBlueGreen", [(0, 0, 1), (0, 1, 0)]
            )

            # Display Saliency Map
            fig, ax = plt.subplots()
            ax.imshow(saliency_map, cmap=custom_cmap)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Please predict the image class first.")
    
    # Advanced Visualization
    st.subheader("Advanced Visualization 🔬")

    # Saliency Map Adjustments
    heatmap_intensity = st.slider("Heatmap Intensity", min_value=0.1, max_value=1.0, value=0.5)
    overlay_opacity = st.slider("Background Opacity", min_value=0.1, max_value=1.0, value=0.5)

    if st.button("Generate Enhanced Saliency Map"):
        # Get class index from model prediction
        class_index = np.argmax(model.predict(img_array))
        
        if class_index is not None:
            # Generate the Saliency Map
            saliency_map = compute_saliency_map(img_array, model, class_index)
            
            # Resize the image to match the saliency map dimensions
            img_resized = np.array(image.resize((224, 224)))
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            
            # Apply the Saliency Map as a heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))

            # Blend heatmap and original image with adjustable opacity
            superimposed_img = cv2.addWeighted(heatmap, heatmap_intensity, img_bgr, overlay_opacity, 0)
            superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

            # Display the result
            st.image(superimposed_img_rgb, caption="Enhanced Saliency Map Overlay", use_container_width=True)
        else:
            st.warning("Please predict the image class first.")
            
        # Forward Propagation Visualization
    st.subheader("🔍 Forward Propagation Visualization")

    # Create a model that outputs the activations of each layer
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    # Get activations for the input image
    activations = activation_model.predict(img_array)

    # Extract layer names for the selectbox
    layer_names = [layer.name for layer in model.layers if "conv" in layer.name or "pool" in layer.name]

    # Add a selectbox for layer selection
    selected_layer = st.selectbox("Select Layer to Visualize 🎯", layer_names)

    # Find the index of the selected layer
    layer_index = [layer.name for layer in model.layers].index(selected_layer)
    
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.download_button("Download Feature Maps", buf.getvalue(), file_name="feature_maps.png", mime="image/png")

    # Visualize activations for the selected layer
    st.subheader(f"Layer: {selected_layer}")
    layer_activation = activations[layer_index]

    # Display feature maps for the selected layer
    num_filters = layer_activation.shape[-1]
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    selected_filter = st.slider("Select Filter", 1, num_filters, 1)
    selected_filter_index = selected_filter - 1  
    fig, ax = plt.subplots()
    ax.imshow(layer_activation[0, :, :, selected_filter_index], cmap='viridis')
    ax.set_title(f"Filter {selected_filter}")  # Keep one-based index for display
    ax.axis('off')
    st.pyplot(fig)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.ravel()
    for j in range(num_filters):
        if j < len(axes):
            axes[j].imshow(layer_activation[0, :, :, j], cmap='viridis')
            axes[j].axis('off')
    st.pyplot(fig)
            
    # Custom Filter Creation
    st.subheader("🎨 Custom Filter Creation")

    # Filter size selection
    filter_size = st.slider("Filter Size", min_value=2, max_value=5, value=3)

    def apply_example_filter(weights):
        """Helper function to set filter weights in the session state"""
        flat_weights = weights.flatten()
        for i in range(len(flat_weights)):
            row = i // weights.shape[1]
            col = i % weights.shape[1]
            st.session_state[f"weight_{row}_{col}"] = flat_weights[i]

    # Example Filters Dictionary
    example_filters = {
        2: {
            "Simple Edge Detector": np.array([
                [1, -1],
                [1, -1]
            ]),
            "Diagonal Edge": np.array([
                [1, -1],
                [-1, 1]
            ]),
            "Simple Gradient": np.array([
                [1, 2],
                [0, -1]
            ])
        },
        3: {
            "Sobel Horizontal": np.array([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]),
            "Sobel Vertical": np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),
            "Laplacian": np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]),
            "Gaussian Blur": np.array([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]),
            "Sharpen": np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]),
            "Ridge Detection": np.array([
                [-1, -1, -1],
                [2, 2, 2],
                [-1, -1, -1]
            ]),
            "Emboss": np.array([
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ])
        },
        4: {
            "Extended Edge": np.array([
                [1, 1, -1, -1],
                [1, 1, -1, -1],
                [1, 1, -1, -1],
                [1, 1, -1, -1]
            ]),
            "Center Focus": np.array([
                [-1, -1, -1, -1],
                [-1, 2, 2, -1],
                [-1, 2, 2, -1],
                [-1, -1, -1, -1]
            ]),
            "Diamond Pattern": np.array([
                [0, 1, 1, 0],
                [1, 2, 2, 1],
                [1, 2, 2, 1],
                [0, 1, 1, 0]
            ])
        },
        5: {
            "Extended Gaussian": np.array([
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ]) / 256,
            "Extended Laplacian": np.array([
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ]),
            "Concentric Pattern": np.array([
                [-1, -1, -1, -1, -1],
                [-1, 2, 2, 2, -1],
                [-1, 2, 3, 2, -1],
                [-1, 2, 2, 2, -1],
                [-1, -1, -1, -1, -1]
            ])
        }
    }

    # Display example filters for current size
    st.subheader(f"📚 Example {filter_size}x{filter_size} Filters")
    cols = st.columns(2)
    for idx, (name, weights) in enumerate(example_filters[filter_size].items()):
        with cols[idx % 2]:
            if st.button(f"Apply {name} Filter"):
                apply_example_filter(weights)
            st.code(str(weights))

    # Create a grid layout for filter weights
    st.write("Set Filter Weights:")
    cols = st.columns(filter_size)
    filter_weights = np.zeros((filter_size, filter_size))

    # Create number inputs in a grid layout
    for i in range(filter_size):
        with cols[i]:
            for j in range(filter_size):
                filter_weights[j][i] = st.number_input(
                    f"Weight [{j},{i}]",
                    min_value=-100.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    key=f"weight_{j}_{i}"
                )

    # Add ReLU toggle
    apply_relu = st.checkbox("Apply ReLU Activation", value=True)

    if st.button("Apply Custom Filter"):
        if uploaded_file:
            # Convert the image to grayscale for single-channel processing
            gray_image = Image.open(uploaded_file).convert('L')
            gray_image = gray_image.resize((224, 224))
            input_array = np.array(gray_image) / 255.0
            
            # Reshape filter weights for convolution
            filter_weights_reshaped = filter_weights.reshape(filter_size, filter_size, 1, 1)
            
            # Create and apply the custom convolution layer
            custom_filter = tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=(filter_size, filter_size),
                padding='same',
                use_bias=False
            )
            
            # Set the custom weights
            custom_filter.build((None, 224, 224, 1))
            custom_filter.set_weights([filter_weights_reshaped])
            
            # Apply convolution
            input_tensor = tf.convert_to_tensor(input_array[np.newaxis, :, :, np.newaxis])
            output = custom_filter(input_tensor)
            
            # Apply ReLU if selected
            if apply_relu:
                output = tf.nn.relu(output)
            
            # Convert to numpy for visualization
            output_array = output.numpy()[0, :, :, 0]
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(input_array, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Filter visualization
            filter_img = ax2.imshow(filter_weights, cmap='viridis')
            ax2.set_title('Filter Weights')
            ax2.axis('off')
            plt.colorbar(filter_img, ax=ax2, fraction=0.046, pad=0.04)
            
            # Filtered image
            filtered_img = ax3.imshow(output_array, cmap='viridis')
            ax3.set_title('Filtered Image' + (' (with ReLU)' if apply_relu else ''))
            ax3.axis('off')
            plt.colorbar(filtered_img, ax=ax3, fraction=0.046, pad=0.04)
            
            # Display the plot
            st.pyplot(fig)
            
            # Additional explanation
            st.markdown("""
            ### Understanding the Visualization:
            1. **Left**: Original grayscale image
            2. **Middle**: Your custom filter weights (blue = negative, yellow = positive)
            3. **Right**: Result of applying your filter""")
            
            if apply_relu:
                st.markdown("""
                🔍 **ReLU Effect**: Notice how ReLU has removed all negative values, 
                keeping only the positive activations. This non-linearity helps neural 
                networks learn more complex patterns.""")
                
        else:
            st.warning("Please upload an image first!")

    # Custom Layer Creation
    st.sidebar.header("Custom Layer Creation 🛠")
    num_filters = st.sidebar.slider("Number of Filters", min_value=1, max_value=64, value=16)
    kernel_size = st.sidebar.slider("Kernel Size", min_value=1, max_value=11, value=3)
    stride = st.sidebar.slider("Stride", min_value=1, max_value=5, value=1)
    padding = st.sidebar.selectbox("Padding", ["valid", "same"])

    # Create a session state to store weights
    if 'custom_weights' not in st.session_state:
        st.session_state.custom_weights = None
        st.session_state.prev_filters = num_filters
        st.session_state.prev_kernel = kernel_size

    if st.sidebar.button("Apply Custom Layer"):
        # Check if we need to generate new weights
        generate_new_weights = (st.session_state.custom_weights is None or 
                            st.session_state.prev_filters != num_filters or 
                            st.session_state.prev_kernel != kernel_size)
        
        # Create the custom layer
        custom_layer = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding=padding,
            use_bias=True
        )
        
        # Generate input shape for layer initialization
        dummy_input = tf.keras.layers.Input(shape=(224, 224, 3))
        _ = custom_layer(dummy_input)
        
        if generate_new_weights:
            # Store the new weights
            st.session_state.custom_weights = custom_layer.get_weights()
            st.session_state.prev_filters = num_filters
            st.session_state.prev_kernel = kernel_size
        else:
            # Use stored weights
            custom_layer.set_weights(st.session_state.custom_weights)
        
        st.subheader("🔥 Custom Layer Visualization")

        # Preprocess Image
        img_array_custom = np.array(image.resize((224, 224))) / 255.0
        img_array_custom = np.expand_dims(img_array_custom, axis=0)

        # Apply Custom Layer
        feature_maps = custom_layer(img_array_custom).numpy()

        # Handle visualization differently based on number of filters
        if num_filters == 1:
            # Single filter case
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(feature_maps[0, :, :, 0], cmap='viridis')
            ax.axis('off')
        else:
            # Multiple filters case
            grid_size = int(np.ceil(np.sqrt(num_filters)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            axes = axes.ravel()
            for i in range(num_filters):
                if i < len(axes):
                    axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
                    axes[i].axis('off')
            # Hide empty subplots
            for i in range(num_filters, len(axes)):
                axes[i].axis('off')
                axes[i].set_visible(False)
        
        st.pyplot(fig)