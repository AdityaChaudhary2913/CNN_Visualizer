import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.applications.efficientnet import decode_predictions
import matplotlib.pyplot as plt
import cv2

import warnings
warnings.filterwarnings("ignore")

# Set up the app
st.set_page_config(page_title="CNN Visualization App", layout="wide")
st.title("CNN Visualization App 📊")

# Sidebar for Model Selection
st.sidebar.header("Configuration Settings")
model_type = st.sidebar.selectbox("Choose Model", ["VGG16", "EfficientNetB0"])

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
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    class_index = None

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

    # # Visualization Section
    # st.subheader("🔬 CNN Visualizations")

    # # Select Layer for Visualizations
    # layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
    # selected_layer = st.selectbox("Select Layer for Visualization 🎯", layer_names)

    # # Feature Map Visualization
    # if st.button("Visualize Feature Maps"):
    #     feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(selected_layer).output)
    #     feature_maps = feature_model.predict(img_array)

    #     # Display Feature Maps
    #     num_filters = feature_maps.shape[-1]
    #     grid_size = int(np.ceil(np.sqrt(num_filters)))

    #     fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    #     axes = axes.ravel()
    #     for i in range(num_filters):
    #         if i < len(axes):
    #             axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
    #             axes[i].axis('off')
    #     st.pyplot(fig)

    # Grad-CAM Visualization Section
    st.subheader("🔥 Grad-CAM Overlay Visualization")

    def compute_grad_cam(img_array, model, layer_name, class_index):
        last_conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, model_outputs = grad_model(img_array)
            tape.watch(conv_outputs)
            class_score = model_outputs[:, class_index]

        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = np.mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        for i in range(conv_outputs.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    if st.button("Generate Grad-CAM Overlay"):
        grad_cam_layer = "top_conv" if model_type == "EfficientNetB0" else "block5_conv3"
        grad_cam_heatmap = compute_grad_cam(img_array, model, grad_cam_layer, class_index)

        img = np.array(image.resize((224, 224)))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))

        # Blend heatmap and original image
        superimposed_img = cv2.addWeighted(heatmap, 0.5, img_bgr, 0.5, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        st.image(superimposed_img_rgb, caption="Grad-CAM Overlay", use_container_width=True)

    # Custom Layer Creation
    st.sidebar.header("Custom Layer Creation 🛠️")
    num_filters = st.sidebar.slider("Number of Filters", min_value=1, max_value=64, value=16)
    kernel_size = st.sidebar.slider("Kernel Size", min_value=1, max_value=11, value=3)
    stride = st.sidebar.slider("Stride", min_value=1, max_value=5, value=1)
    padding = st.sidebar.selectbox("Padding", ["valid", "same"])

    if st.sidebar.button("Apply Custom Layer"):
        custom_layer = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding=padding
        )
        
        st.subheader("🔥 Custom Layer Visualization")

        # Preprocess Image
        img_array_custom = np.array(image.resize((224, 224))) / 255.0
        img_array_custom = np.expand_dims(img_array_custom, axis=0)

        # Apply Custom Layer
        feature_maps = custom_layer(img_array_custom).numpy()

        # Display Feature Maps
        num_filters = feature_maps.shape[-1]
        grid_size = int(np.ceil(np.sqrt(num_filters)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.ravel()
        for i in range(num_filters):
            if i < len(axes):
                axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
                axes[i].axis('off')
        st.pyplot(fig)

    # Interactive CNN Builder
    st.sidebar.header("Interactive CNN Builder 🏗️")
    selected_layers = st.sidebar.multiselect(
        "Select Layers",
        ["Conv2D", "MaxPooling2D", "ReLU", "Flatten", "Dense"]
    )

    if st.sidebar.button("Build CNN"):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = inputs
        for layer in selected_layers:
            if layer == "Conv2D":
                x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
            elif layer == "MaxPooling2D":
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            elif layer == "ReLU":
                x = tf.keras.layers.ReLU()(x)
            elif layer == "Flatten":
                x = tf.keras.layers.Flatten()(x)
            elif layer == "Dense":
                x = tf.keras.layers.Dense(128)(x)
        model_custom = tf.keras.Model(inputs, x)

        # Preprocess Image
        img_array_custom = np.array(image.resize((224, 224))) / 255.0
        img_array_custom = np.expand_dims(img_array_custom, axis=0)
        
        st.subheader("🔥 Custom CNN Visualization")

        # Visualize Feature Maps at Each Layer
        for layer in model_custom.layers:
            if "conv" in layer.name or "pool" in layer.name:
                feature_model = tf.keras.Model(inputs=model_custom.input, outputs=layer.output)
                feature_maps = feature_model.predict(img_array_custom)

                st.subheader(f"Layer: {layer.name}")
                num_filters = feature_maps.shape[-1]
                grid_size = int(np.ceil(np.sqrt(num_filters)))

                fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
                axes = axes.ravel()
                for i in range(num_filters):
                    if i < len(axes):
                        axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
                        axes[i].axis('off')
                st.pyplot(fig)

    # Advanced Visualization
    st.subheader("Advanced Visualization 🔬")

    # Grad-CAM Adjustments
    heatmap_intensity = st.slider("Heatmap Intensity", min_value=0.1, max_value=1.0, value=0.5)
    overlay_opacity = st.slider("Overlay Opacity", min_value=0.1, max_value=1.0, value=0.5)

    if st.button("Generate Enhanced Grad-CAM"):
        grad_cam_layer = "top_conv" if model_type == "EfficientNetB0" else "block5_conv3"
        grad_cam_heatmap = compute_grad_cam(img_array, model, grad_cam_layer, class_index)

        img = np.array(image.resize((224, 224)))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))

        # Blend heatmap and original image with adjustable opacity
        superimposed_img = cv2.addWeighted(heatmap, heatmap_intensity, img_bgr, overlay_opacity, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        st.image(superimposed_img_rgb, caption="Enhanced Grad-CAM Overlay", use_container_width=True)

    # Class Activation Mapping (CAM)
    if st.button("Generate CAM"):
        cam_layer = "top_conv" if model_type == "EfficientNetB0" else "block5_conv3"
        cam_model = tf.keras.Model(inputs=model.input, outputs=[model.get_layer(cam_layer).output, model.output])
        conv_outputs, preds = cam_model(img_array)
        class_weights = np.mean(conv_outputs[0], axis=(0, 1))
        cam = np.dot(conv_outputs[0], class_weights)

        plt.imshow(cam, cmap='hot')
        plt.axis('off')
        st.pyplot()
        
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

    st.markdown("""
    ### How Filters and Channels Work:
    - **Input Channels:** The number of channels in the input image (e.g., 3 for RGB).
    - **Filters:** Each filter applies a convolution operation to the input.
    - **Output Feature Maps:** The number of output feature maps equals the number of filters.
    """)