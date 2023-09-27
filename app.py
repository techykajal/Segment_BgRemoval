# %% [markdown]
# ## Gradio UI for Instance Segmentation and BG Removal using SAM

# %% [markdown]
# #### Import Libraries

# %%
import gradio as gr
import numpy as np
import cv2
import sys
import time
from segment_anything import sam_model_registry, SamPredictor
from matplotlib import pyplot as plt
import torch
import torchvision
from PIL import Image

# %% [markdown]
# #### Gradio Interface

# %%
positive_mask_coordinates = set() #Use set to avoid repetition of coordinates
mask_list=[]
final_mask= None
img_h=None
img_w=None
with gr.Blocks() as res:
    gr.Markdown(
        """
    # Background Remover
    Click on the image to record positive (left click). The background will be removed from the image based on these points.
    """
    )
    with gr.Row():
        input_img = gr.Image(label="Upload Image")
        output_mask_img = gr.Image(label="Mask Image") 
    with gr.Row():
        with gr.Row():
            clear_btn = gr.Button(value="Clear",variant="secondary")
            reset_btn = gr.Button(value="Reset Coordinates",variant="primary")            
        with gr.Row():
            select_mask_btn = gr.Button(value="Select Mask",variant="primary")
            reject_mask_btn = gr.Button(value="Reject Mask",variant="secondary")
    with gr.Row():
        output_final_mask_img = gr.Image(label="Final Mask Image")
        output_bg_removed_img = gr.Image(label="Background Removed Image") 
    with gr.Row():
        with gr.Row():
            save_mask_btn=gr.Button(value="Save Final Mask", variant="secondary")
            bg_btn=gr.Button(value="Remove Background", variant="primary")
            save_bg_btn=gr.Button(value="Save Background Image", variant="secondary")

    def segment_image(image, input_point, input_label):
        """
        Segment an input image using a Spatially Adaptive Mask (SAM) model.

        Args:
        image (ndarray): An input image array of shape (height, width, channels) representing the image to be segmented.
        input_point (list): A list of two integers representing the x,y coordinates of the point in the image to use as the mask seed.
        input_label (int): An integer representing the label of the object to be segmented.

        Returns:
        mask_image (ndarray): A binary mask array of shape (height, width, 3) representing the segmented image.
        """
        sam_checkpoint = "Model_Checkpoints/sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "default"
        sys.path.append("..")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # Load the SAM model checkpoint
        sam.to(device=device) # Move the model to the specified device (GPU or CPU)
        predictor = SamPredictor(sam)
        predictor.set_image(image) # Set the input image for the predictor

        # Predict the masks, scores, and logits for the input point and label using the SAM model
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Choose the model's best mask based on the highest score
        mask_input = logits[np.argmax(scores), :, :]

        # Predict the final mask for the chosen mask input using the SAM model
        masks_, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )

        if masks_.any():
            print("masks")
        else:
            # If no points are selected, return a mask with all pixels set to 1
            masks_ = np.ones((image.shape[0], image.shape[1]))

        # Set the color for the binary mask image (black and white by default)
        random_color=False
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([1.5])], axis=0)
        else:
            color = np.array([25, 25, 1.0])

        # Reshape the mask array to (height, width, 1) and multiply it with the color array to create the final mask image
        h, w = masks_.shape[-2:]
        mask_image = masks_.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # Return the final mask image
        return mask_image

    def concatenate_masks(masklist, final_mask):
        """
        Concatenates a list of binary masks using the bitwise OR operation.

        Args:
        - masklist (list): a list of binary masks, each represented as a numpy ndarray.
        - final_mask (ndarray): a numpy ndarray that represents the final concatenated mask.

        Returns:
        - final_mask (ndarray): a numpy ndarray that represents the final concatenated mask.
        """
        # Initialize the final mask as an array of zeros with the same shape as the first mask in the list
        final_mask = np.zeros_like(masklist[0])

        # Iterate through each mask in the list and apply a bitwise OR operation to the final mask
        for mask in masklist:
            final_mask = cv2.bitwise_or(final_mask, mask)

        # Return the final concatenated mask
        return final_mask


    def get_final_mask(masklist, img_h, img_w):
        """
        Combines a list of masks into a final mask by concatenating them together.
        The final mask has the same dimensions as the original image specified by img_h and img_w.

        Args:
        masklist (list): A list of binary mask arrays.
        img_h (int): Height of the original image.
        img_w (int): Width of the original image.

        Returns:
        final_mask (ndarray): A binary mask array of shape (img_h, img_w) representing the final mask.
        """
        global final_mask # Use a global variable to store the final mask
        if final_mask is None: # If final_mask is None (i.e. not yet initialized), initialize it as all zeros
            final_mask = np.zeros((img_h, img_w)) 
        final_mask = concatenate_masks(masklist, final_mask) # Concatenate the masklist to the final_mask
        return final_mask

    
    def remove_background(image, mask):
        """
        Remove the background of an image using a binary mask.

        Args:
        image (ndarray): An RGB image to be processed.
        mask (ndarray): A binary mask with the same height and width as the input image.

        Returns:
        bg_removed_image (ndarray): An RGB image with the background removed.

        Raises:
        ValueError: If the shape of the mask is invalid (i.e. not (h, w, 3) or (h, w)).
        """
        # Check if mask has a valid shape (i.e. (h, w, 3) or (h, w))
        if len(mask.shape) == 2 or mask.shape[2] == 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) # Convert grayscale mask to RGB
        elif mask.shape[2] != 3:
            raise ValueError("Invalid mask shape. Must be (h, w, 3) or (h, w).")

        bg_removed_image = cv2.bitwise_and(image, mask) # Apply the mask to the input image using bitwise AND
        return bg_removed_image


    def run_app_mask(input_img):
        """
        Takes an input image, segments it using the positive_mask_coordinates,
        and returns a binary mask of the segmented region.

        Args:
        input_img (ndarray): An input image array.

        Returns:
        mask (ndarray): A binary mask array representing the segmented region of the input image.
        """
        input_points = np.array(list(positive_mask_coordinates)) # Convert positive_mask_coordinates to numpy array
        input_labels = np.ones(len(positive_mask_coordinates)) # Create an array of ones with length equal to the number of points in positive_mask_coordinates
        img_h, img_w = input_img.shape[:2] # Get the height and width of the input image
        list_masks = [] # Initialize an empty list to hold all masks
        for i in range(len(input_points)):
            # Segment the input image using the i-th point in input_points and corresponding label in input_labels
            mask = segment_image(input_img, input_points[i:i+1], input_labels[i:i+1])  
            list_masks.append(mask) # Add the resulting mask to list_masks
        mask = get_final_mask(list_masks, img_h, img_w) # Concatenate all masks in list_masks into a final mask
        return mask # Return the final mask as output of the function


    def run_app_final_mask(mask_list):
        """
        Runs the application by generating a final mask from a list of binary masks.

        Args:
        mask_list (list): A list of binary mask arrays.

        Returns:
        final_mask (ndarray): A binary mask array of the same dimensions as the original image, representing the final mask.
        """
        global img_h, img_w # Access the global variables img_h and img_w
        final_mask = get_final_mask(mask_list, img_h, img_w) # Generate the final mask by calling get_final_mask function
        return final_mask


    def run_app_bg_removed(input_img):
        """
        Removes the background from the input image using the final_mask accumulated from previous masks.
        The final_mask should be a binary mask array of the same dimensions as the input image.

        Args:
        input_img (ndarray): A numpy array representing the input image.

        Returns:
        bg_removed_image (ndarray): A numpy array representing the input image with the background removed.
        """
        bg_removed_image = remove_background(input_img, final_mask) # Remove the background from the input image using the final_mask
        return bg_removed_image


    def get_select_coords(img, evt: gr.SelectData):
        """
        Update positive_mask_coordinates set with the new click event and return the mask image.

        Args:
        img (ndarray): An input image array.
        evt (gr.SelectData): A SelectData object that contains the coordinates of the click event.

        Returns:
        out (ndarray): A binary mask array representing the mask image after running the run_app_mask function.
        """
        global img_h, img_w
        img_h, img_w = img.shape[:2]  # Get the height and width of the input image
        print("click event for positive coordinates : x,y ", evt, evt.index[0], evt.index[1], positive_mask_coordinates)
        positive_mask_coordinates.add((evt.index[0], evt.index[1])) # Update positive_mask_coordinates set with the new click event
        out = run_app_mask(img) # Generate the mask image by running the run_app_mask function on the input image
        out = np.uint8(255 * out) # Convert the mask image to a binary array of 0's and 255's
        return out


    def clear(*args):
        """
        Clears all state related to masks.

        Args:
        args: Optional positional arguments that are ignored.

        Returns:
        A list of Nones with the same length as the number of positional arguments passed in.
        """
        global mask_list, positive_mask_coordinates, final_mask # Use global variables for mask list, positive mask coordinates and final mask
        print("clearing all state")
        positive_mask_coordinates.clear() # Clear the positive mask coordinates list
        mask_list=[] # Reset the mask list to an empty list
        final_mask= None # Reset the final mask to None
        return [None for _ in args] # Return a list of Nones with the same length as the number of positional arguments passed in

    
    def select_mask(mask):
        """
        Adds the input mask to a list of masks and returns a combined mask obtained by concatenating all masks in the list.

        Args:
        mask (ndarray): A binary mask array to be added to the mask list.

        Returns:
        combined_mask (ndarray): A binary mask array obtained by concatenating all masks in the mask list.
        """
        global mask_list # Use a global variable to store the mask list
        mask_list.append(mask) # Add the input mask to the mask list
        combined_mask = run_app_final_mask(mask_list) # Obtain the combined mask by concatenating all masks in the mask list using the 'run_app_final_mask' function
        return combined_mask

    def reject_mask():
        """
        Returns a None object indicating that the mask was rejected.
        """
        return None

    
    def reset_coords():
        """
        Clears the global variable `positive_mask_coordinates` which stores the coordinates of positive pixels in a mask.
        """
        print("Reset: ")
        positive_mask_coordinates.clear()

        
    def save_final_mask(final_mask):
        """
        Saves the final binary mask as an image file in JPEG format.

        Args:
        final_mask (ndarray): A binary mask array of shape (img_h, img_w) representing the final mask to be saved.

        Returns:
        None
        """
        img = Image.fromarray(final_mask, "RGB") # Convert the binary mask to a PIL image object
        path = "Foreground_Predicted_Masks_SAM/Final_Mask.jpg" # Specify the file path and name for the saved image
        img.save(path) # Save the image to the specified path

        
    def save_bg_removed_img(bg_removed_image):
        """
        Saves the background removed image as an image file in JPEG format.

        Args:
        bg_removed_image (ndarray): An image array of shape (img_h, img_w, 3) representing the background removed image to be saved.

        Returns:
        None
        """
        bg_removed_img = Image.fromarray(bg_removed_image, "RGB") # Convert the image array to a PIL image object
        bg_removed_path = "SAM_bg_removed_Images/Bg_Removed_Image.jpg" # Specify the file path and name for the saved image
        bg_removed_img.save(bg_removed_path) # Save the image to the specified path


    input_img.select(get_select_coords, [input_img], [output_mask_img])
    bg_btn.click(run_app_bg_removed, inputs=[input_img], outputs=[output_bg_removed_img])
    clear_btn.click(clear, inputs=[input_img, output_final_mask_img, output_mask_img, output_bg_removed_img], outputs=[input_img, output_mask_img, output_bg_removed_img])
    select_mask_btn.click(select_mask, inputs=[output_mask_img], outputs=[output_final_mask_img])
    reject_mask_btn.click(reject_mask, [], [output_mask_img])
    reset_btn.click(reset_coords, [], [])
    save_mask_btn.click(save_final_mask, inputs=[output_final_mask_img])
    save_bg_btn.click(save_bg_removed_img, inputs=[output_bg_removed_img])

if __name__ == "__main__":
    res.launch(share=True)

# %%



