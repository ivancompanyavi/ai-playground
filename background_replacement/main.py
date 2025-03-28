import numpy as np
import torch
from PIL import Image
import argparse
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, AutoFeatureExtractor
from diffusers import StableDiffusionPipeline


class BackgroundReplacer:
    def __init__(self, segmentation_model="facebook/detr-resnet-50-panoptic",
                 diffusion_model="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the background replacer with a segmentation model and text-to-image model.

        Args:
            segmentation_model (str): HuggingFace model identifier for the segmentation model
            diffusion_model (str): HuggingFace model identifier for the text-to-image model
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load segmentation model
        print(f"Loading segmentation model {segmentation_model}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            segmentation_model)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            segmentation_model).to(self.device)

        # Load text-to-image model
        print(f"Loading text-to-image model {diffusion_model}...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            diffusion_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        # Enable memory optimization for Stable Diffusion if on CUDA
        if torch.cuda.is_available():
            self.txt2img_pipe.enable_attention_slicing()

        print(f"All models loaded successfully. Using device: {self.device}")

    def _create_mask(self, image):
        """
        Create a binary mask separating foreground from background.

        Args:
            image (PIL.Image): Input image

        Returns:
            PIL.Image: Binary mask where white indicates foreground, same size as input image
        """
        # Store original image size
        original_width, original_height = image.size

        # Prepare the image for the model
        inputs = self.feature_extractor(
            images=image, return_tensors="pt").to(self.device)

        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the processed image size from the feature extractor
        # The feature extractor typically resizes the image for the model
        processed_height, processed_width = inputs["pixel_values"].shape[-2:]

        # Post-process in the format expected by the model
        target_sizes = [[processed_width, processed_height]]
        results = self.feature_extractor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes)[0]

        # Extract segmentation map and metadata
        panoptic_seg = results["segmentation"].cpu().numpy()
        segments_info = results["segments_info"]

        print(f"Found {len(segments_info)} segments")
        if segments_info:
            print(f"First segment info: {segments_info[0]}")

        # Initialize an empty mask
        mask = np.zeros_like(panoptic_seg, dtype=np.uint8)

        # Define COCO thing classes (objects that are typically foreground)
        thing_classes = set([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ])

        # Mark foreground pixels in the mask
        foreground_found = False
        for segment in segments_info:
            # Check if this segment is a foreground object
            if segment["label_id"] in thing_classes:
                segment_id = segment["id"]
                # Use 255 for white in 8-bit
                mask[panoptic_seg == segment_id] = 255
                foreground_found = True
                print(
                    f"Added segment {segment_id} with label {segment['label_id']} as foreground")

        # Fallback if no foreground objects were detected
        if not foreground_found:
            print("No foreground objects detected. Using central segment as fallback.")
            h, w = mask.shape
            center_y, center_x = h // 2, w // 2
            center_segment_id = panoptic_seg[center_y, center_x]
            mask[panoptic_seg == center_segment_id] = 255

        # Convert to PIL Image for easy resizing
        mask_pil = Image.fromarray(mask).convert("L")

        # Resize mask to match original image dimensions
        mask_pil = mask_pil.resize(
            (original_width, original_height), Image.LANCZOS)

        # Debugging: Save mask for inspection
        mask_pil.save("debug/debug_mask.png")
        print(f"Debug mask saved with size: {mask_pil.size}")

        return mask_pil

    def generate_background(self, prompt, width=512, height=512, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate a background image based on a text prompt.

        Args:
            prompt (str): Text description of the desired background
            width (int): Width of the generated image
            height (int): Height of the generated image
            num_inference_steps (int): Number of denoising steps (higher = better quality but slower)
            guidance_scale (float): How closely the model should follow the prompt (higher = more faithful to prompt)

        Returns:
            PIL.Image: The generated background image
        """
        print(f"Generating background from prompt: '{prompt}'")

        # Add a hint to generate a background
        enhanced_prompt = f"background scene of {prompt}, wide angle, detailed environment"

        # Generate the image
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = self.txt2img_pipe(
                prompt=enhanced_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        return image

    def replace_background(self, img_name, background_prompt=None, background_img_path=None):
        """
        Replace the background of the foreground image with either a generated background from a prompt
        or an existing background image.

        Args:
            img_name (str): Name of the image
            background_prompt (str, optional): Text prompt to generate a background image
            background_img_path (str, optional): Path to an existing background image (used if prompt is None)

        Returns:
            PIL.Image: The resulting image with replaced background
        """
        # Load foreground image
        foreground_img = Image.open(
            f'input/{img_name}.png').convert("RGB")
        original_size = foreground_img.size
        print(f"Foreground image size: {original_size}")

        # Get or generate background image
        if background_prompt is not None:
            # Generate background from prompt
            background_img = self.generate_background(
                background_prompt,
                width=original_size[0],
                height=original_size[1]
            )
            print(f"Generated background image size: {background_img.size}")
        elif background_img_path is not None:
            # Load existing background image
            background_img = Image.open(background_img_path).convert("RGB")
            # Resize background to match foreground dimensions
            background_img = background_img.resize(
                original_size, Image.LANCZOS)
            print(
                f"Loaded background image size after resize: {background_img.size}")
        else:
            raise ValueError(
                "Either background_prompt or background_img_path must be provided")

        # Create foreground mask
        mask_img = self._create_mask(foreground_img)
        print(f"Created mask with size: {mask_img.size}")

        # Ensure all images have the exact same size
        foreground_img = foreground_img.resize(original_size, Image.LANCZOS)
        background_img = background_img.resize(original_size, Image.LANCZOS)
        mask_img = mask_img.resize(original_size, Image.LANCZOS)

        # Verify sizes match
        assert foreground_img.size == background_img.size == mask_img.size, "Image sizes don't match!"
        print(
            f"Final sizes - Foreground: {foreground_img.size}, Background: {background_img.size}, Mask: {mask_img.size}")

        # Composite the images using PIL's composite function
        # This avoids numpy broadcasting issues
        result_img = Image.composite(foreground_img, background_img, mask_img)

        # Save the result if an output path is provided
        result_img.save(f"output/{img_name}.png")
        print(f"output/{img_name}.png")

        return result_img


# Example usage
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Replace the background of an image using AI')

    # Required arguments
    parser.add_argument('--foreground', '-f', required=True,
                        help='Path to the foreground image (with the subject)')

    # Optional arguments
    parser.add_argument('--background-image', '-bi',
                        help='Path to the background image (either this or --background-prompt must be provided)')
    parser.add_argument('--background-prompt', '-bp',
                        help='Text prompt to generate a background (either this or --background-image must be provided)')
    parser.add_argument('--segmentation-model', '-sm', default='facebook/detr-resnet-50-panoptic',
                        help='HuggingFace model identifier for segmentation (default: facebook/detr-resnet-50-panoptic)')
    parser.add_argument('--diffusion-model', '-dm', default='runwayml/stable-diffusion-v1-5',
                        help='HuggingFace model identifier for diffusion (default: runwayml/stable-diffusion-v1-5)')
    parser.add_argument('--steps', '-s', type=int, default=50,
                        help='Number of inference steps for background generation (default: 30)')
    parser.add_argument('--guidance', '-g', type=float, default=7.5,
                        help='Guidance scale for background generation (default: 7.5)')

    # Parse arguments
    args = parser.parse_args()

    # Check if either background image or prompt is provided
    if args.background_image is None and args.background_prompt is None:
        with open("input/prompt.txt", "r") as file:
            args.background_prompt = file.read().strip()

    # Initialize the background replacer with specified models
    replacer = BackgroundReplacer(
        segmentation_model=args.segmentation_model,
        diffusion_model=args.diffusion_model
    )

    # Process the image
    result = replacer.replace_background(
        img_name=args.foreground,
        background_img_path=args.background_image,
        background_prompt=args.background_prompt,
    )
