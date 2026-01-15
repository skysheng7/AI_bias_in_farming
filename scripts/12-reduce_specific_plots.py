"""Reduce the size of specific PNG files for easier sharing and storage"""

from pathlib import Path
from PIL import Image
import click


@click.command()
@click.option(
    "--source_folder",
    default="./results/plots",
    type=str,
    help="Path to the folder containing source images",
)
@click.option(
    "--dest_folder",
    default="./results/plots_small",
    type=str,
    help="Path to the destination folder for resized images",
)
@click.option(
    "--scale_factor",
    default=0.12,
    type=float,
    help="Scaling factor for image size (e.g., 0.12 = 12% of original size)",
)
def main(source_folder, dest_folder, scale_factor):
    """Reduce size of specific plot images and save to plots_small folder."""

    # List of specific files to process
    target_files = [
        "basic_dall-e-3_plot_grid.png",
        "3d_country_plot_dairy.png",
        "3d_country_plot_pig.png",
        "3d_general_plot.png",
    ]

    source_path = Path(source_folder)
    dest_path = Path(dest_folder)

    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing images with scale factor: {scale_factor}")
    print(f"Source folder: {source_path}")
    print(f"Destination folder: {dest_path}")
    print("-" * 50)

    for filename in target_files:
        source_file = source_path / filename

        if not source_file.exists():
            print(f"⚠️  File not found: {filename}")
            continue

        # Load image
        img = Image.open(source_file)
        original_size = img.size

        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Calculate new size
        new_size = tuple(int(dim * scale_factor) for dim in original_size)

        # Resize image
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save resized image
        dest_file = dest_path / filename
        img_resized.save(dest_file, format="PNG", optimize=True)

        # Calculate file sizes
        original_size_mb = source_file.stat().st_size / (1024 * 1024)
        new_size_mb = dest_file.stat().st_size / (1024 * 1024)
        reduction_pct = ((original_size_mb - new_size_mb) / original_size_mb) * 100

        print(f"✓ {filename}")
        print(f"  Dimensions: {original_size} → {new_size}")
        print(
            f"  File size: {original_size_mb:.2f} MB → {new_size_mb:.2f} MB ({reduction_pct:.1f}% reduction)"
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
