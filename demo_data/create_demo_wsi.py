from PIL import Image

img = Image.open("demo_data/sample_wsi.png")

img.save(
    "demo_data/small_sample_wsi.jpg",
    quality=40,
    optimize=True
)