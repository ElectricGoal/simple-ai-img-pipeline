from PIL import Image

image = Image.open("canny_image.png")

# reverse color
image = Image.eval(image, lambda x: 255 - x)
image.save("canny_image_reversed.png")