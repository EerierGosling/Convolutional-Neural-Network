from PIL import Image, ImageDraw
import random
import numpy as np

# folder = "test_data/"

# for shape in ["circle", "square", "triangle"]:
#     for i in range(100):
#         img = Image.new('RGB', (50, 50), color="#FFFFFF")
#         draw = ImageDraw.Draw(img)

#         if shape == "circle":
#             radius = random.randint(3,10)
#             draw.circle((random.randint(radius, 50-radius), random.randint(radius, 50-radius)), radius, fill="#000000")
        
#         elif shape == "square":
#             side = random.randint(3,10)
#             x = random.randint(0, 50-side)
#             y = random.randint(0, 50-side)
#             draw.rectangle(((x, y), (x+side, y+side)), fill="#000000")
        

#         elif shape == "triangle":
#             pass

#         img.save(folder + shape + f"/{i}.png")

def make_shape(shape:str) -> np.ndarray:
    img = Image.new('RGB', (50, 50), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    if shape == "circle":
        radius = random.randint(3,10)
        x = y = 25
        draw.circle((x, y), radius, fill="#000000") #(random.randint(radius, 50-radius), random.randint(radius, 50-radius))
    
    elif shape == "square":
        side = random.randint(3,10)
        # x = random.randint(0, 50-side)
        # y = random.randint(0, 50-side)
        x = y = 25
        draw.rectangle(((x, y), (x+side, y+side)), fill="#000000")
    

    elif shape == "triangle":
        pass

    return np.array(img)

    