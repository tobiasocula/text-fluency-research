from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path

#font = ImageFont.truetype("times.ttf", 80)
#font = ImageFont.truetype(str(Path.cwd() / "arial" / "ARIAL.ttf"), 80)
#font = ImageFont.truetype(str(Path.cwd() / "amatic" / "Amatic-Bold.ttf"), 100)
font = ImageFont.truetype(str(Path.cwd() / "chunkfive" / "Chunk Five Print.otf"), 80)

text = """This sentence contains every letter of the alphabet.
the quick brown fox jumps over the lazy dog."""

#text = "abcdefghijklmnopqrstuvwxyz"

newtext = ""
for char in [x.capitalize() for x in text]:
    newtext += char

img = Image.new("L", (1200,1600), 255)
draw = ImageDraw.Draw(img)

lines = textwrap.wrap(newtext, width=20)

y = 50
for line in lines:
    draw.text((50,y), line, font=font, fill=0)
    y += 100

img.save("chunkfive_text.png")