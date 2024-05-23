# SAMPLE USAGE:
# python3 mydraw_individual.py < add_chars.dat
# python 01_mydraw_individual.py --font /usr/share/fonts/truetype/MS/Arial-Unicode-MS.ttf < 00_arial_chars.txt
# python 01_mydraw_individual.py --font /usr/share/fonts/truetype/MS/PingFang-SC-Regular.ttf < 00_pingfang_chars.txt
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--font", type=str) # "/usr/share/fonts/truetype/MS/Arial-Unicode-MS.ttf" , "/usr/share/fonts/truetype/MS/PingFang-SC-Regular.ttf"
args = parser.parse_args()

width,height = 24,24
font_size=22
font_color=(0,0,0)

for line in sys.stdin:
 
	#print(i) 
	unicode_text = line.strip()
	try :
		i = ord(unicode_text)
		print(i, unicode_text)
	except TypeError :
		print("ERROR :", i)
		continue
	
	if i%500==0: 
		sys.stderr.write("%d\n"%i)
		sys.stderr.flush()

	im  =  Image.new ( "RGB", (width,height) )
	draw  =  ImageDraw.Draw ( im )
	unicode_font = ImageFont.truetype(args.font, font_size)
	font_name = args.font.split('/')[-1][:-4]
	try:
		draw.text ( (1,-4), unicode_text, font=unicode_font) #, fill=font_color )
	except SystemError:
		continue 
	# print(f"/home/nykim/HateSpeech/02_images/{font_name}/{i}.ppm")
	im.save(f"/home/nykim/HateSpeech/02_images/{font_name}/{i}.ppm")
