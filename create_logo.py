from PIL import Image, ImageDraw, ImageFont
import os

def create_logo(filename, size=(400, 200)):
    # Create a new image with a gradient background
    img = Image.new('RGB', size, (15, 32, 39))
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for y in range(size[1]):
        r = int(15 + (y / size[1]) * 20)
        g = int(32 + (y / size[1]) * 30)
        b = int(39 + (y / size[1]) * 40)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    
    # Add text
    text = "SmartLearn"
    text_color = (0, 198, 255)  # Bright cyan color
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), text, font=None)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the text
    draw.text((x, y), text, fill=text_color)
    
    # Add a subtle glow effect
    for i in range(3):
        offset = i + 1
        draw.text((x-offset, y), text, fill=(0, 198, 255, 50))
        draw.text((x+offset, y), text, fill=(0, 198, 255, 50))
    
    # Save the image
    img.save(filename)

def main():
    # Create static/images directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Create logo
    logo_path = os.path.join('static/images', 'logo.png')
    create_logo(logo_path)
    print(f'Created {logo_path}')

if __name__ == '__main__':
    main() 