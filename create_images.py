from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image(filename, text, size=(300, 200), bg_color=(70, 130, 180), text_color=(255, 255, 255)):
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), text, font=None)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the text
    draw.text((x, y), text, fill=text_color)
    
    # Save the image
    img.save(filename)

def main():
    # Create static/images directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Create placeholder images
    images = {
        'logo.png': 'Logo',
        'chem_photo.jpg': 'Chemistry',
        'maths_photo.jpg': 'Mathematics',
        'phy_photo.jpg': 'Physics',
        'SMARTLEARN.png': 'SmartLearn'
    }
    
    for filename, text in images.items():
        filepath = os.path.join('static/images', filename)
        create_placeholder_image(filepath, text)
        print(f'Created {filepath}')

if __name__ == '__main__':
    main() 