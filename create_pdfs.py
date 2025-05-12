from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_chemistry_pdf(filename, title, content):
    c = canvas.Canvas(f"Finally/static/pdfs/{filename}", pagesize=letter)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, 750, title)
    
    c.setFont("Helvetica", 12)
    y = 700
    for line in content.split('\n'):
        c.drawString(72, y, line)
        y -= 20
    
    c.save()

# Create directory if it doesn't exist
os.makedirs("Finally/static/pdfs", exist_ok=True)

# Module 1: Introduction to Chemistry
intro_content = """
Chemistry is the study of matter and the changes it undergoes.
This module covers:
- Basic concepts and definitions
- States of matter
- Chemical reactions
- Atomic structure
- Periodic table

Key Learning Objectives:
1. Understand the fundamental principles of chemistry
2. Learn about different states of matter
3. Explore basic chemical reactions
4. Study atomic structure and the periodic table
"""

create_chemistry_pdf("chemistry_module1.pdf", "Introduction to Chemistry", intro_content)

# Module 2: Atomic Structure
atomic_content = """
Atomic Structure and the Periodic Table
This module covers:
- Atomic theory
- Subatomic particles
- Electron configuration
- Periodic trends
- Element properties

Key Learning Objectives:
1. Understand atomic theory and models
2. Learn about protons, neutrons, and electrons
3. Master electron configuration
4. Comprehend periodic trends
"""

create_chemistry_pdf("chemistry_module2.pdf", "Atomic Structure", atomic_content)

# Module 3: Chemical Bonding
bonding_content = """
Chemical Bonding and Molecular Structure
This module covers:
- Ionic bonds
- Covalent bonds
- Metallic bonds
- Molecular geometry
- VSEPR theory

Key Learning Objectives:
1. Understand different types of chemical bonds
2. Learn about molecular geometry
3. Apply VSEPR theory
4. Predict molecular shapes
"""

create_chemistry_pdf("chemistry_module3.pdf", "Chemical Bonding", bonding_content) 