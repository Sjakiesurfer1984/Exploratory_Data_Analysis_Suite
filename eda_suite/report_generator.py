# ==============================================================================
# report_generator.py
#
# This module provides the ReportGenerator class, which is responsible for
# creating shareable documents from the results of an analysis session. By
# separating this functionality, we keep the core analysis and visualisation
# code clean and focused on their respective tasks.
#
# Author: Tim Vos
# Last Modified: 2 October 2025
# ==============================================================================

import docx
from docx.shared import Inches
import io
from typing import List
from datetime import datetime

class ReportGenerator:
    """
    Generates reports from analysis artifacts, such as cached plot images.
    
    This class is a stateless service; it does not hold any data itself. It simply
    provides the functionality to process a given set of inputs (the plot cache)
    and produce a file as output.
    """

    def create_word_document(self, plot_cache: List[io.BytesIO], filename: str):
        """
        Creates a Microsoft Word document from a list of in-memory plot images.
        
        This method iterates through a cache of plot images, embedding each one
        into a newly created .docx file, complete with titles and proper formatting.
        
        Args:
            plot_cache (List[io.BytesIO]): A list of plot images, where each image
                                           is stored in an in-memory binary buffer.
            filename (str): The desired name for the output .docx file.
        """
        # A user-friendly check to ensure the filename has the correct extension.
        if not filename.endswith('.docx'):
            filename += '.docx'

        # Initialise a new, blank Word document object.
        document = docx.Document()
        
        # --- Add Report Header ---
        # Add a main title to the document. The 'level=1' corresponds to the
        # "Heading 1" style in Microsoft Word.
        document.add_heading('Exploratory Data Analysis Report', level=1)
        
        # Capture the current time to provide context for when the report was generated.
        # This is good practice for reproducibility and tracking analysis versions.
        timestamp = datetime.now().strftime('%A, %d %B %Y, %I:%M %p')
        document.add_paragraph(f"Report generated on: {timestamp}")
        document.add_paragraph(
            "This document contains the visualisations generated during the analysis session."
        )
        
        # --- Embed Plots from Cache ---
        # We use `enumerate` to get both the index (for numbering) and the item
        # as we loop through the list of cached plot images.
        for i, img_buffer in enumerate(plot_cache):
            # Add a subheading for each plot.
            document.add_heading(f'Plot {i + 1}', level=2)
            
            # Embed the image from the in-memory buffer into the document.
            # We explicitly set the width to a standard size (6.0 inches) to ensure
            # all plots are consistently formatted and fit well on a standard A4 page.
            document.add_picture(img_buffer, width=Inches(6.0))
            
            # For better readability, insert a page break after each plot. This ensures
            # that each visualisation starts on a fresh page.
            document.add_page_break()

        # --- Save the Final Document ---
        # It's robust practice to wrap file I/O operations in a try...except block
        # to gracefully handle potential filesystem errors (e.g., permission denied).
        try:
            document.save(filename)
            print(f"Successfully generated report: '{filename}'")
        except Exception as e:
            print(f"Error: Could not save the document. Reason: {e}")