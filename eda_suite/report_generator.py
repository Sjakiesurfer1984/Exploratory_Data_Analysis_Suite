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

    def create_word_document(
        self,
        plot_cache: List[io.BytesIO],
        analyzer_name: str = "analyzer",
        df_preview=None,
    ) -> None:
        """
        Creates a fully formatted timestamped Word report.
    
        Args:
            plot_cache (List[io.BytesIO]): Cached plot images.
            analyzer_name (str): Logical name for the analyzer instance.
            df_preview (pd.DataFrame | None): Optional DataFrame head() for preview.
        """
        from datetime import datetime
        import docx
        from docx.shared import Inches
    
        # Generate descriptive timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"eda_report_{analyzer_name}_{timestamp}.docx"
    
        document = docx.Document()
        document.add_heading(f"Exploratory Data Analysis Report – {analyzer_name}", level=1)
        document.add_paragraph(f"Report generated on: {timestamp}")
        document.add_paragraph("This document summarises exploratory findings and visualisations.")
    
        # Add optional DataFrame preview
        if df_preview is not None:
            document.add_heading("Data Preview (First 5 Rows)", level=2)
            table = document.add_table(rows=1, cols=len(df_preview.columns))
            table.style = 'Table Grid'
            hdr_cells = table.rows[0].cells
            for i, col_name in enumerate(df_preview.columns):
                hdr_cells[i].text = str(col_name)
            for _, row in df_preview.iterrows():
                row_cells = table.add_row().cells
                for i, val in enumerate(row):
                    row_cells[i].text = str(val)
            document.add_page_break()
    
        # Add plots
        for i, img_buffer in enumerate(plot_cache):
            document.add_heading(f"Plot {i + 1}", level=2)
            document.add_picture(img_buffer, width=Inches(6.0))
            document.add_page_break()
    
        try:
            document.save(filename)
            print(f"Successfully generated report: '{filename}'")
        except Exception as e:
            print(f"Error: Could not save the document. Reason: {e}")
