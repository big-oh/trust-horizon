"""Prepend the cover to the paper without touching the paper's pages, links or bookmarks.

Needs pypdf (pip install pypdf). Run from this directory after building the cover:
    python merge.py   -> ../main_ieee_with_cover.pdf
"""
from pypdf import PdfReader, PdfWriter

paper = "../main_ieee.pdf"
w = PdfWriter()
w.append("gradient.pdf", outline_item="Cover")
w.append(paper)
w.add_metadata(dict(PdfReader(paper).metadata or {}))
w.write("../main_ieee_with_cover.pdf")
