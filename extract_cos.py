import pdfplumber

with pdfplumber.open('COS.pdf') as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(f"Page {page.page_number}:")
        print(text)
        print("\n" + "="*50 + "\n")