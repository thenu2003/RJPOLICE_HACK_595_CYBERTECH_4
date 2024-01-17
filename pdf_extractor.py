import fitz  # PyMuPDF
from main import gen_pdf_inp
def extract_text_from_pdf(gen_pdf_inp):
    pdf_document = fitz.open(gen_pdf_inp)
    text_list = []

    for page_num in range(pdf_document.page_count):
        try:
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            text_list.append(text)

        except Exception as e:
            print(f"Exception during processing (Page {page_num + 1}): {str(e)}")

    pdf_document.close()
    for i, text in enumerate(text_list, start=1):
        print(f"Page {i}:\n{text}\n")
    return text_list
    

# Example: Replace 'your_fir.pdf' with the path to your FIR PDF
fir_pdf_path = gen_pdf_inp
result = extract_text_from_pdf(fir_pdf_path)

# Print the text from each page



# In[ ]:




