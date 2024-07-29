import streamlit as st
import fitz  # PyMuPDF
import base64
import os
from streamlit_extras.stylable_container import stylable_container

# Ensure the tmp directory exists
os.makedirs("tmp", exist_ok=True)

# Function to highlight terms in the PDF using PyMuPDF
def highlight_pdf(file_path, search_terms):
    doc = fitz.open(file_path)
    for term in search_terms:
        term = term.strip()
        if term:
            for page in doc:
                text_instances = page.search_for(term)
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
    highlighted_pdf_path = os.path.join("tmp", "highlighted_document.pdf")
    doc.save(highlighted_pdf_path)
    doc.close()
    return highlighted_pdf_path

# Function to display the PDF file
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to render the compliance document overview
def compliance_document_overview():
    if st.button("Back"):
        st.session_state.page = 'homepage'
        st.experimental_rerun()

    st.header("Compliance Document Overview")

    search_col1, search_col2 = st.columns([9, 1])

    with search_col1:
        search_terms = st.text_input("Search for key terms/phrases", placeholder="Example: Restrictions", key="search_terms")

    with search_col2:
        with stylable_container(
            key="logo",
            css_styles="""
                button {
                display: block;
                margin-top: 28px;
                border-radius: 15px; /* Adjust this value for desired roundness */
                border: 2px solid grey; /* Example: 3px solid black border */
                }
                button:hover {
                background-color: #f0f0f0; /* Light grey background on hover */
                }
            """
        ):
            if st.button("🔍", key="search_button", use_container_width=True):
                if search_terms and any(search_terms):
                    st.session_state.searched = True

    # Section to display the compliance document with scrolling
    st.markdown("## Document Content")
    
    if st.session_state.selected_doc == "California Privacy Rights Act":
        pdf_file = './used docs/cppa_regs.pdf'
    elif st.session_state.selected_doc == "GDPR":
        pdf_file = "./used docs/CELEX_32016R0679_EN_TXT.pdf"
    elif st.session_state.selected_doc == "HIPAA":
        pdf_file = "./used docs/hipaa-simplification-201303.pdf"
    else:
        st.warning("No document selected")
        return

    if st.session_state.get('searched', False):
        highlighted_pdf_path = highlight_pdf(pdf_file, search_terms.split(','))
        display_pdf(highlighted_pdf_path)
    else:
        display_pdf(pdf_file)

# Main function to run the Streamlit page
def app():
    compliance_document_overview()


# if __name__ == "__main__":
#     app()
