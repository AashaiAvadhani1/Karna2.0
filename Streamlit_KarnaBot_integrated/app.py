import streamlit as st
import signup_page
import signin_page
import homepage
import compliance_overview


import sys
import os

# Add the project root directory and the KarnaBot directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'KarnaBot')))


from KarnaBot import main_bot

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'signin'

def main():
    if st.session_state.page == 'signin':
        signin_page.app()
    elif st.session_state.page == 'signup':
        signup_page.app()
    elif st.session_state.page == 'homepage':
        homepage.app()
    elif st.session_state.page == 'compliance_overview':
        compliance_overview.app()
    elif st.session_state.page == 'karnabot':
        main_bot.deploy_bot()


if __name__ == "__main__":
    main()
