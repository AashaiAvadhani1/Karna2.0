import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import hashlib

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Hardcoded credentials
users = {
    "aashai123@gmail.com": hash_password("password123")
}

def authenticate_user(username, password):
    if username in users and users[username] == hash_password(password):
        return True
    else:
        return False

def app():
    with stylable_container(
            key="logo",
            css_styles="""
                img {
                display: block;
                margin-left: 200px;
                width: 100%;
                border-radius: 15px; /* Adjust this value for desired roundness */
                border: 3px solid #000; /* Example: 3px solid black border */
    }"""):
        st.image('logo.png', width=300)  # Replace 'logo.png' with the path to your logo image
    st.markdown("<h3 style='text-align: left; color: black;'>Sign In</h1>", unsafe_allow_html=True)

    # Sign In Form
    with st.form(key='sign_in_form'):
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        
        # Centering the Sign in button with stylable_container
        with stylable_container(
            key="sign_in_button",
            css_styles="""
                button {
                width: 100%; /* Make button full width of the container */
                background-color: #1E88E5; /* Blue background color */
                color: white; /* White text color */
                border: none; /* No border */
                border-radius: 5px; /* Rounded corners */
                padding: 10px 0; /* Padding to increase the height */
                font-size: 16px; /* Larger font size */
                font-weight: bold; /* Bold font */
            }
            button:hover {
                background-color: #1565C0; /* Slightly darker blue on hover */
            }
            """
        ):
            submit_button = st.form_submit_button('Sign in')

        if submit_button:
            if authenticate_user(email, password):
                st.success('Login Successful!')
                st.session_state.page = 'homepage'
                st.experimental_rerun()
            else:
                st.error('Login Failed: Incorrect username or password.')

    col1,col2 = st.columns([1,1])
    
    with col1:
        with stylable_container(
            key="forgot_button",
            css_styles="""
                button {
                    background-color: transparent; /* No background color */
                    color: grey; /* Grey text */
                    border: none; /* No border */
                    border-radius: 0; /* Remove any border radius */
                    text-decoration: underline; /* Underlined text to mimic a link */
                    padding: 5px 10px; /* Optional: Adjust padding to fit your design needs */
                }
                button:hover {
                    background-color: transparent; /* Ensure no background on hover */
                    color: darkgrey; /* Slightly darker color on hover for feedback */
                }
                """,
        ): 
            if st.button('Forgot password'):
                st.session_state.page = 'forgot_password'
                st.rerun()
    with col2: 
        with stylable_container(
            key="sign_up_button",
            css_styles="""
                button {
                    background-color: transparent; /* No background color */
                    color: #1E88E5;
                    border: none; /* No border */
                    border-radius: 0; /* Remove any border radius */
                    text-decoration: underline; /* Underlined text to mimic a link */
                    padding: 5px 10px; /* Optional: Adjust padding to fit your design needs */
                    margin-left: 260px;
                }
                button:hover {
                    background-color: transparent; /* Ensure no background on hover */
                    color: darkgrey; /* Slightly darker color on hover for feedback */
                }
                """,
        ): 
            if st.button('Sign up'):
                st.session_state.page = 'signup'
                st.rerun()

    # GitHub sign-in
    st.markdown("<h6 style='text-align: center; color: black;'>Sign in with Github</h6>", unsafe_allow_html=True)
    with stylable_container(
            key="github_logo",
            css_styles="""
                img {
                display: block;
                margin-left: 300px;
                width: 100%;
                border-radius: 15px; /* Adjust this value for desired roundness */
                border: 3px solid #000; /* Example: 3px solid black border */
    }"""):
        st.image('github_logo.png', width=100)

# Uncomment the line below to test the sign-in page independently with `streamlit run yourscript.py`
# if __name__ == '__main__':
#     app()
