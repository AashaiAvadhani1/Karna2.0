import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Function to handle the signup process
def signup_user(first_name, last_name, email, password):
    # Here you would implement the logic to add the user data to your database
    # For this example, let's just simulate a successful signup
    return True

# Function to compare passwords
def passwords_match(password, password2):
    return password == password2


def app():
    st.title('Sign Up')

    with st.form(key='signup_form'):
        first_name = st.text_input('First name')
        last_name = st.text_input('Last name')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        password2 = st.text_input('Re-enter password', type='password')

        
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
            submit_button = st.form_submit_button(label='Sign up')

        if submit_button:
            if not passwords_match(password, password2):
                st.error('Passwords do not match. Please try again.')
            else:
                # Add additional validation if necessary (e.g., check if email already exists)
                account_creation = signup_user(first_name, last_name, email, password)
                if account_creation:
                    st.success('Account created successfully!')
                    # You could redirect the user to the sign-in page or their dashboard here
                else:
                    st.error('An error occurred during sign up.')

    if st.button('Back'):
        st.session_state.page = 'signin'
        st.experimental_rerun()

# Uncomment the line below to test the signup page independently with `streamlit run signup.py`
# if __name__ == '__main__':
#     app()
