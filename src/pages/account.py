import streamlit as st
from streamlit.elements import markdown

# Function to display the user account page with a table
def display_account_page(user_account):
    st.title("Account Information")

    st.markdown(
        """
        <style>
        .account-table {
            width: 100%;
            border-collapse: collapse;
        }
        .account-table th, .account-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .account-table th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #f2f2f2;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <table class="account-table">
            <tr>
                <th>Name</th>
                <td>{user_account.name}</td>
            </tr>
            <tr>
                <th>Company</th>
                <td>{user_account.company}</td>
            </tr>
            <tr>
                <th>Email</th>
                <td><a href="mailto:{user_account.email}">{user_account.email}</a></td>
            </tr>
            <tr>
                <th>Phone</th>
                <td>{user_account.phone}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )
