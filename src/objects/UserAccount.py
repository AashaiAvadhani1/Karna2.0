class UserAccount:
    def __init__(self, name, company, email, phone):
        self.name = name
        self.company = company
        self.email = email
        self.phone = phone

    def __repr__(self):
        return f"UserAccount(name={self.name}, company={self.company}, email={self.email}, phone={self.phone})"

    @staticmethod
    def create_user_account(name, company, email, phone):
        return UserAccount(name, company, email, phone)