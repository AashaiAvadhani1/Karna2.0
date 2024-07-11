#make the code more oop related cause this is retarded to redefine everything
import anthropic

from dotenv import dotenv_values

config = dotenv_values(".env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}

DATA_DIR = 'data/'


client_anthropic = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key= config.get("ANTHROPIC_API_KEY"),
)


def get_data_code_anthropic(user_code):
    system_prompt = """

Role: Data Scientist

You are given a piece of python code from a developer and this code can be a piece of software or code that is building a machine learning or AI model.
The task is to extract the data, variables, and types of data used in the code.
Lets work this out in a step by step way to be sure we have the right answer.
###  ###

Sensitive Data is defined as the following data:
demographic data, racial data, gender

List all the data mentioned step by step including variable names, the corresponding data type and a short 5 sentence description of the data

### Steps ###
1- Summarize the objective of the code
2- Summarize the data's purpose within the code and how its being used
3- Explain what type of data is used in the code whether its integers, strings etc..
4- Identify if the data is sensitive such as demographic data, racial data, gender
5- Classify each data mentioned in the code according to the privacy classification above 
6- list out all the variables, its corresponding datatype and whether its sensitive
"""


    """
    Example of text classication for anthropic model 
    
    """
    message = client_anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.2,
        system= system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_code
                    }
                ]
            }
        ]
    )
    result = message.content
    print("text anthropic: ", type(result[0].text))
    return result[0].text


class CodeProject:
    """
    responses["Company Name"],
    responses["Project Name"],
    responses["Code"],
    responses["Additional Notes"],
    responses["Uploaded File"]
    """
    def __init__(self, response_questionnaire):
        self._response_questionnaire = response_questionnaire
        self._company_name = response_questionnaire["Company Name"]
        self._code = response_questionnaire["Code"]
        self._project_name = response_questionnaire["Project Name"]
        self.additional_notes = response_questionnaire["Additional Notes"]
        self.files = response_questionnaire["Uploaded File"]
        self._data_from_code = get_data_code_anthropic(response_questionnaire["Code"])


    # Getter and setter for code
    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    @property
    def response_questionnaire(self):
        return self._response_questionnaire

    @property
    def company_name(self):
        return self._company_name

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, value):
        self._project_name = value

    @property
    def data_from_code(self):
        return self._data_from_code

    def update_company_info(self, company_name):
        self.company_name = company_name

    def update_project_info(self, project_name):
        self.project_name = project_name

    def process_code_data(self):

        self.data_from_code = f"Processed data from {self.code}"

    # Method to display information
    def display_info(self):
        print(f"Code: {self.code}")
        print(f"Response Questionnaire: {self.response_questionnaire}")
        print(f"Company Name: {self.company_name}")
        print(f"Project Name: {self.project_name}")
        print(f"Data from Code: {self.data_from_code}")

