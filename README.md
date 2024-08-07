# Karna: A Compliance Checking Chatbot for Code Analysis

## **Table of Contents**
1. **Market Opportunity**
2. **Motivation**
3. **Introducing Karna**
4. **Data Background**
5. **Model Training**
6. **Model Inference**
7. **Prompt Testing**
8. **User Experience (UX)**
9. **Key Risks and Future Work**
10. **Meet the Team!**

## **Motivation**

Compliance is hard. Compliance Assessments take too much time to complete. Karna reads code and develops compliance assessments using AI.

The motivation behind Karna is to empower data scientists and software engineers by providing them with a robust tool that eliminates the need to consult legal experts for every privacy-related query within their code. Navigating the complexities of privacy legislation like GDPR, CCPA, HIPAA, and the Privacy Act of 1974 can be daunting and time-consuming. Traditionally, ensuring compliance requires frequent interactions with legal professionals, which can slow down development processes and increase costs. Karna addresses this challenge by offering an AI-driven solution that provides instant access to comprehensive legal insights and compliance checks. This allows developers to seamlessly integrate privacy considerations into their workflows, ensuring their code meets regulatory standards without the constant need for legal intervention. By streamlining the compliance process, Karna enables developers to focus more on innovation and less on legal complexities, ultimately accelerating the development of privacy-compliant software solutions, without worrying about global compliance barriers. 

[![logo.png](https://i.postimg.cc/9QLxftyF/logo.png)](https://postimg.cc/0zJGWStL)

## **Market Opportunity**


The data governance and privacy market has become increasingly important as organizations seek to comply with regulations, protect sensitive information, and gain competitive advantages through effective data management. This market intersects with competition law, as data protection and competition authorities apply different frameworks and pursue different policy objectives.

The data governance and privacy market is experiencing significant growth, with various projections for its total addressable market:
Liminal projects a TAM of $65 billion in 2027 for oveall compliance, with an expected compound annual growth rate (CAGR) of 19.3%.
Mordor Intelligence estimates the Data Governance Market size at $3.27 billion in 2024, projected to reach $8.03 billion by 2029, growing at a CAGR of 19.72%.
A more optimistic projection suggests the Data Governance Market could hit $13.92 billion by 2031.


## **Introducing Karna**
Karna is a cutting-edge GenAI solution tailored for data scientists and software engineers to ensure their code adheres to key privacy legislation, including GDPR (E.U. General Data Protection Regulation), CCPA (California Consumer Privacy Act), HIPAA (Health Insurance Portability and Accountability Act), and the Privacy Act of 1974. This innovative application features an intuitive chatbot, enabling users to ask detailed questions about the entirety of these legal documents. Additionally, Karna provides powerful tools to analyze and verify your code's compliance with specific laws, ensuring your projects meet all necessary regulatory standards. By leveraging Karna, you can streamline your compliance processes, reduce legal risks, and focus on building high-quality, privacy-compliant software.

## **Data Background**
To ensure Karna's effectiveness, we curated a comprehensive dataset comprising GDPR, CPA, various U.S. legislation regarding data privacy, and other related legal texts. This dataset was meticulously preprocessed to facilitate accurate analysis and compliance checking.

## **Model Training**
Training Karna involved:
1. **Data Processing:** Preparing legal texts and code snippets for comparative analysis.
2. **Knowledge Extraction:** Extracting key compliance rules and regulations from legal documents.
3. **Model Development:** Building algorithms to interpret and match code against compliance standards.
4. **Validation:** Validating the model's accuracy and reliability in compliance checking scenarios.

Detailed methodologies and scripts for training Karna are available in the `training` directory.

## **Model Inference**
Karna's inference capabilities allow it to analyze new code submissions in real-time. Users receive immediate feedback on compliance status and actionable insights on necessary modifications.

## **Prompt Testing**
We rigorously tested Karna using differently engineered prompts tailored to compliance checking, ensuring robust performance across various code samples and legal frameworks.

## **User Experience (UX)**
Karna offers an intuitive user interface designed for data scientists and developers. It supports interactive sessions where users can query specific legal clauses or receive guidance on compliance issues related to GDPR, CPA, and other regulations. Setup instructions and UX documentation are provided in the `UX` directory.

## **Key Risks and Future Work**
While Karna represents a significant advancement, ongoing challenges include adapting to evolving legal standards and expanding its knowledge base to cover additional regulations. Future work will focus on:
- Enhancing real-time compliance checking capabilities.
- Integrating more legal frameworks and updates.
- Improving user feedback mechanisms based on interaction data.

## **Meet the Team!**
Our dedicated team of AI researchers, legal experts, and software engineers collaborated to develop Karna:
- **Aashai Avadhani:** Lead ML and AI Expert
- **Ankit Gubiligari:** Data Scientist
- **Nakul Vadlamudi:** Data Scientist
- **Tyler Sapsford:** Data Scientist

---

We believe Karna will revolutionize compliance checking in software development and data science, providing valuable support to data scientists and developers, and lightening the load for legal professionals as well as programmers. For detailed documentation and to start using Karna, please refer to the resources available in this repository.


#### RAG pipeline using LangChain, Gemini pro, Faiss, Mistral, Anthropic
This is Karna, the first ever automated Privacy Risk Assessment Tool that reads code and understands privacy risks to your company. 
