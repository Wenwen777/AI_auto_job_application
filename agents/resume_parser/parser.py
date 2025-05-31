import os
import json
import re
import pdfplumber
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def extract_text_from_pdf(file_path):
    ''' This function takes a pdf and creates a string object containing all of its text'''
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages)

def extract_clean_json(text):
    ''' Remove markdown-style code block wrappers like ```json ... ```'''
    json_block = re.search(r"\{[\s\S]+\}", text)
    if json_block:
        return json.loads(json_block.group(0))
    else:
        raise ValueError("No valid JSON block found")   
    

if __name__ == "__main__":
    text = extract_text_from_pdf(os.environ["RESUME_PATH"])

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

    prompt = ChatPromptTemplate.from_template(
    """
    Extract the following fields from the resume text:
    - full name
    - email
    - summary (Try to summarize the resume in 4 sentences)
    - skills (comma-separated)
    - job domains (e.g. software-engineering, mechanical-engineering, biotech)
    - years of experience
    - education
    - visa required (true/false)
    - preferred job location
    - role seniority (entry-level, senior)


    Resume Text:
    {context}

    Respond in JSON format.
    """
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    response = doc_chain.invoke({"context": [Document(page_content=text)]})
    clean_response = extract_clean_json(response)
    print(clean_response)

    # Save to a local JSON file
    with open("user_profile.json", "w") as f:
        json.dump(clean_response, f, indent=2)
