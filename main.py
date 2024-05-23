# This is the main script for processing pathology reports in PDF format using various LLMs. 
# The script will load the PDF files, extract the text, and process the text using various LLMs 
# to extract structured variables and justification of the choice.
# The script will then save the structured variables and justification of the choice to a CSV file.
# The script uses the langchain_community library to interact with the LLMs and the pandas library to handle the data.
# The script is divided into several sections: loading the data, setting up the LLMs, 
# defining the prompt template, processing the reports, and saving the results to a CSV file.
# The script uses the Ollama class from the langchain_community.llms module to interact with the LLMs and
# the ChatPromptTemplate class from the langchain.prompts module to define the prompt template. 
# The script then processes the reports using the LLMs and saves the results to a CSV file.

import os
import glob
import random

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
import os
import pandas as pd

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import pandas as pd
import json

import subprocess

class path_variables(BaseModel):
    site: str = Field(description="site of the cancer as described in the pathology report")    
    laterality: str = Field(description="laterality of the cancer as described in the pathology report")
    histology: str = Field(description="histology of the cancer as described in the pathology report") 
    stage: str = Field(description="stage of the cancer as described in the pathology report") 
    grade: str = Field(description="grade of the cancer as described in the pathology report")
    behavior: str = Field(description="behavior of the cancer as described in the pathology report")



def load_random_pdfs(folder_path, num_files):
    # Get a list of all PDF files in the sub-folders and sub-subfolders
    pdf_files = glob.glob(os.path.join(folder_path, '**/*.pdf'), recursive=True)

    # Select a given number of PDF files randomly
    random_files = random.sample(pdf_files, num_files)

    return random_files




def process_pdfs(pdf_file_to_open, llm_model):

    chat = Ollama(model=llm_model, temperature=0.0)

    template_string = """You are a helpful assistant with knowlede in surgical pathology. \
        Your task is to process the given surgical pathology report and extract specific information and justify the extracted information in one sentence. \
        The reports are related to various cancers and have been converted into text using OCR from PDF files. \
        Therefore, ignore any OCR errors and focus on the content of the report. \
        For each report, fill the following categories "Site", "Laterality (left or right)", "Histology", "Stage (TNM format)", "Grade (Grade I (Low grade or well-differentiated), \
            Grade II (Intermediate grade or moderately differentiated), Grade III (High grade or poorly differentiated),  and Grade IV (High grade or undifferentiated))", "Behavior".\
        An example output is given here: \
        1. "Site": brain. \
        2. "Laterality": left. \
        3. "Histology": adenocarcinoma, as the report mentioned the histology of the tumor. \
        4. "Stage": T2N0Mx, as the tumor invaded the muscularis propria and the lymph nodes were not affected based on the report. \
        5. "Grade": III, as the tumor showed moderate differentiation based on the report. \
        6. "Behavior": malignant, as the tumor showed invasion of the surrounding tissues based on the report. \
        Here is the report {report}.
        Restrict your output to the six categories only that include "Site", "Laterality", "Histology", "Stage", "Grade", and "Behavior" and one sentence for the justification of the choice. \
        For the missing information, say "not provided".
        """

    prompt_template = ChatPromptTemplate.from_template(template_string)

    print("------------------------------------------------------------------------------------------------------")
    
    print(pdf_file_to_open)

    loader = PyPDFLoader(pdf_file_to_open)
    pages = loader.load()
    report = ' '.join(page.page_content for page in pages)
    
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")
    print("Input report after OCR:")
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")
    print(report)
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")

    llm_input_report = prompt_template.format_messages(report=report)
    extracted_data = chat.invoke(llm_input_report)

    print("First Stage Processing - LLM Extracted Data and Justification:") 
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")
    print(extracted_data)
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")

    return extracted_data, report

# this function will extract the structured variables from the LLM output

def extract_json_output(extracted_report_data, model):

    query_string = """ 
        DO NOT MAKE UP ANY INFORMATION. THIS IS A RETRIEVAL TASK ONLY. \
        Structure the information presented in a pathology report into JSON format. \
        The missing information should be represented as null. \
        DO NOT MAKE UP ANY INFORMATION. Here is the report \
        """ 
    parser = JsonOutputParser(pydantic_object=path_variables)

    prompt = PromptTemplate(
        template="Answer the user query. \n{format_instructions}\n{query}\n{report}",
        input_variables=["query", "report"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = Ollama(model=llm_model, temperature=0.0)

    chain = prompt | model | parser

    
    try:
        json_variables = chain.invoke({"query":query_string, "report": extracted_report_data})
    except Exception as e:
        print(f"An error occurred: {e}")
        json_variables = []

    
    return json_variables



if __name__ == "__main__":
    

    # folder_path = r'C:\Users\4475358\OneDrive - Moffitt Cancer Center\Moffitt\Datasets\Pathology Reports'
    # num_files = 1
    # pdf_files = load_random_pdfs(folder_path, num_files)
    
    
    # pdf_files = [r'C:\Users\4475358\OneDrive - Moffitt Cancer Center\Moffitt\Datasets\Pathology Reports\failures\TCGA-59-A5PD.pdf']
    # pdf_files = [r'D:\Works\SomeCodes\path-reports-variables\somereports\brain-1.pdf']
    # pdf_files = [r'D:\Works\SomeCodes\path-reports-variables\somereports\Kidney-2.pdf']
    
    
    # pdf_files = [r'D:\Works\SomeCodes\path-reports-variables\somereports\lung-1.pdf']
    # pdf_files = [r'D:\Works\SomeCodes\path-reports-variables\somereports\colon-1.pdf']
    # pdf_files = [r'D:\Works\SomeCodes\path-reports-variables\somereports\Uterus.pdf']
    pdf_files = [r'D:\Works\SomeCodes\path-reports-variables\somereports\kidney.pdf']
    print('Processing:', pdf_files)
    
    num_reports = len(pdf_files)

    # mistra, mixtral, gemma, command-r, llama3,  llama3:70b
    #llm_model = 'llama3:8b-instruct-q6_K'
    llm_model = "mixtral"


    print('We are processing a total of:', num_reports)

    #extracted_data_all = []
    #ids = []
    json_objects = []

    for i in range(num_reports):
        #pat_id = os.path.splitext(pdf_files[i])[0]
        pdf_file = pdf_files[i]
        extracted_data, report_ocr_text = process_pdfs(pdf_file, llm_model)
        
       
        json_variables = extract_json_output(extracted_data, llm_model)
        json_variables['pdf_file_name_path'] = pdf_file
        json_variables['ocr_text'] = report_ocr_text
        json_variables['llm_output'] = extracted_data
        #print("------------------------------------------------------------------------------------------------------")
        print("Second Stage Processing - Discrete Variables:") 
        print("------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------")
        print(json.dumps(json_variables, indent=4))
        json_objects.append(json_variables)
        #extracted_data_all.append(extracted_data)
        #subprocess.Popen([pdf_file],shell=True)
        #ids.append(pat_id)
 
    
    
    # Create a new DataFrame
    # new_df = pd.DataFrame()

    #new_df['case_submitter_id'] = ids
    #new_df['llm_output'] = pd.Series(extracted_data_all)

    # Save the DataFrame to a CSV file
    # save_file_name = llm_model + '_variables-from-PDFs.csv'
    # new_df.to_csv(save_file_name, index=False)
    
    # save_file_name = llm_model + '_variables-from-PDFs.json'
    # with open(save_file_name, 'w') as f:
    #     json.dump(json_objects, f, indent=4)