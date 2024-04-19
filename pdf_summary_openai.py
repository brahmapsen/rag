#install pyPDF2 
import openai,os
import PyPDF2

print("Summarize PDF")
openai.api_key = 'sk-'

pdf_summary_text = ""
pdf_file_path = "./vbc-lewin-group.pdf"
# pdf_file_path = "./CCSQ.pdf"
pdf_file = open(pdf_file_path, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

for page_num in range(len(pdf_reader.pages)):
   page_text = pdf_reader.pages[page_num].extract_text().lower()

   response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
         {"role": "system", "content": "You are a helpful research assistant."},
         {"role": "user", "content": f"Summarize this: {page_text}"},
      ],
   )
   
   page_summary = response["choices"][0]["message"]["content"]
   pdf_summary_text += page_summary + "\n"

   print('Summarized page ', page_num)
   if(page_num > 5):
      break
   
pdf_summary_file = pdf_file_path.replace(os.path.splitext(pdf_file_path)[1], "_summary.txt")
with open(pdf_summary_file, "w+") as file:
   file.write(pdf_summary_text)