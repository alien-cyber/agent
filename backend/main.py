from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI
from pymongo import MongoClient
from pdf2image import convert_from_bytes

import os
from openai import OpenAI
import easyocr

reader = easyocr.Reader(['en'])  # Supports multiple languages



load_dotenv()


client_mongo = MongoClient(os.getenv("MONGO_URI"))
db = client_mongo['medical_app']
collection = db['user_data']


client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API"),
)
def get_completion(symptoms, history, file_info):
    if file_info:
        symptoms +=f"Medical report: {file_info}"
    prompt = f"""You are a doctor. A patient comes to you with the following symptoms: {symptoms}. 
            The patient also has the following medical history: {history}. 
            Diagnose the possible disease and prescribe HOME treatment. When mentioning which doctor to consult, please mention the type of doctor alone ,Example:dentist,cardiologist,etc,. 
            Return the response in the following JSON format AND Dont add any extra information in the before and after JSON. return JUST JSON:
            \n\n{{\n  disease: ____,\n  Home_treatment: ____,\n  TO_AVOID: ____,\n  DOCTOR_TO_CONSULT: ____\n}}"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="aaditya/Llama3-OpenBioLLM-70B",
        messages=messages,
        temperature=0.6,
    )
    return response.choices[0].message.content





app = FastAPI()

# Serve static files (if needed for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def serve_form():
    form_html = Path("templates/index.html").read_text()
    return HTMLResponse(content=form_html)

# Handle form submission
@app.post("/submit/")
async def process_form(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    symptoms: str = Form(...),
    history: str = Form(...),
    file: UploadFile = File(None)
):
    data = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "history": history,
        "file_name": file.filename if file else "No file uploaded"
    }
    
    # Process the file (if uploaded)
    # if file:
    #     file_path = f"uploads/{file.filename}"
    #     with open(file_path, "wb") as f:
    #         f.write(await file.read())
   
    if file:
        content_type = file.content_type

        if content_type.startswith("image/"):
            # Read text directly from image
            file_info = reader.readtext(file.file.read(), detail=0)

        elif content_type == "application/pdf":
            # Convert PDF to images and extract text from each page
            pdf_bytes = file.file.read()
            images = convert_from_bytes(pdf_bytes)
            text_list = ""

            for img in images:
                text = reader.readtext(img, detail=0)
                text_list+=text
            
            file_info += text_list

        else:
            file_info = None
    print(file_info)

    
    
    output = get_completion(symptoms, history,file_info)
    disease = firstindex(output, "disease:")
    home_treatment = firstindex(output, "Home_treatment:")
    to_avoid = firstindex(output, "TO_AVOID:")  
    doctor_to_consult = firstindex(output, "DOCTOR_TO_CONSULT:")
    print(output[disease:home_treatment])
    data["disease"] = output[disease+8:home_treatment]
    data["home_treatment"] = output[home_treatment+15:to_avoid]
    data["to_avoid"] = output[to_avoid+9:doctor_to_consult]
    data["doctor_to_consult"] = output[doctor_to_consult+18:]

    result = collection.insert_one(data)
    data["_id"] = str(result.inserted_id)
   
    



    return {"message": "Form submitted successfully", "data": data}


def firstindex(haystack: str, needle: str) -> int:
        if not needle:
            return 0
        
        # KMP Algorithm to preprocess the pattern
        def computeLPS(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        # Precompute LPS array
        lps = computeLPS(needle)
        
        i = j = 0  # Pointers for haystack and needle
        while i < len(haystack):
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            
            if j == len(needle):
                return i - j  # Found pattern, return start index
            
            if i < len(haystack) and haystack[i] != needle[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return -1