from fastapi import FastAPI, Form, File, UploadFile,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel
import os
import io

from pymongo import MongoClient
from pdf2image import convert_from_bytes
import torch
from PIL import Image
import torchvision.transforms as T

from openai import OpenAI
import easyocr

from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool

from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    FunctionDeclaration,
    GenerateContentConfig,
    Tool,
    ToolConfig,

)
    

# vertexai.init(project="gemini-449109", location="us-central1")
from google import genai
from fastapi.middleware.cors import CORSMiddleware



MODEL_ID = "gemini-2.0-flash-001"  # @param {type: "string"}

reader = easyocr.Reader(['en'])  # Supports multiple languages


aiclient = genai.Client(vertexai=True, project="Your-project-id", location="us-central1")
load_dotenv()


client_mongo = MongoClient(os.getenv("MONGO_URI"))
db = client_mongo['medical_app']
collection = db['user_data']


client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API"),
)
def get_diagnosis_from_api(symptoms, history="", file_contents_if_uploaded="", skin_conditioon_if_given=""):
    if skin_conditioon_if_given:
        symptoms += f"Skin condition: {skin_conditioon_if_given}. "
    
    if file_contents_if_uploaded:
        symptoms +=f"Medical report: {file_contents_if_uploaded}"
   
   
   
    # prompt = f"""You are a doctor. A patient comes to you with the following symptoms: {symptoms}. 
    #         The patient also has the following medical history: {history}. 
    #         Diagnose the possible disease and prescribe HOME treatment. When mentioning which doctor to consult, please mention the type of doctor alone ,Example:dentist,cardiologist,etc,. 
    #         Return the response in the following JSON format AND Dont add any extra information in the before and after JSON. return JUST JSON:
    #         \n\n{{\n  disease: ____,\n  Home_treatment: ____,\n  TO_AVOID: ____,\n  DOCTOR_TO_CONSULT: ____\n}}"""
    
    
    
    prompt = f"""You are a doctor. A patient comes to you with the following symptoms: {symptoms}. 
            The patient also has the following medical history: {history}. 
            Diagnose the possible disease and prescribe HOME treatment. When mentioning which doctor to consult, please mention the type of doctor alone ,Example:dentist,cardiologist,etc,. 
           """
    
    
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="aaditya/Llama3-OpenBioLLM-70B",
        messages=messages,
        temperature=0.6,
    )
    return response.choices[0].message.content



model = torch.load('./skin-model-pokemon.pt', map_location=torch.device('cpu'), weights_only=False)
device = torch.device('cpu')
model.to(device)



classes = ['acanthosis-nigricans',
                'acne',
                'acne-scars',
                'alopecia-areata',
                'dry',
                'melasma',
                'oily',
                'vitiligo',
                'warts']

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    file: UploadFile = File(None),
    photo: UploadFile = File(None)

):
    data = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "history": history,
        "file_bytes": file.filename if file else "No file uploaded",
        "photo": photo.filename if photo else "No photo uploaded"

    }
    
    # # Process the file (if uploaded)
    if file:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
    if photo:
        photo_path = f"uploads/{photo.filename}"
        with open(photo_path, "wb") as f:
            f.write(await photo.read())
   
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
    if photo:
  
        image_bytes = await photo.read()
        
        # Convert to PIL image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get the transforms
        tr = get_transforms()

        # Process image without saving to disk
        result = predict(model, img, tr, classes)
    else: 
        result = None
    print(result,photo.filename)

    
    
    output = get_diagnosis_from_api(symptoms, history,file_info, result)
    disease = firstindex(output, "disease:")
    home_treatment = firstindex(output, "Home_treatment:")
    to_avoid = firstindex(output, "TO_AVOID:")  
    doctor_to_consult = firstindex(output, "DOCTOR_TO_CONSULT:")
    print(output[disease:home_treatment])
    data["disease"] = output[disease+8:home_treatment]
    data["home_treatment"] = output[home_treatment+15:to_avoid]
    data["to_avoid"] = output[to_avoid+9:doctor_to_consult]
    data["doctor_to_consult"] = output[doctor_to_consult+18:]


#   THE BELOW IS MONGODB STORAGE CODE
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


def predict(model, img, tr, classes):
    img_tensor = tr(img)
    out = model(img_tensor.unsqueeze(0))
    pred, idx = torch.max(out, 1)
    return classes[idx]

def get_transforms():
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.ToTensor())
    return T.Compose(transform)








class ChatRequest(BaseModel):
    user_id: str
    message: str


chat_histories = {}  # user_id -> list of messages





def chat_user(  message: str = Form(...)
              ):
    return {"message": message,"action":"chat"}


    
def get_file_from_user():
    return {"action":"request_file_upload"}


get_diagnosis = FunctionDeclaration(
    name="get_diagnosis",
    description="Get the disease diagnosis and treatment of the user.Use this tool even if you want to tell about the disease to the user",
    parameters={
        "type": "OBJECT",
        "properties": {
            "symptoms": {"type": "STRING", "description": "Symptoms said by the user"},
            "history": {"type": "STRING", "description": "History of the user if not present then give empty string"},
            # "file_contents_if_uploaded": {"type": "STRING", "description": "medical File contents of the user if uploaded"},
            "skin_conditioon_if_given": {"type": "STRING", "description": "Skin condition of the user if given,if not given then give empty string"},
        },
    },
)

get_file_from_user_if_needed = FunctionDeclaration(
    name="get_file_from_user_if_needed",
    description="A function to retrieve the user's medical file. It should be called by you if the user says they have a medical file, after you have asked whether they have one or not",
    parameters={
        "type": "OBJECT",
        "properties": {
            "message": {"type": "STRING", "description": "Message to the user asking for the specific medical file"},
            
        },
    },
)


chat_with_user=FunctionDeclaration(
      name="chat_with_user",
      description="This should be used if you want to ask the user about something.(If you want to make the user upload a medical file first ask him wheather he has a file or not via this)",
      parameters={
        "type": "OBJECT",
        "properties": {
            "message": {"type": "STRING", "description": "Chat Message you want to give"},
            
        },
    },
)


medical_chat_app_tool = Tool(
    function_declarations=[
        get_diagnosis,
        get_file_from_user_if_needed,
        chat_with_user,
        
    ],
)



config = GenerateContentConfig(temperature=0, tools=[medical_chat_app_tool])


config.tool_config = ToolConfig(
    function_calling_config=FunctionCallingConfig(
        mode=FunctionCallingConfigMode.ANY,  # The default model behavior. The model decides whether to predict a function call or a natural language response.
    )
)
     
  


function_handler = {
   "get_diagnosis": get_diagnosis_from_api,
   "get_file_from_user_if_needed": get_file_from_user,
   "chat_with_user": chat_user,
}





def send_chat_message(message):
 

    message += """
      You are a central Medical AI with multiple tools to respond to a user query use that tools to answer . 
    """

    response = aiclient.models.generate_content(
    model=MODEL_ID,
    contents=message,
    config=config,
)
    


    function_call = response.function_calls[0]

    action="chat"

        # Check for a function call or a natural language response
    if function_call.name in function_handler.keys():
            # Extract the function call name
        function_name = function_call.name
        if function_name=="get_diagnosis":
        
            params = {key: value for key, value in function_call.args.items()}
            function_api_response = function_handler[function_name](**params)
            return {"response": function_api_response,"action":action}
        elif function_name=="chat_with_user":
        
            params = {key: value for key, value in function_call.args.items()}

            return {"response": params["message"],"action":action}
        elif function_name == "get_file_from_user_if_needed":
            return {"response": "Please upload your medical file.","action":"request_file_upload"}
        

    return {"response": "error","action":action}



@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id
    message = request.message

    # Check if the user ID is valid (you can implement your own validation logic)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user ID")
      
    if user_id not in chat_histories:
        # file_prev_info=collection.find_one({"email": user_id}, {"file_info": 1, "_id": 0,"name":0,"allergies":1,"existing_medical_condition":1})
        chat_histories[user_id] = [{"role": "system", "content": "You are a helpful assistant. Previous medical data of the user is: "}]

    # Append user's message
    chat_histories[user_id].append({"role": "user", "content": message})

    # Process the chat message and get a response from the Gemini model
    full_context = "\n".join(
    f"{msg['role'].capitalize()}: {msg['content']}"
    for msg in chat_histories[user_id]
)
   
    data = send_chat_message(full_context)

    return {"response": data["response"],"action":data["action"]}
   


@app.post("/file_upload")
async def file_upload(file: UploadFile = File(...), user_id: str = Form(...)):
    

    
    # file_path = f"uploads/{file.filename}"
    # with open(file_path, "wb") as f:
    #     f.write(await file.read())
  
   

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
            file_info = ""

    
    chat_histories[user_id].append({"role": "user", "content": "UPLOADED_FILE_INFO: "+str(file_info)})
    
#     collection.update_one(
#     {"email": user_id},
#     {"$push": {"file_info": file_info}}
# )
   
   
    # Call the function to get diagnosis from API using the uploaded file contents
    full_context = "\n".join(
    f"{msg['role'].capitalize()}: {msg['content']}"
    for msg in chat_histories[user_id]
)

    response = send_chat_message(full_context)
    chat_histories[user_id].append({"role": "assistant", "content": response})


    return {"message": "File uploaded successfully", "response": response}
  



  
