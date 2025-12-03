from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import pytorch_lightning as pl
import torchmetrics
import os
import json
import sys


# Imports for Agent & RAG
from pydantic import BaseModel, Field
from tavily import TavilyClient
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as langchain_tool

# RAG specific imports
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

app = Flask(__name__)

# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
TAVILY_CLIENT = TavilyClient(api_key=TAVILY_API_KEY)
OPENAI_MODEL = "gpt-4o-mini"

# --- 1. Define Pydantic Model ---
class PlantCareCard(BaseModel):
    plant_name: str = Field(..., description="The name of the plant")
    watering_schedule: str = Field(..., description="How often to water the plant")
    sunlight_requirements: str = Field(..., description="The amount of sunlight needed")
    toxicity_and_safety: str = Field(..., description="Details about toxicity to pets/humans and safety precautions")
    additional_tips: str = Field(..., description="Any additional care tips for the plant")

# --- 2. Define System Prompt ---
SYSTEM_PROMPT = """You are Dr. Botanica, a horticulture expert.
TASK: Generate a Plant Care Data Card.
STRATEGY:
1. FIRST, check the 'search_plant_care_guides' tool (if available).
2. If unavailable, check 'plant_care_tool' (if available).
3. If NO tools are available, rely on your internal knowledge.
OUTPUT: Respond with a valid JSON object matching the PlantCareCard schema.
"""

# --- 3. Define Tools ---

@langchain_tool
def plant_care_tool(plant_name: str) -> str:
    """Searches the live web for plant care information using Tavily API."""
    if not plant_name: return "Error: plant_name required"
    try:
        search_results = TAVILY_CLIENT.search(query=f"{plant_name} care toxicity", max_results=3)
        return str(search_results.get('results', []))
    except Exception as e:
        return f"Error: {str(e)}"

# RAG Tool Setup
retriever_tool = None
if os.path.exists("./lancedb_data"):
    print("[INFO] Loading LanceDB Knowledge Base...")
    try:
        db = lancedb.connect("./lancedb_data")
        table = db.open_table("plant_care_guides")
        vector_store = LanceDB(connection=table, embedding=OpenAIEmbeddings())
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retriever_tool = create_retriever_tool(
            retriever,
            name="search_plant_care_guides",
            description="Searches verified guides. ALWAYS use this first."
        )
    except Exception as e:
        print(f"[WARN] Could not load table: {e}")
else:
    print("[WARN] No LanceDB folder found. RAG disabled.")

# --- 4. Initialize THREE Agents ---
print("[INFO] Initializing Agents...")
llm_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# Agent 1: RAG (DB + Web)
tools_rag = [retriever_tool, plant_care_tool] if retriever_tool else [plant_care_tool]
agent_rag = create_react_agent(model=llm_model, tools=tools_rag)

# Agent 2: Web Only (Web)
tools_web = [plant_care_tool]
agent_web = create_react_agent(model=llm_model, tools=tools_web)

# Agent 3: Standard (Pure Internal Knowledge - NO Tools)
tools_standard = [] 
agent_standard = create_react_agent(model=llm_model, tools=tools_standard)

print("[INFO] Agents Ready!")

# --- 5. PyTorch Logic ---
class PretrainedLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
    def forward(self, x): return self.model(x)

CLASS_NAMES = ['Aloevera', 'Banana', 'Bilimbi', 'Cantaloupe', 'Cassava', 'Coconut', 'Corn', 'Cucumber', 'Curcuma', 'Eggplant', 'Galangal', 'Ginger', 'Guava', 'Kale', 'Longbeans', 'Mango', 'Melon', 'Orange', 'Paddy', 'Papaya', 'Peper Chili', 'Pineapple', 'Pomelo', 'Shallot', 'Soybeans', 'Spinach', 'Sweet Potatoes', 'Tobacco', 'Waterapple', 'Watermelon']

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

if os.path.exists('best-pretrained-model.ckpt'):
    print("[INFO] Loading PyTorch Model...")
    try:
        base_model = models.efficientnet_b0(weights=None)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, len(CLASS_NAMES))
        model = PretrainedLightningModule.load_from_checkpoint('best-pretrained-model.ckpt', model=base_model, num_classes=len(CLASS_NAMES))
        model.to(device).eval()
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
else:
    print("[WARN] No model checkpoint found.")

# --- 6. Helper Generator (Updated for 3 Modes) ---
def generate_care_card(plant_name, mode="rag"):
    """
    mode: 'rag' (DB+Web), 'web' (Web only), 'standard' (No tools)
    """
    try:
        if mode == 'rag':
            agent = agent_rag
            print(f"   --> Using RAG Agent for {plant_name}")
        elif mode == 'web':
            agent = agent_web
            print(f"   --> Using WEB Agent for {plant_name}")
        else: # standard
            agent = agent_standard
            print(f"   --> Using STANDARD Agent for {plant_name}")

        response = agent.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f"Plant: {plant_name}")]})
        content = response["messages"][-1].content
        structured = llm_model.with_structured_output(PlantCareCard).invoke(f"Extract JSON from: {content}")
        return structured.dict()
    except Exception as e:
        return {"error": str(e)}

# --- 7. Routes ---
@app.route('/test_agent', methods=['POST'])
def test_agent():
    """Comparison Endpoint: Simulate RAG vs Web vs Standard"""
    data = request.json
    plant = data.get('plant')
    mode = data.get('mode', 'rag') # Default to rag
    return jsonify(generate_care_card(plant, mode=mode))

@app.route('/predict', methods=['POST'])
def predict():
    """Production Endpoint: Image -> Class -> RAG"""
    if not model: return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    img = Image.open(request.files['file']).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = torch.softmax(model(tensor), dim=1)
        conf, idx = torch.max(preds, 1)
        plant = CLASS_NAMES[idx.item()]
    
    # Default to RAG for production
    card = generate_care_card(plant, mode='rag')
    
    formatted_info = f"""**{card.get('plant_name')}**
**Watering:** {card.get('watering_schedule')}
**Safety:** {card.get('toxicity_and_safety')}
"""
    return jsonify({'prediction': plant, 'confidence': conf.item()*100, 'plant_info': formatted_info})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)