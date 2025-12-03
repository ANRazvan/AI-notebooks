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

# LangChain and Agent imports
from pydantic import BaseModel, Field
from tavily import TavilyClient
from loguru import logger
# Added SystemMessage here
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as langchain_tool

app = Flask(__name__)

# API Keys - in production, use environment variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Initialize clients
TAVILY_CLIENT = TavilyClient(api_key=TAVILY_API_KEY)
OPENAI_MODEL = "gpt-4o-mini"

# Pydantic Model for Plant Care Card
class PlantCareCard(BaseModel):
    plant_name: str = Field(..., description="The name of the plant")
    latin_name: str = Field(..., description="The Latin name of the plant")
    watering_schedule: str = Field(..., description="How often to water the plant")
    sunlight_requirements: str = Field(..., description="The amount of sunlight needed")
    additional_tips: str = Field(..., description="Any additional care tips for the plant")

# System prompt for the agent
SYSTEM_PROMPT = """You are Dr. Botanica, a world-class botanist and horticulture expert with over 30 years of experience. 
You specialize in plant care, cultivation techniques, and sustainable gardening practices.

Your expertise includes:
- Deep knowledge of plant taxonomy and botanical families
- Practical experience with indoor and outdoor cultivation
- Understanding of plant physiology and growth requirements
- Expertise in organic pest management and plant health
- Knowledge of traditional and modern propagation techniques

TASK: Generate a comprehensive Plant Care Data Card for the given plant.

REQUIREMENTS:
1. Provide accurate, scientific information based on botanical research
2. Include practical, actionable care instructions
3. Consider both novice and experienced gardeners
4. Mention any safety concerns (toxicity, allergens)
5. Provide culturally relevant information when applicable
6. Use clear, professional language

OUTPUT FORMAT:
You must respond with a valid JSON object matching the PlantCareCard schema.
All fields must be completed with accurate, detailed information.
"""

# LangChain Tool for Web Search
@langchain_tool
def plant_care_tool(plant_name: str) -> str:
    """
    Searches the web for plant care information using Tavily API.
    
    Args:
        plant_name (str): The name of the plant to get care instructions for.
    Returns:
        str: Raw web search results about the plant care.
    """
    if not plant_name:
        raise ValueError("plant_name is required")
    
    search_query = f"{plant_name} plant care watering sunlight requirements tips latin name"
    
    try:
        search_results = TAVILY_CLIENT.search(
            query=search_query,
            search_depth="advanced",
            max_results=5
        )
        
        formatted_results = []
        for i, result in enumerate(search_results.get('results', []), 1):
            formatted_results.append(
                f"Result {i}:\n"
                f"Source: {result['url']}\n"
                f"Content: {result['content']}\n"
            )
        
        search_context = "\n".join(formatted_results)
        logger.info(f"Found {len(search_results.get('results', []))} search results for {plant_name}")
        
        return search_context if search_context else "No search results found."
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Error during web search: {str(e)}"

# Initialize LangChain agent at startup
print("Initializing LangChain agent...")
llm_model = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

# Create agent with tools
# FIXED: Removed messages_modifier completely
agent = create_react_agent(
    model=llm_model,
    tools=[plant_care_tool]
)

print("Agent initialized successfully!")

# Define the same Lightning Module class from your notebook
class PretrainedLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

# Plant class names (update these based on your dataset)
CLASS_NAMES = [
    'Aloevera', 'Banana', 'Bilimbi', 'Cantaloupe', 'Cassava', 'Coconut', 'Corn', 'Cucumber', 
    'Curcuma', 'Eggplant', 'Galangal', 'Ginger', 'Guava', 'Kale', 'Longbeans', 'Mango', 
    'Melon', 'Orange', 'Paddy', 'Papaya', 'Peper Chili', 'Pineapple', 'Pomelo', 'Shallot', 
    'Soybeans', 'Spinach', 'Sweet Potatoes', 'Tobacco', 'Waterapple', 'Watermelon'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
def load_model():
    base_model = models.efficientnet_b0(weights=None)
    num_classes = len(CLASS_NAMES)
    num_ftrs = base_model.classifier[1].in_features
    base_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Load the checkpoint
    # Ensure map_location handles CPU/GPU correctly automatically
    model = PretrainedLightningModule.load_from_checkpoint(
        'best-pretrained-model.ckpt',
        model=base_model,
        num_classes=num_classes
    )
    model.eval()
    return model

# Load model at startup
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()
model = model.to(device)
print(f"Model loaded successfully on {device}")

@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            top3_predictions = [
                {
                    'class': CLASS_NAMES[idx.item()],
                    'confidence': prob.item() * 100
                }
                for prob, idx in zip(top3_prob[0], top3_idx[0])
            ]
        
        plant_care_card = get_plant_care_info_agent(predicted_class)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence_score, 2),
            'top3': top3_predictions,
            'plant_info': plant_care_card
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def get_plant_care_info_agent(plant_name: str) -> dict:
    """Get detailed plant care information using LangChain agent with web search"""
    try:
        user_input = f"Generate a Plant Care Data Card for: {plant_name}"
        
        # FIXED: Inject System Message here
        response = agent.invoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT), 
                HumanMessage(content=user_input)
            ]
        })
        
        # Extract the last message content (the AI's final answer)
        agent_output = response["messages"][-1].content
        
        # Parse into JSON
        structured_llm = llm_model.with_structured_output(PlantCareCard)
        care_card = structured_llm.invoke(f"Extract the plant data from this text into the requested JSON format:\n\n{agent_output}")
        
        formatted_info = f"""**{care_card.plant_name}**
        **Latin Name:** {care_card.latin_name}
        
        **Watering Schedule:**
        {care_card.watering_schedule}
        
        **Sunlight Requirements:**
        {care_card.sunlight_requirements}
        
        **Additional Tips:**
        {care_card.additional_tips}
        """
        return formatted_info
    
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return f"**{plant_name}**\n\nError: Unable to fetch detailed care information at this time.\n({str(e)})"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)