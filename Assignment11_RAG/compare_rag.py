import requests
import json
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_URL = "http://127.0.0.1:5000/test_agent"
TEST_PLANTS = ["Aloevera", "Corn"]

def get_care_card(plant_name, mode):
    """
    Modes: 'standard', 'web', 'rag'
    """
    payload = {
        "plant": plant_name,
        "mode": mode
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_comparison():
    print(f"\n{'='*70}")
    print(f"ULTIMATE SHOWDOWN: Standard vs. Web vs. RAG")
    print(f"{'='*70}\n")

    for plant in TEST_PLANTS:
        print(f" PLANT: {plant.upper()}")
        print("-" * 50)
        
        # 1. Standard (Pure GPT)
        print("[STANDARD GPT] (No Tools)...")
        res_std = get_care_card(plant, "standard")
        print(f"   Safety: {res_std.get('toxicity_and_safety', 'Error')}")
        print("")

        # 2. Web (Tavily Only)
        print("[WEB AGENT] (Searching Tavily)...")
        res_web = get_care_card(plant, "web")
        print(f"   Safety: {res_web.get('toxicity_and_safety', 'Error')}")
        print("")
        
        # 3. RAG (LanceDB)
        print("[RAG AGENT] (Local DB + Fallback)...")
        res_rag = get_care_card(plant, "rag")
        print(f"   Safety: {res_rag.get('toxicity_and_safety', 'Error')}")
        
        print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        requests.get("http://127.0.0.1:5000/")
        run_comparison()
    except:
        print("Error: Flask server not running. Run 'python app.py' first.")