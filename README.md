# AI-notebooks
Jupyter notebooks containing some assignments solved during an internship.

# Description of the assignments
## Assignment 1 
- Accomodation with python language
## Assignment 2 - numpy
- New York AirBnB dataset
- plotting and data analyzing
## Assignment 3 - data processing
- Titanic Dataset
- Data exploration, outliers, missing values
- encoding, scaling,
- feature engineering, feature selection, train-test splitting
- pipelines
## Assignment 4 - Regression
- California Housing Dataset - Linear Regression
- Predicting House Prices
- Data exploration & cleaning
- Feature engineering
- Encoding, scaling
- Modeling, training and evaluation
## Assignment 5 - Classification
- Heart Disease Dataset - Classification
- Data exploration & preprocessing
- Feature analysis
- Model implementation & evaluation
## Assignment 6 - model evaluation in supervised learning
- Mushroom dataset - Classification - Logistic Regression, SVM, Decision Tree
- Data exploration & preprocessing
- Model Selection & Evaluation
- K-Folds Cross Validation
- Grid Search
## Assignment 7 - unsupervised learning
- Mall Customer Dataset / Credit Card Dataset
- Preprocessing
- Dimensionality reduction ( PCA, TSNE )
- Clustering ( KMeans, Gaussian Mixture Model, DBScan, Mean Shift Clustering, Agglomerative Clustering)
## Assignment 8 - Neural Networks
- Energy Efficiency Dataset, Wine Dataset, Letters Dataset
- Regression and Classification
- Feed Forward neural networks
## Assignment 9 - Pytorch & PytorchLightning
- Plant Types Dataset - image classification
- Convolutional Neural Networks, SympleCNN
- Data transformations, dataloaders
- Pretrained models ( EfficientNet, ResNet18)
- Pytorch Lighting module - training, validation and testing steps
- model callbacks - early stopping, model checkpoint, csv logger

## Assignment 10 - LLM Integration
- Continued on assignment 9
- Built a LangChain Agent using gpt-4o-mini
- Tavily API Tool for Web Search ( information about the plant, passed as context to the LLM Request)
- Implemented a PlantCareCard , a pydantic BaseModel used for formatting the result

## Assignment 11 - RAG Architecture
- Continue on assignment 10
- WebBaseLoader for loading the given URLs 
- RecursiveCharacterSplitter for splitting in chunks
- Implemented a vector store using LanceDB
- Implemented 3 agents for the LLM call using different tools:
    - One without any tools
    - One with Tavily API and VectorStore 
    - One with TavilyAPI
- Added a new field to the PlantCareCard (Toxicity to animals) 
- Implemented code for the agents comparison, for 3 different plants
- The advantage of having a VectorStore was seen in the responses, in the toxicity field
- Without any tools, gpt-4o-mini didn`t know about some plants being poisonous, whilst the other had relevant information
