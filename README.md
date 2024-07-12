# LLMFoundationModels
LLM GenAI Foundation CUDA Langchain Models
Lot of training models.

Generating Product Description SEO CTA Twitter Keywords from Images using GenAI Multimodal API (OpenAI gpt-4o)

Objective: Show a basic model to generate product description, SEO CTA and Twitter keywords from product images using a GenAI multimodal API, OpenAI gpt-4o

Model Overview:
1.	Product image (retail products) was passed as part of the API call and the prompt was asked to return a description
2.	The short description returned was then further improved by another API call with a prompt
3.	Different prompt API calls were made to then extract different types of keywords
4.	Finally, HTML tags with all information was produced
5.	For 13 images, API calls cost ~ $3 for all training/testing. GPU cost ~ $5 (2 Q RTX 8000 48GB)
6.	This model can be scaled for thousands to millions of product images, depending on budget

Improvements / Alternative:
1.	The model is basic and can be improved by using other GenAI API providers, more detailed prompts, performing DL NLP on text to get better keywords
2.	The alternative is to tune your own model using custom product data on a open source LLM (llama, etc)
a.	That will require initial training cost but can be customized for industry/product specific outputs and minimal API call costs once model is productionized
b.	Trade-off will have to be calculated, as GenAI API providers are regularly reducing costs and the scale of their training is not easy to replicate given the high cost
3.	I still prefer training custom model using product specific data to get more relevant output and control the cost/outpt

Sample Images with output HTML files:


