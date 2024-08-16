## Introduction

MentalHealthBot is a chatbot that provides mental health resources and support to users. The chatbot is designed to help users who are struggling with mental health issues by providing them with information, resources, and support. This is a personal project that I created to demonstrate how chatbots can be used to support mental health and well-being.

## Usage

To use the chatbot, you should use Docker to run the chatbot locally. You can run the chatbot by following these steps:

1. Clone the repository to your local machine.
2. Navigate to Discord Developer Portal and create a new application.
3. Put Hugging Face token and Discord token in .env file.
4. Run the following command to build the Docker image:
```
docker build -t mentalhealthbot .
```
1. Run the following command to start the Docker container:
```
docker run -it mentalhealthbot
```
1. The chatbot will start running and you can interact with it by sending messages in the chat with slash commands.
```
/ask <your question>
```

## Training Data

The chatbot is trained on a dataset of mental health resources which is publicly available on Kaggle: [NLP Mental Health Conversations](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations). The training data is stored in the `data` directory and is used to train the chatbot to respond to user queries. The training data is in the form of question-answer pairs and is used to train the chatbot to respond to user queries.

## Technologies

The chatbot is built using the following technologies:

- Hugging Face Transformers: For training the chatbot model with LoRA and quantisation.
- Accelerate: For speeding up the training process with distributed training.
- Discord.py: For creating the chatbot application.
- Docker: For containerizing the chatbot application and LLM server.

## Future Work

In the future, I plan to add more features to the chatbot to make it more interactive and engaging for users. Some of the features that I plan to add include:

- Better inference engine for generating responses: I plan to replace Hugging Face Transformers with a more efficient inference engine for generating responses like vLLM.
- Accessibility: I plan to make the chatbot more accessible to users with disabilities by adding features like text-to-speech and speech-to-text.