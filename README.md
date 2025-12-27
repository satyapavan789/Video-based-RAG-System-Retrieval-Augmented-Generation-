Large Language Models often hallucinate when answering questions without grounded context.
This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries using video content by retrieving relevant information from transcribed videos before generating responses.

Solution Overview

The system converts video data into searchable knowledge and uses a retriever–generator architecture to provide context-aware, grounded answers.
Due to local hardware constraints, the project was prototyped on a representative subset of the dataset, while keeping the pipeline fully scalable.

Pipeline Architecture
Step 1: Video → Audio
Extract audio from video files using FFmpeg and convert videos to mp3 format.
File: 01_videos_to_mp3.py

Step 2: Audio → Text (Transcription)
Convert audio files to text using Whisper
Generate timestamped transcripts in JSON format
File: 02_mp3_to_json.py

Step 3: Text Preprocessing
Clean and normalize transcripts
Chunk long text into smaller segments suitable for embeddings
File: 03_preprocess_json.py

Step 4: RAG Query Processing
Generate embeddings for text chunks
Store embeddings in a vector store
Retrieve relevant chunks using cosine similarity  based on user query 
Generate grounded responses using an LLM
File: 04_process_incoming_query.py

Tech Stack:
-Python
-Numpy, Pandas
-OpenAI Whisper – Speech-to-text
-scikit-learn -embedding generation & cosine similarity
-LLM (ollama3.2)

## Design Decisions
-Used local embeddings due to compute constraints 
-Step wise scripts for modular debugging
