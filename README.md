RAG AI Assistant for Video Data
(Retrieval-Augmented Generation Project)

Overview:
-Developed a Retrieval-Augmented Generation (RAG) assistant to extract and answer questions from video content by combining speech-to-text transcription with retrieval-based language modeling.

Problem Statement:
-Video content contains valuable information but is difficult to search or query efficiently.
-The goal was to enable text-based querying of video content using AI techniques.

Approach:
-Converted video to audio and generated transcripts using Whisper AI.
-Processed and chunked transcripts for efficient retrieval.
-Implemented a retrieval-based generation pipeline using a locally hosted LLM via Ollama.
-Retrieved relevant transcript segments and generated contextual answers based on user queries.

Key Features
-Question answering over video content.
-Local LLM inference (no external API dependency).
-Modular pipeline separating transcription, retrieval, and generation.

Tools & Technologies:
Python | Whisper AI | FFmpeg | Vector Search | Ollama (Local LLM)

Outcome:
This project demonstrates practical experience with modern AI workflows, including transcription, retrieval systems, and large language model integration, highlighting applied understanding of RAG concepts.
