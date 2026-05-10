# Google NotebookLM RAG Clone

A RAG-powered application where a user can upload any document and have a conversation with it.
This application fulfills the requirements of Assignment 03.

## Chunking Strategy

This project uses the `RecursiveCharacterTextSplitter`. 
This strategy divides documents into manageable pieces using a list of delimiters (such as double newlines, single newlines, spaces, and empty characters).
It recursively tries to split the text at these delimiters until the chunk size is less than the specified maximum. This is particularly effective for text documents like PDFs because it tries to keep semantically related pieces of text (like paragraphs and sentences) together as much as possible before breaking them apart.

Parameters used:
- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters

## Setup & Local Development

1. Copy `.env.example` to `.env` and configure your `GEMINI_API_KEY`.
2. Start the Qdrant Vector Database using Docker:
   ```bash
   docker-compose up -d
   ```
3. Install dependencies:
   ```bash
   npm install
   ```

## Usage

### Web UI (Recommended for Live Project)
Start the Express server:
```bash
npm start
```
Then visit `http://localhost:3000` to upload a PDF and chat with it through the browser.

### CLI Application

#### 1. Indexing a Document

To index a PDF or text document:
```bash
node index.js index <path_to_pdf>
```
*Example: `node index.js index sample.pdf`*

#### 2. Asking Questions

To ask a question based on the indexed document:
```bash
node index.js ask "What is the document about?"
```

## Deployment (Live Project Link)
For easy deployment (e.g. Render, Railway):
1. Push this repository to GitHub.
2. Link it to Render/Railway as a Web Service.
3. Ensure `GEMINI_API_KEY` and `QDRANT_URL` (if using Qdrant Cloud) are set in your deployment environment variables.
4. The deployment service will automatically run `npm start`.
