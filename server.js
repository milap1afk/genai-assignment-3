import "dotenv/config";
import express from "express";
import multer from "multer";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenAIEmbeddings, ChatGoogleGenAI } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import fs from "fs";

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static("public"));
app.use(express.json());

const upload = multer({ dest: 'uploads/' });

const COLLECTION_NAME = "notebook_lm_rag_collection";
const getVectorStoreConfig = () => ({
  url: process.env.QDRANT_URL || "http://localhost:6333",
  collectionName: COLLECTION_NAME,
  ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY })
});

// Endpoint to upload and index document
app.post("/api/upload", upload.single("document"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }
        
        console.log(`Uploaded file: ${req.file.path}`);
        const loader = new PDFLoader(req.file.path);
        const docs = await loader.load();

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const chunkedDocs = await textSplitter.splitDocuments(docs);
        const embeddings = new GoogleGenAIEmbeddings({
            model: "text-embedding-004",
        });

        await QdrantVectorStore.fromDocuments(chunkedDocs, embeddings, getVectorStoreConfig());
        
        // Clean up
        fs.unlinkSync(req.file.path);

        res.json({ message: "Document indexed successfully", chunks: chunkedDocs.length });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
});

// Endpoint to ask question
app.post("/api/ask", async (req, res) => {
    try {
        const { query } = req.body;
        if (!query) {
            return res.status(400).json({ error: "Query is required" });
        }

        const embeddings = new GoogleGenAIEmbeddings({
            model: "text-embedding-004",
        });

        const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, getVectorStoreConfig());
        const retriever = vectorStore.asRetriever({ k: 4 });
        const searchedChunks = await retriever.invoke(query);

        if (searchedChunks.length === 0) {
            return res.json({ answer: "No relevant context found in the indexed documents." });
        }

        const model = new ChatGoogleGenAI({
            modelName: "gemini-1.5-flash",
            temperature: 0.1
        });
        
        const systemPrompt = `You are an AI Assistant that helps resolve user queries based strictly on the provided context from indexed documents.

Rule:
- Only answer based on the available context provided below.
- If the answer cannot be found in the context, clearly state that you do not know. Do not use outside knowledge.
- Keep your answers concise and accurate.

Context:
${searchedChunks.map(c => c.pageContent).join('\n\n')}
`;

        const response = await model.invoke([
            ["system", systemPrompt],
            ["human", query]
        ]);

        res.json({ 
            answer: response.content,
            sources: searchedChunks.map(c => c.metadata.source || "Unknown")
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
