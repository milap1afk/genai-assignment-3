import "dotenv/config";
import express from "express";
import multer from "multer";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenAIEmbeddings, ChatGoogleGenAI } from "@langchain/google-genai";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());

const upload = multer({ dest: '/tmp/' });

app.get("/", (req, res) => {
    try {
        const html = fs.readFileSync(path.join(__dirname, "public", "index.html"), "utf-8");
        res.send(html);
    } catch (e) {
        res.status(500).send("Error loading frontend: " + e.message);
    }
});

// Endpoint to upload and parse document
app.post("/api/upload", upload.single("document"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }
        
        console.log(`Uploaded file: ${req.file.path}`);
        const loader = new PDFLoader(req.file.path);
        const docs = await loader.load();
        
        // Clean up temp file
        fs.unlinkSync(req.file.path);

        res.json({ 
            message: "Document loaded successfully", 
            text: docs.map(d => d.pageContent).join('\n')
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
});

// Endpoint to ask question
app.post("/api/ask", async (req, res) => {
    try {
        const { query, documentText } = req.body;
        if (!query || !documentText) {
            return res.status(400).json({ error: "Query and document context are required" });
        }

        // Run full RAG pipeline in memory (perfect for Vercel Serverless)
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const chunkedDocs = await textSplitter.createDocuments([documentText]);

        const embeddings = new GoogleGenAIEmbeddings({
            model: "text-embedding-004",
        });

        const chunkTexts = chunkedDocs.map(c => c.pageContent);
        const chunkEmbeddings = await embeddings.embedDocuments(chunkTexts);
        const queryEmbedding = await embeddings.embedQuery(query);

        const scoredChunks = chunkedDocs.map((chunk, i) => ({
            chunk,
            score: cosineSimilarity(queryEmbedding, chunkEmbeddings[i])
        }));

        scoredChunks.sort((a, b) => b.score - a.score);
        const searchedChunks = scoredChunks.slice(0, 4).map(c => c.chunk);

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

if (process.env.NODE_ENV !== 'production' && !process.env.VERCEL) {
    app.listen(port, () => {
        console.log(`Server running at http://localhost:${port}`);
    });
}

export default app;
