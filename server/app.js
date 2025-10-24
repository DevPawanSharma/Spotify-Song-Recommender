// server/app.js
import express from "express";
import cors from "cors";
import axios from "axios";
import dotenv from "dotenv";
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

const output = await client.textClassification({
	model: "j-hartmann/emotion-english-distilroberta-base",
	inputs: "I like you. I love you",
	provider: "hf-inference",
});

console.log(output);
dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Load configuration from environment variables
const MODEL_NAME = process.env.MODEL_NAME || "j-hartmann/emotion-english-distilroberta-base";
const HF_TOKEN = process.env.HF_TOKEN;
const PORT = process.env.PORT || 5000;

// Validate required environment variables
if (!HF_TOKEN) {
  console.error("❌ Missing Hugging Face API token (HF_TOKEN) in environment variables");
  process.exit(1);
}

// Construct API URL dynamically
const HF_URL = `https://api-inference.huggingface.co/models/${MODEL_NAME}`;

app.post("/detect-mood", async (req, res) => {
  const { text } = req.body;

  if (!text || text.trim() === "") {
    return res.status(400).json({ error: "Text is required" });
  }

  try {
    const response = await axios.post(
      HF_URL,
      { inputs: text },
      {
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
      }
    );

    if (!Array.isArray(response.data) || response.data.length === 0) {
      return res.status(500).json({ error: "Invalid response from Hugging Face API" });
    }

    const emotions = response.data[0];
    // Pick emotion with highest confidence score
    const topEmotion = emotions.reduce((prev, curr) => (prev.score > curr.score ? prev : curr));

    console.log(`Detected emotion: ${topEmotion.label} (${topEmotion.score.toFixed(3)})`);

    res.json({
      mood: topEmotion.label.toLowerCase(),
      confidence: topEmotion.score,
    });
  } catch (err) {
    console.error("Error calling Hugging Face API:", err.response?.data || err.message);
    res.status(500).json({ error: "Failed to analyze mood" });
  }
});

app.get("/", (req, res) => {
  res.send("🎵 Mood Detection API is running...");
});

app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
});
