// ==========================
// STEP 0: IMPORTS
// ==========================

// Load .env variables
import "dotenv/config";

// OpenAI SDK
import OpenAI from "openai";

// Read local files
import fs from "fs";

// Terminal input/output
import readline from "readline";

// Supabase client
import { createClient } from "@supabase/supabase-js";


// ==========================
// STEP 1: CREATE CLIENTS
// ==========================

// OpenAI client for embeddings + chat model
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Supabase client for database operations
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);


// ==========================
// STEP 2: DEFINE FILE PATHS
// ==========================

// These are our data sources
const DOCS_FILE = "docs.txt";
const FAQ_FILE = "faq.txt";
const JSON_FILE = "api-data.json";


// ==========================
// STEP 3: CREATE EMBEDDING FUNCTION
// ==========================

/**
 * Converts text into an embedding vector.
 *
 * Example:
 * "What is RAG?"
 * becomes:
 * [0.12, -0.44, 0.91, ...]
 *
 * Supabase stores vectors, but OpenAI creates them.
 */
async function createEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });

  return response.data[0].embedding;
}


// ==========================
// STEP 4: LOAD docs.txt SOURCE
// ==========================

/**
 * Reads docs.txt and splits it into chunks.
 *
 * Each chunk becomes a document object:
 * {
 *   source: "docs.txt",
 *   type: "text",
 *   content: "..."
 * }
 */
function loadDocsSource() {
  const text = fs.readFileSync(DOCS_FILE, "utf-8");

  return text
    .split(".")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => ({
      source: "docs.txt",
      type: "text",
      content: chunk,
    }));
}


// ==========================
// STEP 5: LOAD faq.txt SOURCE
// ==========================

/**
 * Reads faq.txt.
 *
 * We split by empty lines because FAQ usually looks like:
 *
 * Question: ...
 * Answer: ...
 *
 * Question: ...
 * Answer: ...
 */
function loadFAQSource() {
  const text = fs.readFileSync(FAQ_FILE, "utf-8");

  return text
    .split("\n\n")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => ({
      source: "faq.txt",
      type: "faq",
      content: chunk,
    }));
}


// ==========================
// STEP 6: LOAD api-data.json SOURCE
// ==========================

/**
 * Reads structured JSON data.
 *
 * Example JSON:
 * [
 *   {
 *     "title": "Vector Database",
 *     "content": "A vector database stores embeddings."
 *   }
 * ]
 *
 * We convert each JSON object into normal text,
 * because embeddings need text input.
 */
function loadJSONSource() {
  const raw = fs.readFileSync(JSON_FILE, "utf-8");
  const items = JSON.parse(raw);

  return items.map((item) => ({
    source: "api-data.json",
    type: "json",
    content: `${item.title}: ${item.content}`,
  }));
}


// ==========================
// STEP 7: LOAD ALL SOURCES
// ==========================

/**
 * Combines all sources into one array.
 *
 * This is important because RAG needs one unified knowledge base.
 */
function loadAllSources() {
  return [
    ...loadDocsSource(),
    ...loadFAQSource(),
    ...loadJSONSource(),
  ];
}


// ==========================
// STEP 8: INSERT SOURCES INTO SUPABASE
// ==========================

/**
 * This function is for ingestion.
 *
 * Ingestion means:
 * - load documents
 * - create embeddings
 * - store text + embeddings in database
 *
 * Run this only once.
 * After inserting data, comment it again to avoid duplicates.
 */
async function insertSources() {
  const documents = loadAllSources();

  for (const doc of documents) {
    // Convert document content into embedding vector
    const embedding = await createEmbedding(doc.content);

    // Store document in Supabase
    const { error } = await supabase.from("documents").insert({
      content: doc.content,
      source: doc.source,
      type: doc.type,
      embedding: embedding,
    });

    if (error) {
      console.error("Insert error:", error.message);
      continue;
    }

    console.log(`Inserted: ${doc.source} | ${doc.type}`);
  }

  console.log("All sources inserted into Supabase.");
}


// ==========================
// STEP 9: SEARCH DOCUMENTS
// ==========================

/**
 * This function searches Supabase using vector similarity.
 *
 * Steps:
 * 1. Convert user question into embedding
 * 2. Send embedding to Supabase RPC function
 * 3. Supabase compares vectors internally
 * 4. Supabase returns best matching chunks
 */
async function searchDocuments(question) {
  const questionEmbedding = await createEmbedding(question);

  const { data, error } = await supabase.rpc("match_documents", {
    query_embedding: questionEmbedding,
    match_count: 5,
  });

  if (error) {
    throw new Error(error.message);
  }

  return data;
}


// ==========================
// STEP 10: ASK QUESTION FROM TERMINAL
// ==========================

/**
 * This function lets the user type a question in the terminal.
 */
function askQuestion() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question("Ask a question: ", (question) => {
      rl.close();
      resolve(question);
    });
  });
}


// ==========================
// STEP 11: GENERATE FINAL ANSWER
// ==========================

/**
 * This function sends:
 * - retrieved context
 * - user question
 *
 * to the LLM.
 *
 * The LLM answers using only the retrieved context.
 */
async function generateAnswer(question, results) {
  const context = results
    .map((item) => {
      return `[Source: ${item.source} | Type: ${item.type}]\n${item.content}`;
    })
    .join("\n\n");

  const completion = await openai.chat.completions.create({
    model: "gpt-4.1-mini",
    messages: [
      {
        role: "system",
        content:
          "Answer ONLY using the provided context. Mention the source if useful. If the answer is not in the context, say you don't know.",
      },
      {
        role: "user",
        content: `
Context:
${context}

Question:
${question}
        `,
      },
    ],
  });

  return completion.choices[0].message.content;
}


// ==========================
// STEP 12: MAIN APP
// ==========================

/**
 * Main RAG flow:
 *
 * 1. Optional: insert sources into Supabase
 * 2. Ask user question
 * 3. Search relevant chunks in Supabase
 * 4. Build context
 * 5. Generate answer
 * 6. Print result
 */
async function main() {
  /**
   * FIRST RUN ONLY:
   *
   * Uncomment this line one time:
   * await insertSources();
   *
   * Then run:
   * node index.js
   *
   * After data is inserted, comment it again.
   */
  // await insertSources();

  // Ask user question
  const question = await askQuestion();

  // Retrieve relevant chunks from Supabase
  const results = await searchDocuments(question);

  // Show retrieved sources before final answer
  console.log("\nRetrieved Sources:");
  results.forEach((item) => {
    console.log(`- ${item.source} (${item.type}) | score: ${item.similarity}`);
  });

  // Generate final answer
  const answer = await generateAnswer(question, results);

  // Print final answer
  console.log("\nAnswer:");
  console.log(answer);
}


// ==========================
// STEP 13: RUN APP
// ==========================

main();