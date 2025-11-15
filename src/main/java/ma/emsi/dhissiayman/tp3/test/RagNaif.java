package ma.emsi.dhissiayman.tp3.test;

/**
 * TP4 - Test 1 : Implémentation d’un RAG naïf avec LangChain4j
 * Auteur : DHISSI AYMAN
 *
 * Ce test illustre la pipeline suivante :
 *  1) Chargement d’un fichier PDF
 *  2) Découpage en segments
 *  3) Génération d'embeddings (AllMiniLM-L6V2)
 *  4) Stockage dans un EmbeddingStore
 *  5) Création d’un Assistant utilisant un ContentRetriever
 *  6) Interaction utilisateur en boucle
 */

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import ma.emsi.dhissiayman.tp3.assistant.Assistant;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) throws URISyntaxException {

        // ---------------------------------------------------------
        // 0) Création du ChatModel (Gemini) utilisé par l'assistant
        // ---------------------------------------------------------
        String apiKey = System.getenv("GEMINI_KEY");

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();

        // ---------------------------------------------------------
        // PHASE 1 : Extraction + embeddings du PDF
        // ---------------------------------------------------------

        URL resource = RagNaif.class.getClassLoader().getResource("langchain4j.pdf");
        if (resource == null) {
            throw new IllegalStateException(
                    "Erreur : le fichier 'langchain4j.pdf' est introuvable dans /resources.");
        }

        Path pdfPath = Paths.get(resource.toURI());

        // Lecture du PDF avec Apache Tika
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        // Découpage du document (500 tokens avec overlap de 50)
        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(document);

        // Génération des embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // Stockage en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("✔ PHASE 1 terminée : "
                + segments.size() + " segments indexés.");

        // ---------------------------------------------------------
        // PHASE 2 : Création du RAG + assistant
        // ---------------------------------------------------------

        // Récupération d’informations pertinentes depuis les embeddings
        EmbeddingStoreContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Création de l'assistant avec mémoire
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        // Question imposée
        String questionInitiale = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        String reponseInitiale = assistant.chat(questionInitiale);
        System.out.println("Question : " + questionInitiale);
        System.out.println("Réponse : " + reponseInitiale);

        // ---------------------------------------------------------
        // Boucle interactive avec l’utilisateur
        // ---------------------------------------------------------
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("\n==================================================");
                System.out.println("Posez votre question (tapez 'fin' pour quitter) : ");
                String question = scanner.nextLine();

                if ("fin".equalsIgnoreCase(question)) break;
                if (question.isBlank()) continue;

                String reponse = assistant.chat(question);

                System.out.println("--------------------------------------------------");
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
