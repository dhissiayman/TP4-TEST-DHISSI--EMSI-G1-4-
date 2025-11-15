package ma.emsi.dhissiayman.tp3.test;


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
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;


import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Test2Logging {
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // plus verbeux que INFO

        // Handler console
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        // Pour éviter de dupliquer les logs
        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }


    public static void main(String[] args) throws URISyntaxException {

        // ---------------------------------------------------------
        // 0) Configuration du logger.
        // ---------------------------------------------------------

        configureLogger();

        // ---------------------------------------------------------
        // 0) Création du ChatModel (comme dans le TP2)
        // ---------------------------------------------------------
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash") // ou gemini-2.5-flash si dispo
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ---------------------------------------------------------
        // PHASE 1 : enregistrement des embeddings
        // ---------------------------------------------------------

        URL resource = RagNaif.class.getClassLoader().getResource("langchain4j.pdf");
        if (resource == null) {
            throw new IllegalStateException("Le fichier langchain4j.pdf n'a pas été trouvé dans resources");
        }
        Path pdfPath = Paths.get(resource.toURI());

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Phase 1 terminée : " +
                segments.size() + " segments enregistrés dans le magasin d'embeddings.");

        // ---------------------------------------------------------
        // PHASE 2 : utilisation des embeddings pour répondre aux questions
        // ---------------------------------------------------------

        // 1) Création du ContentRetriever (2 résultats, score >= 0.5)
        EmbeddingStoreContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 2) Ajout d'une mémoire de 10 messages + création de l'assistant
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .contentRetriever(contentRetriever)
                        .build();

        // 3) Première question imposée par l'énoncé
        String questionInitiale = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        String reponseInitiale = assistant.chat(questionInitiale);
        System.out.println("Question : " + questionInitiale);
        System.out.println("Réponse : " + reponseInitiale);

        // 4) Boucle pour poser plusieurs questions sans recompiler
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (tapez 'fin' pour quitter) : ");
                String question = scanner.nextLine();
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("--------------------------------------------------");
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
