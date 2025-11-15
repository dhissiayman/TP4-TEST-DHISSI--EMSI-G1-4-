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
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.dhissiayman.tp3.assistant.Assistant;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3Routage {

    // ---------- LOGGING POUR VOIR LE ROUTAGE ----------
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }

    // ---------- MÉTHODE UTILITAIRE : INGESTION D'UN PDF ----------
    private static EmbeddingStore<TextSegment> ingestPdfAsEmbeddingStore(
            String resourceName,
            EmbeddingModel embeddingModel) throws URISyntaxException {

        URL resource = Test3Routage.class.getClassLoader().getResource(resourceName);
        if (resource == null) {
            throw new IllegalStateException("Le fichier " + resourceName + " n'a pas été trouvé dans resources");
        }
        Path pdfPath = Paths.get(resource.toURI());

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        System.out.println("Ingestion terminée pour " + resourceName + " : "
                + segments.size() + " segments enregistrés.");
        return store;
    }

    public static void main(String[] args) throws URISyntaxException {

        // 0) Logging détaillé LangChain4j (pour voir le routage)
        configureLogger();

        // 1) ChatModel Gemini
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 2) Modèle d'embedding partagé
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // ---------------------------------------------------------
        // PHASE 1 : Ingestion des 2 documents dans 2 EmbeddingStores
        // ---------------------------------------------------------

        // Fichier 1 : cours IA / RAG
        EmbeddingStore<TextSegment> iaStore =
                ingestPdfAsEmbeddingStore("langchain4j.pdf", embeddingModel);

        // Fichier 2 : autre contenu (non IA)
        EmbeddingStore<TextSegment> autreStore =
                ingestPdfAsEmbeddingStore("QCM_MAD-AI_COMPLET.pdf", embeddingModel);

        // ---------------------------------------------------------
        // PHASE 2 : 2 ContentRetrievers + QueryRouter + RetrievalAugmentor
        // ---------------------------------------------------------

        // 2 ContentRetrievers, un par source
        ContentRetriever iaRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(iaStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        ContentRetriever autreRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(autreStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // Map<ContentRetriever, String> pour décrire chaque source au LM
        Map<ContentRetriever, String> retrieverToDescription = Map.of(
                iaRetriever, "Documents de cours sur l'IA, les LLM, le RAG, LangChain4j, etc.",
                autreRetriever, "Documents qui ne parlent pas d'IA (autres matières / autres sujets)."
        );

        // QueryRouter basé sur le LLM (LanguageModelQueryRouter)
        QueryRouter queryRouter = LanguageModelQueryRouter.builder()
                .chatModel(chatModel)
                .retrieverToDescription(retrieverToDescription)
                .build();

        // RetrievalAugmentor utilisant ce QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // ---------------------------------------------------------
        // Assistant avec RetrievalAugmentor (et mémoire 10 messages)
        // ---------------------------------------------------------
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        // ---------------------------------------------------------
        // Tests interactifs (et observation du routage dans les logs)
        // ---------------------------------------------------------
        System.out.println("===== Test routage RAG (2 sources) =====");
        System.out.println("Exemples de questions à essayer :");
        System.out.println("- \"Qu'est-ce que le RAG ?\"");
        System.out.println("- \"Explique-moi ce qu'est une collection en Java\" (si ton 2e PDF parle de Java)");
        System.out.println("- \"À quoi sert LangChain4j ?\"");
        System.out.println("Tapez 'fin' pour quitter.");
        System.out.println("========================================");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("\nVotre question : ");
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
                System.out.println("--------------------------------------------------");
            }
        }
    }
}

