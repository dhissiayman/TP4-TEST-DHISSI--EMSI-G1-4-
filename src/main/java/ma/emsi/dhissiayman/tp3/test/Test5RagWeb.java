package ma.emsi.dhissiayman.tp3.test;

/**
 * TP4 - Test 5 : RAG hybride (PDF + Web) avec Tavily
 * Auteur : DHISSI AYMAN
 *
 * Objectif :
 *  - Reprendre le RAG naïf basé sur un PDF (langchain4j.pdf)
 *  - Ajouter une recherche Web via Tavily (WebSearchEngine)
 *  - Combiner les deux sources avec un DefaultQueryRouter
 *  - Construire un RetrievalAugmentor qui interroge :
 *      • le contenu du PDF (RAG classique)
 *      • des sources Web (via Tavily)
 *
 * Remarque :
 *  - La clé Tavily doit être définie dans la variable d'environnement TAVILY_API_KEY
 *  - La clé Gemini doit être définie dans GEMINI_KEY
 */

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.emsi.dhissiayman.tp3.assistant.Assistant;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5RagWeb {

    /**
     * Configure le logger Java pour voir :
     *  - les logs internes de LangChain4j
     *  - les requêtes / réponses HTTP liées à Gemini et Tavily
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws URISyntaxException {

        // ---------------------------------------------------------
        // 0) Activation du logging
        // ---------------------------------------------------------
        configureLogger();

        // ---------------------------------------------------------
        // 1) ChatModel Gemini (LLM principal)
        // ---------------------------------------------------------
        String geminiKey = System.getenv("GEMINI_KEY");
        if (geminiKey == null) {
            throw new IllegalStateException("La variable d'environnement GEMINI_KEY n'est pas définie");
        }

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ---------------------------------------------------------
        // PHASE 1 : RAG naïf sur le PDF (comme dans RagNaif)
        // ---------------------------------------------------------

        URL resource = Test5RagWeb.class.getClassLoader().getResource("langchain4j.pdf");
        if (resource == null) {
            throw new IllegalStateException("Le fichier langchain4j.pdf n'a pas été trouvé dans resources");
        }
        Path pdfPath = Paths.get(resource.toURI());

        // Lecture du PDF
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        // Découpage en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
        List<TextSegment> segments = splitter.split(document);

        // Calcul des embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // Stockage dans un EmbeddingStore en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Phase 1 terminée : "
                + segments.size() + " segments enregistrés dans le magasin d'embeddings.");

        // ---------------------------------------------------------
        // PHASE 2 : ajout d'un ContentRetriever Web (Tavily) + QueryRouter
        // ---------------------------------------------------------

        // 2.1 ContentRetriever sur le PDF (RAG local)
        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 2.2 WebSearchEngine Tavily (recherche Web externe)
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null) {
            throw new IllegalStateException("La variable d'environnement TAVILY_API_KEY n'est pas définie");
        }

        WebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // 2.3 ContentRetriever basé sur le Web (Tavily)
        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(tavilyEngine)
                        // Possibilité d'ajouter des options (maxResults, etc.) si nécessaire
                        .build();

        // 2.4 QueryRouter qui combine les 2 retrievers (PDF + Web)
        // DefaultQueryRouter interroge tous les ContentRetrievers fournis.
        QueryRouter queryRouter = new DefaultQueryRouter(pdfRetriever, webRetriever);

        // 2.5 RetrievalAugmentor basé sur ce QueryRouter
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // ---------------------------------------------------------
        // 3) Assistant avec RAG hybride (PDF + Web) + mémoire
        // ---------------------------------------------------------
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        // ---------------------------------------------------------
        // 4) Boucle de test interactive
        // ---------------------------------------------------------
        System.out.println("===== Test 5 - RAG avec récupération Web (Tavily) - DHISSI AYMAN =====");
        System.out.println("Exemples de questions à tester :");
        System.out.println("- \"Qu'est-ce que le RAG ?\" (devrait utiliser surtout le PDF)");
        System.out.println("- \"Quelle est la dernière version de LangChain4j ?\" (devrait interroger le Web)");
        System.out.println("- \"Qui est le créateur de LangChain4j ?\"");
        System.out.println("Tapez 'fin' pour quitter.");
        System.out.println("========================================================");

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
