package ma.emsi.dhissiayman.tp3.test;

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
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
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

public class Test4PasDeRag {

    // ---------- LOGGING LANGCHAIN4J (comme dans Test2Logging) ----------
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }

    // ---------- PHASE 1 : ingestion d'un seul PDF dans un EmbeddingStore ----------
    private static EmbeddingStore<TextSegment> ingestRagPdf(
            String resourceName,
            EmbeddingModel embeddingModel) throws URISyntaxException {

        URL resource = Test4PasDeRag.class.getClassLoader().getResource(resourceName);
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

        System.out.println("Ingestion RAG terminée pour " + resourceName + " : "
                + segments.size() + " segments enregistrés.");
        return store;
    }

    public static void main(String[] args) throws URISyntaxException {

        // 0) Logging détaillé
        configureLogger();

        // 1) ChatModel Gemini (utilisé à la fois pour l'assistant et pour le routage)
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)   // pour bien voir ce qui se passe
                .build();

        // 2) Modèle d'embeddings + ingestion du support RAG
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore =
                ingestRagPdf("langchain4j.pdf", embeddingModel); // support de cours RAG

        // 3) ContentRetriever basé sur ce seul EmbeddingStore
        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 4) PromptTemplate pour décider "RAG ou pas"
        PromptTemplate routerTemplate = PromptTemplate.from(
                "Est-ce que la requête suivante porte sur l'IA ou le contenu du cours LangChain4j ? " +
                        "Réponds seulement par 'oui', 'non' ou 'peut-être'.\n" +
                        "Requête : {{query}}"
        );

        // 5) QueryRouter personnalisé : "utiliser le RAG ou pas ?"
        QueryRouter queryRouter = new QueryRouter() {
            @Override
            public List<ContentRetriever> route(Query query) {

                // On construit le Prompt à partir du template
                Prompt prompt = routerTemplate.apply(Map.of(
                        "query", query.text()
                ));

                // On interroge directement le ChatModel avec le texte du Prompt
                String answer = chatModel.chat(prompt.text()).trim().toLowerCase();

                System.out.println("[Router] Question : " + query.text());
                System.out.println("[Router] Réponse du LM pour le routage : " + answer);

                if (answer.startsWith("non")) {
                    // ❌ La requête ne porte pas sur l'IA → pas de RAG
                    return List.of();
                } else {
                    // ✅ "oui" ou "peut-être" → on utilise le ContentRetriever
                    return List.of(contentRetriever);
                }
            }
        };

        // 6) RetrievalAugmentor utilisant ce QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 7) Assistant avec RetrievalAugmentor (et mémoire de 10 messages)
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        // 8) Boucle de test : d'abord "Bonjour", puis une vraie question IA
        System.out.println("===== Test 4 - Pas de RAG =====");
        System.out.println("Exemples à tester :");
        System.out.println("- \"Bonjour\"");
        System.out.println("- \"Qu'est-ce que le RAG ?\"");
        System.out.println("- \"À quoi sert LangChain4j ?\"");
        System.out.println("Tapez 'fin' pour quitter.");
        System.out.println("================================");

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
