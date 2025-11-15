package ma.emsi.dhissiayman.tp3.test;

/**
 * TP4 - Test 3 : Routage des requêtes entre plusieurs sources (RAG multi-documents)
 * Auteur : DHISSI AYMAN
 *
 * Objectif :
 *  - Ingestion de 2 documents distincts (2 PDF) dans 2 EmbeddingStores séparés
 *  - Création de 2 ContentRetrievers : un pour chaque source
 *  - Utilisation d’un LanguageModelQueryRouter pour choisir la bonne source
 *  - Construction d’un RetrievalAugmentor basé sur ce routage
 *
 * Idée :
 *  Le LLM reçoit une description en langage naturel de chaque source
 *  (Map<ContentRetriever, String>), et décide quel ContentRetriever utiliser
 *  en fonction de la question.
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

    /**
     * Configuration du logger pour afficher les détails de LangChain4j,
     * notamment le routage des requêtes et les appels au LLM.
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }

    /**
     * Ingestion d’un PDF :
     *  - Chargement du fichier dans les resources (classpath)
     *  - Parsing via Apache Tika
     *  - Découpage en TextSegments
     *  - Calcul des embeddings
     *  - Stockage dans un EmbeddingStore en mémoire
     *
     * @param resourceName  nom du fichier PDF dans /resources
     * @param embeddingModel modèle d'embedding à utiliser
     * @return EmbeddingStore contenant les segments + embeddings
     */
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

        // ---------------------------------------------------------
        // 0) Activation du logging détaillé (LangChain4j + HTTP)
        // ---------------------------------------------------------
        configureLogger();

        // ---------------------------------------------------------
        // 1) Création du ChatModel Gemini (utilisé pour :
        //    - répondre aux questions
        //    - faire le routage via LanguageModelQueryRouter)
        // ---------------------------------------------------------
        String apiKey = System.getenv("GEMINI_KEY");
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // ---------------------------------------------------------
        // 2) Modèle d'embeddings partagé par les 2 sources
        // ---------------------------------------------------------
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // ---------------------------------------------------------
        // PHASE 1 : ingestion de 2 documents dans 2 EmbeddingStores
        // ---------------------------------------------------------

        // Fichier 1 : support de cours IA / RAG / LangChain4j
        EmbeddingStore<TextSegment> iaStore =
                ingestPdfAsEmbeddingStore("langchain4j.pdf", embeddingModel);

        // Fichier 2 : autre contenu (non IA)
        EmbeddingStore<TextSegment> autreStore =
                ingestPdfAsEmbeddingStore("QCM_MAD-AI_COMPLET.pdf", embeddingModel);

        // ---------------------------------------------------------
        // PHASE 2 : 2 ContentRetrievers + QueryRouter + RetrievalAugmentor
        // ---------------------------------------------------------

        // ContentRetriever pour la source IA
        ContentRetriever iaRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(iaStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // ContentRetriever pour l’autre source
        ContentRetriever autreRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(autreStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // Map<ContentRetriever, String> : description en langage naturel
        // de chaque source, lue par le LLM pour choisir la route.
        Map<ContentRetriever, String> retrieverToDescription = Map.of(
                iaRetriever,
                "Documents de cours sur l'IA, les LLM, le RAG, LangChain4j, etc.",
                autreRetriever,
                "Documents qui ne parlent pas directement d'IA (autres matières / autres sujets)."
        );

        // QueryRouter basé sur le LLM :
        // Le LLM lit la question + les descriptions et décide quel retriever utiliser.
        QueryRouter queryRouter = LanguageModelQueryRouter.builder()
                .chatModel(chatModel)
                .retrieverToDescription(retrieverToDescription)
                .build();

        // RetrievalAugmentor qui s'appuie sur ce QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // ---------------------------------------------------------
        // Assistant avec RetrievalAugmentor + mémoire (10 messages)
        // ---------------------------------------------------------
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        // ---------------------------------------------------------
        // Boucle de tests interactifs (vérifier le routage via les logs)
        // ---------------------------------------------------------
        System.out.println("===== Test 3 - Routage RAG (2 sources) - DHISSI AYMAN =====");
        System.out.println("Exemples de questions à essayer :");
        System.out.println("- \"Qu'est-ce que le RAG ?\" (devrait aller vers le cours IA)");
        System.out.println("- \"Explique-moi ce qu'est une collection en Java\" (si présent dans le 2e PDF)");
        System.out.println("- \"À quoi sert LangChain4j ?\"");
        System.out.println("Tapez 'fin' pour quitter.");
        System.out.println("============================================================");

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
