"""
uv run scripts/run.py --system VanillaRAGNewQGenFlow --help
"""
import time
from typing import List
from services.indicies import QueryService, SearchHit
from services.llms.ai71_client import AI71Client
from services.llms.ec2_llm_client import EC2LLMClient
from services.llms.mini_tgi_client import MiniTGIClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from services.logits_reranker import LogitsReranker
from systems.vanilla_rag.query_expansion import expand_queries
from systems.vanilla_rag.vanilla_rag_optimized_prompt_prompts import ANSWER_SYSTEM_PROMPT
from utils.doc_listing_utils import truncate_doc_listings, truncate_docs
from utils.logging_utils import get_logger


class VanillaRAGNewQGenFlow(RAGSystemInterface):
    def __init__(self,
                 llm_client='ai71',
                 qgen_model_id='tiiuae/falcon3-10b-instruct',
                 qgen_api_base=None,
                 k_queries=5,
                 rag_prompt_version="vanilla_rag",
                 use_logits_reranker=False,
                 expand_doc=False,
                 initial_retrieval_k_docs=50,
                 logits_prompt_version="identify_source",
                 context_words_limit=15_000):
        """
        Initialize the BasicRAG2.

        Args:
            llm_client: Client for the LLM. Options are 'ai71' or 'ec2_llm', default is 'ai71'.
            module_query_gen: Select this module for query generation. Options are 'with_number'|'without_number', default is 'with_number'.
            rag_prompt_version: The version of the prompt to use, vanilla_rag, new, Default is vanilla_rag.
            expand_doc: If True, expand the chunks into full docs and preserve ranking. Default is False.
            initial_retrieval_k_docs: The number of initial retrieval documents to use. Default is 50.
            logits_prompt_version: The version of the prompt to use for logits reranking, identify_source | can_answer. Default is identify_source.
            expand_words_limit: The number of words to keep after expanded chunks to documents.
        """
        if llm_client == 'ai71':
            self.rag_llm_client = AI71Client()
            self.qgen_llm_client = AI71Client()
        elif llm_client == 'ec2_llm':
            self.rag_llm_client = EC2LLMClient()
            self.qgen_llm_client = EC2LLMClient(
                model_id=qgen_model_id, api_base=qgen_api_base)
        self.use_logits_reranker = use_logits_reranker
        if use_logits_reranker:
            self.logits_llm = MiniTGIClient()
            self.reranker = LogitsReranker(
                self.logits_llm, prompt_version=logits_prompt_version)

        # if module_query_gen == 'with_num':
        self.logger = get_logger('vanilla_rag')
        self.k_queries = int(k_queries)
        self.initial_retrieval_k_docs = int(initial_retrieval_k_docs)

        # Store system prompts
        self.rag_prompt_version = rag_prompt_version
        if rag_prompt_version == "vanilla_rag":
            self.rag_system_prompt = "You are a helpful assistant. Answer the question based on the provided documents."
        elif rag_prompt_version == "new":
            self.rag_system_prompt = ANSWER_SYSTEM_PROMPT
        elif rag_prompt_version == "have_faith":
            self.rag_system_prompt = "You must respond based strictly on the information in provided passages. Do not incorporate any external knowledge or infer any details beyond what is given in the passages."

        self.qgen_system_prompt = f"Generate a list of {k_queries} search query variants based on the user's question, give me one query variant per line. There are no spelling mistakes in the original question. Do not include any other text."
        self.query_service = QueryService()
        self.expand_doc = expand_doc
        self.context_words_limit = int(context_words_limit)

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process the question and return the answer.
        """

        start_time = time.time()
        qs_res = expand_queries(question, self.qgen_llm_client)
        queries = qs_res.sub_queries

        listings: List[List[SearchHit]] = []
        k = int(self.initial_retrieval_k_docs / len(queries))
        for query in queries:
            results = self.query_service.query_fusion(
                query, k=k, per_source_k=100)
            listings.append(results)
        # len(docs) = k * len(queries) = 10 * 5 = 50

        logits_reranker_metrics = {}
        if self.use_logits_reranker:
            # # protect the LLM context length
            # hard_truncated_docs = truncate_doc_listings(
            #     listings=listings, context_word_limit=15_000)
            all_docs = [hit for list in listings for hit in list]
            # rerank and also truncate the documents
            docs = self.reranker.rerank(
                all_docs, question=qs_res.rephrased_query, words_limit=self.context_words_limit)

            # Record key metrics for logits reranker
            logits_reranker_metrics = {
                "logits_reranker_prompt_version": self.reranker.prompt_version,
                "logits_reranker_docs_before": len(all_docs),
                "logits_reranker_docs_after": len(docs) if docs else 0,
                "logits_reranker_highest_score": docs[0].score if docs and len(docs) > 0 else None,
                "logits_reranker_lowest_score": docs[-1].score if docs and len(docs) > 0 else None,
            }

            # If reranking returns nothing, fallback to first 10
            if docs is None or len(docs) == 0:
                docs = all_docs[:10]
                logits_reranker_metrics["logits_reranker_fallback"] = True
        else:
            docs = truncate_doc_listings(
                listings=listings, context_word_limit=self.context_words_limit)

        # If expand_doc is True, expand the chunks using get_doc while preserving original ranking
        if self.expand_doc:
            ordered_doc_ids = [doc.id for doc in docs]
            full_docs = self.query_service.get_docs(ordered_doc_ids)
            docs = truncate_docs(full_docs, self.context_words_limit)
            self.logger.info(f"Expanded documents", question=question,
                             taken_docs=len(docs),
                             original_docs=len(full_docs),
                             original_chunks=len(ordered_doc_ids))

        context = "## Documents: \n\n"
        context += "\n\n".join([doc.metadata.text for doc in docs])

        if self.rag_prompt_version == "vanilla_rag":
            prompt = context + "\n\n## Question: " + question + "\n\n## Answer: "
        elif self.rag_prompt_version == "new":
            prompt = context + "\n\n## Question: " + question + "\n\n## Question Components: " + \
                qs_res.components + "\n\n## Rephrased Question: " + qs_res.rephrased_query
        elif self.rag_prompt_version == "have_faith":
            _prp = "Provide a concise answer to the following question based on the information in the provided passages."
            prompt = context + "\n\n## Question: " + \
                question + "\n\n" + _prp + "\n\n## Answer: "
        self.logger.debug(f"Final prompt", question=question, prompt=prompt)

        answer, _ = self.rag_llm_client.complete_chat_once(
            prompt, self.rag_system_prompt)

        final_prompt = str([
            {"role": "system", "content": self.rag_system_prompt},
            {"role": "user", "content": prompt}
        ])

        # Prepare metadata with logits reranker metrics if available
        metadata = {"final_prompt": final_prompt}
        if self.use_logits_reranker:
            metadata.update(logits_reranker_metrics)

        return RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            metadata=metadata,
            context=[doc.metadata.text for doc in docs],
            doc_ids=[doc.metadata.doc_id for doc in docs],
            generated_queries=queries,
            total_time_ms=(time.time() - start_time) * 1000,
            system_name="VanillaRAGNewQGenFlow",
        )


if __name__ == "__main__":
    # Test the BasicRAG2 system
    rag_system = VanillaRAGNewQGenFlow()
    result = rag_system.process_question(
        "tourist outdoor activities mid january september alice springs ahmedabad compare",
        qid=1
    )

    print("-"*60)
    print("Question:\n", result.question)
    print("-"*60)
    print("Answer:\n", result.answer)
    print("-"*60)
    print("Context:\n", result.context)
    print("-"*60)
    print("Document IDs:", result.doc_ids)
    print("-"*60)
    print("Total Time (s):", result.total_time_ms)
    print("-"*60)
    print("Generated Queries:", result.generated_queries)
    print("-"*60)

    # ref_qa_pair = QAPair.from_dict({
    #     'question': "How does the artwork 'For Proctor Silex' create an interesting visual illusion for viewers as they approach it?",
    #     'answer': "Willie Cole's 'For Proctor Silex (Evidence and Presence)' creates an optical illusion where from a distance, it appears to be an African figurine displayed against patterned cloth, but upon closer inspection, it transforms into an industrial iron, which was the very tool used to burn the pattern into the canvas.",
    #     'context': ['OBERLIN, Ohio—The “Black Atlantic” is a cultural and geographic concept coined in 1993 by Paul Gilroy, and proposes a theory of the African diaspora that addresses points of origin obscured by slave trade and forced displacement, drawing identity from the bonds formed in the course of transport across the Atlantic Ocean. As Gilroy would have it, this produced “a culture that is not specifically African, American, Caribbean, or British, but all of these at once, a black Atlantic culture whose themes and techniques transcend ethnicity and nationality to produce something new …” In a show that opened earlier this year at the Allen Memorial Art Museum at Oberlin College, Afterlives of the Black Atlantic teases out aesthetics and individual visions that arise from this context. The show was co-curated by Andrea Gyorody, Ellen Johnson ’33 Assistant Curator of Modern and Contemporary Art, and Matthew Francis Rarey, Assistant Professor of the Arts of Africa and the Black Atlantic, and Oberlin’s first African Diasporic specialist.\n“We met shortly after I was hired in 2017,” said Gyorody, during a gallery tour with Hyperallergic. “And [Rarey] mentioned, sort of offhandedly, that 2019 would be the 400-year anniversary of the arrival of [the first] slave ships in the United States.” This sparked a discussion that evolved into the mounting of Afterlives, which brings together works from the United States, Europe, Latin America, the Caribbean, and Africa, drawn mostly from the AMAM collection, and supplemented by several loans and a site-specific commission by José Rodríguez. This work, titled “\\sə-kər\\” presents as an 12-foot Virgin of Regla — patron saint of the city of Havana and an adaptive form of the Orisha Yemayá, who protects the seas — identifiable by her regalia though the garments have been hollowed out to create a kind of open teepee, and the face has been replaced by a mirror that captures the visage of the viewer as she approaches.\n“I chose the phonetic spelling because it alludes to that space between and can be read as both ‘sucker’ and ‘succor,’” Rodríguez told The Oberlin Review. At the feet of the structure’s opening, an elaborate arrangement of pennies forms the threshold to the interior space, bedecked with the spiritual tools of Santería practitioners. Santería is a prime example of the African diasporic evolution of culture, as it represents an adaptation of Yoruba spiritual practice to enable its survival under Catholic colonialism in places like Cuba, Puerto Rico, the Dominican Republic, and other places in the mainland Americas. Rodríguez’s virgin seems to watch over the exhibition, while a twentieth-century Bocio figure by an unrecorded artist or workshop, attributed to the Republic of Benin or Togo, stands as a kind of sentinel as one enters the main gallery space. The figure is reminiscent of the Congolese/Central African nkisi, but where nkisi are objects inhabited by spirits, Bocio translates as “empowered cadaver” and exists in connection with Vodun spirituality, a set of practices that fused with outside influences under colonial rule and the slave trade, making its way from its West African origins to its expression as Vodou in Haiti.\nIn addition to works like these, that find roots in specific African traditions, there are a number of contemporary works that grapple with the more ambiguous aspects of identity wrought by slavery and involuntary relocation. “Untitled” (1999) by Leonardo Drew presents an abstracted geography, with one half of the large-scale wall hanging constructed of hundreds of cell-like openings stuffed with cotton, and the other half comprised of mirroring cells made primarily of discarded wood and rusty industrial materials of indeterminate origin. It is not a stretch to imagine this landscape as an abstracted picture of the labor history of Black people in the United States, where North-South divisions have merely offered enslaved and freed Black people different flavors of limiting and exploitative conditions.\nThis hangs adjacent to a bright blue candy spill by the late Félix González-Torres, which likewise transforms the floor space into a sort of proxy-ocean — one which directly implicates the sugar industry responsible for stoking demand in the slave trade, as well as any viewer who responds to the explicit invitation to take a piece of candy. In his lifetime, González-Torres oftentimes dealt with his own HIV-positive status and the condition of others living with AIDS, represented here by the slow disappearance of a sculptural work via audience participation; that the meaning of the work can be so readily adapted to the erosive and gutting power of disappearance and dislocation enacted by the slave trade acts as a powerful illustration of the way contemporary artists of color have responded either literally or thematically to this violent history. This point is echoed in a number of other works, including those by Alison Saar, Fred Wilson, Wangechi Mutu, and Dawoud Bey.\nAfterlives of the Black Atlantic hits all the right notes, bringing a stunning variety of media, sources, and perspectives into dynamic conversation. Each work carries its own weight, but moving through the gallery creates a number of arresting tableaus. In perhaps the most startling reveal of the show, “For Proctor Silex (Evidence and Presence)” by Willie Cole appears, from afar, to be an African figurine displayed against a backdrop of thick patterned cloth. Drawing closer, the figure transforms into an industrial iron, the very implement used to burn the motif into the canvas. This optical shifting between sacred figure and labor implement, decorative adornment and burnt offing, encapsulates the polarities of diasporic experience, and provides one of many moments that will linger in this space of examination.\nAfterlives of the Black Atlantic remains on display at the Allen Memorial Art Museum through May 24, 2020.'],
    #     'question_categories': [],
    #     'user_categories': [],
    #     'document_ids': ['<urn:uuid:b8ccd355-214a-424c-9d1c-5102ef4e8ce1>'],
    #     'qid': '1',
    # })
    # test_evaluator(LLMEvaluator(), [result], [ref_qa_pair])
