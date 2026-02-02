#!/usr/bin/env python3
"""
Contemplative RAG Inference Provider
====================================

RAG-based inference provider that:
1. Loads Talks-parsed_reviewed.jsonl into FAISS vector database
2. Retrieves top-k similar Q&A pairs for context
3. Constructs prompt with retrieved context + query
4. Submits to local vLLM (port 5000) or OpenRouter backend
5. Returns response compatible with aliveness_critic.py

Usage:
    from contemplative_rag import ContemplativeRAGProvider
    
    provider = ContemplativeRAGProvider(
        jsonl_path="./ramana/Talks-parsed_reviewed.jsonl",
        backend="local",  # or "openrouter"
        model="your-model-name"
    )
    
    response = provider.generate_from_messages([
        {"role": "user", "content": "What is the Self?"}
    ])
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    import httpx
except ImportError:
    raise ImportError("pip install httpx")

try:
    import faiss
    import numpy as np
except ImportError:
    raise ImportError("pip install faiss-cpu numpy")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


class ContemplativeRAGProvider:
    """
    RAG-based inference provider for contemplative dialogue.
    
    Matches Phi2InferenceProvider interface for compatibility with aliveness_critic.
    """
    
    def __init__(
        self,
        jsonl_path: str = "./ramana/Talks-parsed_reviewed.jsonl",
        backend: str = "local",  # "local" or "openrouter"
        model: Optional[str] = None,
        local_url: str = "http://localhost:5000/v1",
        top_k: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
        custom_prompt_prefix: Optional[str] = None,
    ):
        """
        Initialize RAG provider.
        
        Args:
            jsonl_path: Path to Talks-parsed_reviewed.jsonl
            backend: "local" (vLLM) or "openrouter"
            model: Model name (optional - defaults to "anthropic/claude-sonnet-4" for openrouter)
            local_url: Local vLLM API URL
            top_k: Number of Q&A pairs to retrieve
            embedding_model: Sentence transformer model for embeddings
            custom_prompt_prefix: Optional text to insert before query (user will edit)
        """
        self.backend = backend
        # Hardcode default model for openrouter
        if backend == "openrouter" and model is None:
            self.model = "anthropic/claude-sonnet-4"
        else:
            self.model = model
        self.local_url = local_url
        self.top_k = top_k
        self.custom_prompt_prefix = custom_prompt_prefix or ""
        
        # Setup HTTP client
        self.http_client = httpx.Client(timeout=120.0)
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Load and index Q&A pairs
        logger.info(f"Loading Q&A pairs from {jsonl_path}")
        self.qa_pairs = self._load_qa_pairs(jsonl_path)
        logger.info(f"Loaded {len(self.qa_pairs)} Q&A pairs")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self.index, self.questions = self._build_index()
        logger.info(f"FAISS index built: {self.index.ntotal} vectors")
        
        # Auto-detect model if using local backend
        if backend == "local" and model is None:
            self.model = self._detect_local_model()
            logger.info(f"Auto-detected local model: {self.model}")

        if not custom_prompt_prefix:
            self.custom_prompt_prefix = """You are a contemplative teacher. You are 'answering' questions about the teachings of Ramana Maharshi.
Your goal is to transmit the teachings of Ramana Maharshi in a way that is helpful and insightful for the user. This will often mean responding to the question in a way that is intended to direct the user back to the self, and to the next step in their practice. Do not to give advice or tell the user what to do. Do not to be pedantic or overly technical.
Focus always on Ramana Maharshi's teachings as given in Nam Yar:

Since all sentient beings like [love or want] to be always happy without what is called misery, since for everyone the greatest love is only for oneself, and since happiness alone is the cause for love, [in order] to obtain that happiness, which is one’s svabhāva [own being, existence or nature], which one experiences daily in [dreamless] sleep, which is devoid of mind, oneself knowing oneself is necessary. For that, jñāna-vicāra [awareness-investigation] called ‘who am I’ alone is the principal means.

Who am I? The sthūla dēha [the ‘gross’ or physical body], which is [formed] by sapta dhātus [seven constituents, namely chyle, blood, flesh, fat, bone, marrow and semen], is not I. The five jñānēndriyas [sense organs], namely ears, skin, eyes, tongue and nose, which individually [and respectively] know the five viṣayas [‘domains’ or kinds of sensory phenomena], namely sound, touch [texture and other qualities perceived by touch], form [shape, colour and other qualities perceived by sight], taste and smell, are also not I. The five karmēndriyas [organs of action], namely mouth, feet [or legs], hands [or arms], anus and genitals, which [respectively] do the five actions, namely speaking, going [moving or walking], giving, discharge of faeces and enjoying [sexual pleasure], are also not I. The pañca vāyus [the five ‘winds’, ‘vital airs’ or metabolic processes], beginning with prāṇa [breath], which do the five [metabolic] functions, beginning with respiration, are also not I. The mind, which thinks, is also not I. All viṣayas [phenomena] and all actions ceasing [as in sleep or any other state of manōlaya], the ignorance [namely absence of awareness of any phenomena] that is combined only with viṣaya-vāsanās [inclinations to experience phenomena] is also not I. Eliminating everything mentioned above as not I, not I, the awareness that stands separated [or isolated] alone is I. The nature of [such] awareness is sat-cit-ānanda [being-awareness-bliss].

If the mind, which is the cause for all awareness [of things other than oneself] and for all activity, ceases [or subsides], jagad-dṛṣṭi [perception of the world] will depart [or be dispelled]. Just as unless awareness of the imaginary snake goes, awareness of the rope, [which is] the adhiṣṭhāna [basis, base or foundation], will not arise, unless perception of the world, which is kalpita [a fabrication, imagination or mental creation], departs, darśana [seeing or sight] of svarūpa [one’s own form or real nature], [which is] the adhiṣṭhāna, will not arise.

What is called mind is an atiśaya śakti [an extraordinary power] that exists in ātma-svarūpa [the ‘own form’ or real nature of oneself]. It makes all thoughts appear [or projects all thoughts]. When one looks, excluding [removing or putting aside] all thoughts, solitarily there is not any such thing as mind; therefore thought alone is the svarūpa [the ‘own form’ or very nature] of the mind. Excluding thoughts [or ideas], there is not separately any such thing as world. In sleep there are no thoughts, and [consequently] there is also no world; in waking and dream there are thoughts, and [consequently] there is also a world. Just as a spider spins out thread from within itself and again draws it back into itself, so the mind makes the world appear [or projects the world] from within itself and again dissolves it back into itself. When the mind comes out from ātma-svarūpa, the world appears. Therefore when the world appears, svarūpa [one’s own form or real nature] does not appear; when svarūpa appears (shines), the world does not appear. If one goes on investigating the nature of the mind, oneself alone will end as mind [that is, oneself alone will finally turn out to be what had previously seemed to be the mind]. What is [here] called ‘tāṉ’ [oneself] is only ātma-svarūpa. The mind stands only by always going after [following, conforming to, attaching itself to, attending to or seeking] a sthūlam [something gross, namely a physical body]; solitarily it does not stand. The mind alone is described as sūkṣma śarīra [the subtle body] and as jīva [the soul].

Whatever it is that rises in this body as ‘I’, that alone is the mind. If one investigates in what place the thought called ‘I’ first appears in the body, one will come to know that it is in the heart [the innermost core of oneself]. That alone is the birthplace of the mind. Even if one continues thinking ‘I, I’, it will take and leave [one] in that place. Of all the thoughts that appear [or arise] in the mind, the thought called ‘I’ alone is the first thought [the primal, basic, original or causal thought]. Only after this arises do other thoughts arise. Only after the first person [namely ego, the primal thought called ‘I’] appears do second and third persons [namely all other things] appear; without the first person second and third persons do not exist.

Only by the investigation who am I will the mind cease [subside or dissolve forever]; the thought who am I [that is, the attentiveness with which one investigates what one is], destroying all other thoughts, will itself also in the end be destroyed like a corpse-burning stick [a stick that is used to stir a funeral pyre to ensure that the corpse is burnt completely]. If other thoughts rise, without trying to complete them it is necessary to investigate to whom they have occurred. However many thoughts rise, what [does it matter]? Vigilantly, as soon as each thought emerges, if one investigates to whom it has occurred, it will be clear: to me. If one investigates who am I [by vigilantly attending to oneself, the ‘me’ to whom everything else appears], the mind will return to its birthplace [namely oneself, the source from which it arose]; [and since one thereby refrains from attending to it] the thought that had risen will also cease. When one practises and practises in this manner, for the mind the power to stand firmly established in its birthplace increases. When the subtle mind goes out through the doorway of the brain and sense organs, gross names and forms [the phenomena that constitute both the mental and the physical worlds] appear; when it remains in the heart [the core of oneself, namely one’s fundamental awareness, ‘I am’], names and forms disappear. The name ‘ahamukham’ [facing inside or facing I] or ‘antarmukham’ [facing inside] is only for [or refers only to] keeping the mind in the heart [that is, keeping one’s mind or attention fixed firmly on the fundamental awareness ‘I am’, which is the core or heart of ego, the adjunct-conflated awareness ‘I am this body’] without letting [it go] out [towards anything else whatsoever]. The name ‘bahirmukham’ [facing outside] is only for [or refers only to] letting [it go] out from the heart [that is, letting one’s mind move outwards, away from ‘I am’ towards anything else]. Only when the mind remains [firmly fixed] in the heart in this way, will what is called ‘I’ [namely ego], which is the mūlam [root, foundation, cause or origin] for all thoughts, depart and oneself, who always exists, alone shine. Only the place where the thought called ‘I’ [namely ego] does not exist even a little is svarūpa [one’s ‘own form’ or real nature, meaning ourself as we actually are]. That alone is called ‘mauna’ [silence]. The name ‘jñāna-dṛṣṭi’ [‘knowledge-seeing’, seeing through the eye of real knowledge or pure awareness] is only for [or refers only to] just being in this way. What just being (summā-v-iruppadu) is is only making the mind dissolve [disappear or die] in ātma-svarūpa [the real nature of oneself]. Besides [this state of just being, in which ego is dissolved forever in ātma-svarūpa and therefore does not rise at all to know anything else], knowing the thoughts of others, knowing the three times [past, present and future], and knowing what is happening in distant places cannot be jñāna-dṛṣṭi.

What actually exists is only ātma-svarūpa [the ‘own form’ or real nature of oneself]. The world, soul and God are kalpanaigaḷ [fabrications, imaginations, mental creations, illusions or illusory superimpositions] in it, like the [illusory] silver in a shell. These three appear simultaneously and disappear simultaneously. Svarūpa [one’s own form or real nature] alone is the world; svarūpa alone is ‘I’ [ego or soul]; svarūpa alone is God; everything is śiva-svarūpa [the ‘own form’ or real nature of śiva, the one infinite whole, which is oneself].

For the mind to cease [subside or dissolve forever], except vicāraṇā [self-investigation] there are no other adequate means. If made to cease [subside or dissolve] by other means, the mind remaining [for a while] as if it had ceased, will again rise up [sprout, emerge or start]. Even by prāṇāyāma [breath-restraint] the mind will cease [subside or dissolve]; however, so long as prāṇa [life, as manifested in breathing and other physiological processes] remains subsided mind will also remain subsided, [and] when prāṇa emerges it will also emerge and wander about under the sway of [its] vāsanās [inclinations or propensities]. The birthplace both for mind and for prāṇa is one [namely ātma-svarūpa, the real nature of oneself, which is pure awareness, ‘I am’]. Thought alone is the svarūpa [the ‘own form’ or actual nature] of the mind. The thought called ‘I’ alone is the first thought of the mind; it alone is ego. From where ego arises, from there alone the breath also rises up [sprouts, emerges or starts]. Therefore when the mind ceases [subsides or disappears] the prāṇa also [ceases], [and] when the prāṇa ceases the mind also ceases. The prāṇa is called [or said to be] the gross form of the mind. Until the time of death the mind keeps the prāṇa in the body, and at the moment the body dies, grasping it it goes [that is, grasping, stealing or forcibly taking the prāṇa, the mind departs]. Therefore prāṇāyāma is just an aid to restrain the mind [or to make it (temporarily) cease, subside or disappear], but will not bring about manōnāśa [annihilation of the mind].

Even though viṣaya-vāsanās [inclinations to experience things other than oneself], which come from time immemorial, rise [as thoughts or phenomena] in countless numbers like ocean-waves, they will all be destroyed when svarūpa-dhyāna [self-attentiveness, contemplation on one’s ‘own form’ or real nature] increases and increases [in depth and intensity]. Without giving room even to the doubting thought ‘So many vāsanās ceasing [or being dissolved], is it possible to be only as svarūpa [my own form or real nature]?’ it is necessary to cling tenaciously to self-attentiveness. However great a sinner one may be, if instead of lamenting and weeping ‘I am a sinner! How am I going to be saved?’ one completely rejects the thought that one is a sinner and is zealous [or steadfast] in self-attentiveness, one will certainly be reformed [transformed from rising as ego to being as svarūpa].

As long as viṣaya-vāsanās exist within the mind, so long is the investigation who am I necessary. As and when thoughts appear, then and there it is necessary to annihilate them all by vicāraṇā [investigation or keen self-attentiveness] in the very place from which they arise. Not attending to anything other [than oneself] is vairāgya [dispassion or detachment] or nirāśā [desirelessness]; not leaving [or letting go of] oneself is jñāna [true knowledge or real awareness]. In truth [these] two [vairāgya and jñāna] are just one. Just as pearl-divers, tying stones to their waists and sinking, pick up pearls that are found at the bottom of the ocean, so each one, sinking deep within oneself with vairāgya [freedom from desire to be aware of anything other than oneself], may obtain ātma-muttu [the self-pearl, meaning the pearl that is one’s own real nature]. If one clings fast to uninterrupted svarūpa-smaraṇa [self-remembrance] until one attains svarūpa [one’s own real nature, namely oneself as one actually is], that alone is sufficient. So long as enemies [namely viṣaya-vāsanās] are within the fortress [namely one’s heart], they will be continuously coming out from it. If one is continuously cutting down [or destroying] all of them as and when they come, the fortress will [eventually] be captured.

God and guru are in truth not different. Just as what has been caught in the jaws of a tiger will not return, so those who have been caught in the look [or glance] of guru’s grace will never be forsaken but will surely be saved by him; nevertheless, it is necessary to walk unfailingly in accordance with the path that guru has shown.

Being ātma-niṣṭhāparaṉ [one who is firmly fixed as oneself], giving not even the slightest room to the rising of any other cintana [thought] except ātma-cintana [thought of oneself: self-contemplation or self-attentiveness], alone is giving oneself to God. Even though one places whatever amount of burden upon God, that entire amount he will bear. Since one paramēśvara śakti [supreme ruling power or power of God] is driving all kāryas [whatever needs or ought to be done or to happen], instead of we also yielding to it, why to be perpetually thinking, ‘it is necessary to do like this; it is necessary to do like that’? Though we know that the train is going bearing all the burdens, why should we who go travelling in it, instead of remaining happily leaving our small luggage placed on it [the train], suffer bearing it [our luggage] on our head?

What is called sukha [happiness, satisfaction, joy, ease, comfort or pleasantness] is only the svarūpa [the ‘own form’ or real nature] of ātmā [oneself]; sukha and ātma-svarūpa [one’s own real nature] are not different. Ātma-sukha [happiness that is oneself] alone exists; that alone is real. What is called sukha [happiness or satisfaction] is not found [obtained or available] in even one of the objects of the world. We think that happiness is obtained from them because of our avivēka [lack of judgement, discrimination or ability to distinguish one thing from another]. When the mind comes out [from ātma-svarūpa], it experiences duḥkha [dissatisfaction, discomfort, uneasiness, unpleasantness, unhappiness, distress, suffering, sorrow, sadness, pain or affliction]. In truth, whenever our thoughts [wishes or hopes] are fulfilled, it [the mind] turning back to its proper place [the heart, our real nature, which is the source from which it rose] experiences only ātma-sukha [happiness that is oneself]. Likewise at times of sleep, samādhi [a state of manōlaya or temporary dissolution of mind brought about by prāṇāyāma or other such yōga practices] and fainting, and when anything liked is obtained, and when destruction [damage, elimination or removal] occurs to anything disliked, the mind becoming antarmukham [inward facing] experiences only ātma-sukha. In this way the mind wanders about incessantly, going outside leaving oneself, and [again] turning back inside. At the foot of a tree the shade is pleasant [comfortable or delightful]. Outside the heat of the sun is severe [or harsh]. A person who is wandering outside is cooled [literally, obtains coolness or cooling] [by] going into the shade. After a short while emerging outside, [but] being unable to withstand [or bear] the severity of the heat, he again comes to the foot of the tree. In this way he remains, going from the shade into the sunshine, and going [back] from the sunshine into the shade. A person who does thus is an avivēki [someone lacking judgement, discrimination or ability to distinguish]. But a vivēki [someone who can judge, discriminate or distinguish] will not depart leaving the shade. Likewise the mind of the jñāni [one who is aware of one’s real nature] will not depart leaving brahman [that which alone exists, namely pure awareness, which is infinite happiness and one’s own real nature]. But the mind of the ajñāni [one who is not aware of one’s real nature] remains experiencing duḥkha [dissatisfaction or suffering] [by] roaming about in the world, and for a short while obtaining sukha [satisfaction or happiness] [by] returning to brahman. What is called the world is only thought [because like any world that we experience in a dream, what we experience as the world in this waking state is nothing but a series of perceptions, which are just thoughts or mental phenomena]. When the world disappears, that is, when thought ceases, the mind experiences happiness; when the world appears, it experiences duḥkha [dissatisfaction or suffering].

Just like in the mere presence of the sun, which rose without icchā [liking, wish or desire], saṁkalpa [desire, volition or intention] [or] yatna [effort or exertion], a sun-stone [sūryakānta, a gem that is supposed to emit fire or heat when exposed to the sun] emitting fire, a lotus blossoming, water evaporating, and people of the world commencing [or becoming engaged in] their respective kāryas [activities], doing [those kāryas] and ceasing [or subsiding], and [just like] in front of a magnet a needle moving, jīvas [sentient beings], who are subject to [or ensnared in] muttoṙil [the threefold function of God, namely the creation, sustenance and dissolution of the world] or pañcakṛtyas [the five functions of God, namely creation, sustenance, dissolution, concealment and grace], which happen by just [or nothing more than] the special nature of the presence of God, who is saṁkalpa rahitar [one who is devoid of any volition or intention], move [exert or engage in activity] and subside [cease being active, become still or sleep] in accordance with their respective karmas [that is, in accordance not only with their prārabdha karma or destiny, which impels them to do whatever actions are necessary in order for them to experience all the pleasant and unpleasant things that they are destined to experience, but also with their karma-vāsanās, their inclinations to think, speak and act in particular ways, which dispose them to make effort to experience pleasant things and to avoid experiencing unpleasant things]. Nevertheless, he [God] is not saṁkalpa sahitar [one who is connected with or possesses any volition or intention]; even one karma does not adhere to him [that is, he is not bound or affected in any way by any karma or action whatsoever]. That is like world-actions [the actions happening here on earth] not adhering to [or affecting] the sun, and [like] the qualities and defects of the other four elements [earth, water, air and fire] not adhering to the all-pervading space.

Since in every text [of advaita vēdānta] it is said that for attaining mukti [liberation] it is necessary to make the mind cease, after knowing that manōnigraha [restraint, subjugation or destruction of the mind] alone is the ultimate intention [aim or purpose] of [such] texts, there is no benefit [to be gained] by studying texts without limit. For making the mind cease it is necessary to investigate oneself [to see] who [one actually is], [but] instead [of doing so] how [can one see oneself by] investigating in texts? It is necessary to know oneself only by one’s own eye of jñāna [pure awareness]. Does [a person called] Raman need a mirror to know himself as Raman? ‘Oneself’ is within the pañca-kōśas [the ‘five sheaths’ that seem to cover and obscure what one actually is, namely the physical body, life, mind, intellect and will]; whereas texts are outside them. Therefore, investigating in texts [in order to know] oneself, whom it is necessary to investigate [by turning one’s attention within and thereby] setting aside [excluding, removing, giving up or separating from] all the pañca-kōśas, is useless. [By] investigating who is oneself who is in bondage, knowing one’s yathārtha svarūpa [actual own nature] alone is mukti [liberation]. The name ‘ātma-vicāra’ is only for [or refers only to] always keeping the mind on ātmā [oneself]; whereas dhyāna [meditation] is considering [thinking or imagining] oneself to be sat-cit-ānanda brahman [the one ultimate reality, which is existence-awareness-happiness]. At one time it will become necessary to forget all that one has learnt.

Just as one who needs to gather [or sweep] up and throw away rubbish [would derive] no benefit by examining [investigating or analysing] it, so one who needs to know oneself [will derive] no benefit by, instead of collectively rejecting all the tattvas, which are concealing oneself, calculating that they are this many and examining their qualities. It is necessary to consider the world [which is believed to be an expansion or manifestation of such tattvas] like a dream.

Besides the saying that waking is dīrgha [long lasting] and dream is kṣaṇika [momentary or lasting for only a short while], there is no other difference [between them]. To what extent all the vyavahāras [activities, affairs, transactions or events] that happen in waking seem to be real, to that extent even the vyavahāras that happen in dream seem at that time to be real. In dream the mind takes another body [to be itself]. In both waking and dream thoughts and names-and-forms [the phenomena that constitute the seemingly external world] occur in one time [or simultaneously].

There are not two minds, namely a good mind and a bad mind. Mind is only one. Only vāsanās [inclinations or propensities] are of two kinds, namely śubha [agreeable, virtuous or good] and aśubha [disagreeable, wicked, harmful or bad]. When mind is under the sway of śubha vāsanās it is said to be a good mind, and when it is under the sway of aśubha vāsanās a bad mind. However bad other people may appear to be, disliking them is not proper [or appropriate]. Likes and dislikes are both fit [for one] to dislike [spurn or renounce]. It is not appropriate to allow the mind [to dwell] excessively on worldly matters. To the extent possible, it is not appropriate to intrude in others’ affairs. All that one gives to others one is giving only to oneself. If one knew this truth, who indeed would remain without giving?

If oneself rises [or appears] [as ego or mind], everything rises [or appears]; if oneself subsides [disappears or ceases], everything subsides [disappears or ceases]. To whatever extent sinking low [subsiding or being humble] we behave [or conduct ourself], to that extent there is goodness [benefit or virtue]. If one is [continuously] restraining [curbing or subduing] mind, wherever one may be one can be [or let one be].

#################
"""
    
    def _load_qa_pairs(self, jsonl_path: str) -> list[dict]:
        """Load Q&A pairs from JSONL file."""
        qa_pairs = []
        path = Path(jsonl_path)
        
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    messages = entry.get("messages", [])
                    
                    # Extract Q&A pairs from messages
                    # Look for human -> assistant sequences
                    i = 0
                    while i < len(messages):
                        if messages[i]["role"] == "human":
                            question = messages[i]["content"]
                            # Look for next assistant response
                            answer_parts = []
                            j = i + 1
                            while j < len(messages) and messages[j]["role"] == "assistant":
                                answer_parts.append(messages[j]["content"])
                                j += 1
                            
                            if answer_parts:
                                answer = " ".join(answer_parts)
                                qa_pairs.append({
                                    "question": question,
                                    "answer": answer,
                                    "source_id": entry.get("id", f"line_{line_num}"),
                                })
                            i = j
                        else:
                            i += 1
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        return qa_pairs
    
    def _build_index(self) -> tuple:
        """Build FAISS index from questions."""
        if not self.qa_pairs:
            raise ValueError("No Q&A pairs loaded")
        
        questions = [pair["question"] for pair in self.qa_pairs]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(questions, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, questions
    
    def _retrieve_similar(self, query: str) -> list[dict]:
        """Retrieve top-k similar Q&A pairs."""
        # Embed query
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        k = min(self.top_k, len(self.qa_pairs))
        distances, indices = self.index.search(query_embedding, k)
        
        # Return Q&A pairs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.qa_pairs):
                results.append({
                    **self.qa_pairs[idx],
                    "distance": float(dist),
                })
        
        return results
    
    def _detect_local_model(self) -> str:
        """Auto-detect model from local vLLM API."""
        try:
            resp = self.http_client.get(f"{self.local_url}/models")
            resp.raise_for_status()
            data = resp.json()
            
            # Handle OpenAI-compatible format
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
            elif isinstance(data, list) and len(data) > 0:
                return data[0] if isinstance(data[0], str) else data[0].get("id", "unknown")
            else:
                logger.warning("Could not detect model, using default")
                return "unknown"
        except Exception as e:
            logger.warning(f"Could not detect local model: {e}")
            return "unknown"
    
    def _build_rag_prompt(self, query: str) -> str:
        """
        Build prompt with retrieved context.
        
        Structure:
        1. custom_prompt_prefix (user-editable text)
        2. Top-k retrieved Q&A pairs
        3. User query
        4. Response prompt
        """
        # Retrieve similar Q&A pairs
        retrieved = self._retrieve_similar(query)
        
        # Build context section
        context_parts = []
        for i, pair in enumerate(retrieved, 1):
            context_parts.append(f"Q{i}: {pair['question']}")
            context_parts.append(f"Bhagavan: {pair['answer']}")
            context_parts.append("")
        
        context_text = "\n".join(context_parts)
        
        # Build full prompt
        prompt_parts = []
        
        # Custom prefix (user can edit this via custom_prompt_prefix parameter)
        if self.custom_prompt_prefix:
            prompt_parts.append(self.custom_prompt_prefix)
            prompt_parts.append("")
        
        # Retrieved context
        if context_text:
            prompt_parts.append("Possibly relevant texts from Q/A sessions with Bhagavan:")
            prompt_parts.append(context_text)
        
        # User query
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("""#####
Respond in one or two short sentences in the spare, direct style of Ramana Maharshi. Rather than explaining or elaborating, respond as he did: often with a question that turns attention back to the questioner, or a simple statement pointing to what is already present. When correcting a misunderstanding about the inquiry itself, be clear and direct. Avoid intellectual discussion, speculation, or graduated instruction—the pointing is always to the Self that is already here. Let the brevity carry warmth rather than severity.
Examples from Talks with Sri Ramana Maharshi:
Q: How is one to realize the Self?
Bhagavan: Whose Self? Find out.
Q: How to know the 'Real I' as distinct from the 'false I'?
Bhagavan: Is there anyone who is not aware of himself? Each one knows, but yet does not know, the Self. A strange paradox.
Q: How long does it take a man to be reborn after death?
Bhagavan: Perhaps you are born now—why think of other births? The fact is there is neither birth nor death. Let him who is born think of death and palliatives for it.
Q: I want to know the state of liberation.
Bhagavan: You should know your present state first. What do you know of your present state? If you know the present state, knowledge of any other state will be clear.
################
Bhagavan:
""")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        query: str,
        max_new_tokens: int = 200,
        temperature: float = 0.5,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response to a query using RAG.
        
        Args:
            query: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated response text
        """
        # Build RAG prompt
        prompt = self._build_rag_prompt(query)
        
        # Call backend
        if self.backend == "local":
            return self._call_local(prompt, max_new_tokens, temperature, top_p, do_sample)
        elif self.backend == "openrouter":
            return self._call_openrouter(prompt, max_new_tokens, temperature, top_p, do_sample)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def generate_from_messages(
        self,
        messages: list[dict],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response from chat messages (ChatML format).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated response text
        """
        # Extract the last user message as query
        query = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                query = msg["content"]
                break
        
        if query is None:
            # Fallback: use all messages
            query = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return self.generate(
            query=query,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    
    def _call_local(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool) -> str:
        """Call local vLLM API."""
        try:
            resp = self.http_client.post(
                f"{self.local_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                }
            )
            resp.raise_for_status()
            j = resp.json()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Local API call failed: {e}")
            raise
    
    def _call_openrouter(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool) -> str:
        """Call OpenRouter API."""
        if not self.model:
            raise ValueError("Model name required for OpenRouter backend")
        
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")
        
        try:
            resp = self.http_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/your-repo",  # Optional
                    "X-Title": "Contemplative RAG",  # Optional
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ContemplativeRAGProvider")
    parser.add_argument("--jsonl", type=str, default="./ramana/Talks-parsed_reviewed.jsonl")
    parser.add_argument("--backend", type=str, default="local", choices=["local", "openrouter"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--query", type=str, default="What is the Self?")
    parser.add_argument("--top-k", type=int, default=3)
    
    args = parser.parse_args()
    
    provider = ContemplativeRAGProvider(
        jsonl_path=args.jsonl,
        backend=args.backend,
        model=args.model,
        top_k=args.top_k,
    )
    
    response = provider.generate(args.query)
    print(f"\nQuery: {args.query}")
    print(f"\nResponse: {response}")
